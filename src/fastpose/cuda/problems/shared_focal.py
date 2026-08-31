"""Relative pose with one shared unknown focal length (6-point) on the GPU.

The refiner is the varying-focal one with the two focals tied together: the
state is [R | t | log f], `NUM_TANGENT` is 6 rather than 7, and the single
focal row of the tangent basis is the *sum* of the two rows
`log_focal_tangent_rows` produces - poselib's
SharedFocalRelativePoseEstimator does exactly that, because one parameter now
moves both. The derived form, the scorer and the per-point jacobian are
literally the same device functions as varying-focal's.

Like varying-focal, the two principal points ride in the driver's `params`
vector, and there is **no cheirality check** anywhere - poselib scores an F
here, so the scorer and the relaxed inlier mask both use the matrix overload.

The solve kernel is the outlier of the whole backend
--------------------------------------------------
It carries **~30.5 KB of per-thread local memory** (3665 doubles plus 144
int64), against ~6.9 KB for the 5-point solver and ~400 bytes for P3P. The
31x46 elimination template alone is 1426 doubles. Local memory is backed by
global memory and the driver reserves it for every resident thread, so the
block size here is a memory-footprint decision rather than a latency one:
`SOLVE_THREADS = 32` keeps the reservation to one warp's worth per block.

The solver is also deliberately **not** `fastmath` (see solvers/shared_focal.py
for the conditioning argument), which the `cuda_jit` shim honours by default -
do not add it here.
"""

import math

from numba import cuda, float32, float64, int64

from fastpose.cuda.backend import cuda_jit
from fastpose.cuda.problem import CudaProblem
from fastpose.cuda.problems.common import (PRIM32, PRIM64, SAMPSON32,
                                           SAMPSON64, SHARED_FOCAL)
from fastpose.refiners.shared_focal import MIN_INLIERS, NUM_TANGENT, STATE_SIZE
from fastpose.refiners.utils import LO_INLIER_SCALE

SAMPLE_SIZE = 6
NUM_PARAMS = 14
MAX_MODELS = 60
DERIVED_SIZE = 9        # the induced fundamental matrix

# ~30.5 KB of per-thread local memory; see the module docstring
SOLVE_THREADS = 32

_sampson_residual = SAMPSON32['sampson_residual']
_sampson_point_jacobian = PRIM32['sampson_point_jacobian']
_essential_from_pose = SAMPSON64['essential_from_pose']
_calibrate_epipolar_core = SAMPSON64['calibrate_epipolar_core']
_model_to_fundamental_core = SAMPSON64['model_to_fundamental_core']
_mat3_mul = PRIM64['mat3_mul']
_rodrigues = PRIM64['rodrigues']
_essential_tangent_rows_core = PRIM64['essential_tangent_rows_core']
_log_focal_tangent_rows = PRIM64['log_focal_tangent_rows']

_solve_shared_focal_core = SHARED_FOCAL['solve_shared_focal_core']


# ---------------------------------------------------------------------------
# minimal solver
# ---------------------------------------------------------------------------

@cuda.jit(cache=True)
def _solve_batch_kernel(data, params, samples, models, counts):
    # one thread per hypothesis; writes up to MAX_MODELS pose models
    # [R | t | f | f] into models[i] and their count into counts[i]
    i = cuda.grid(1)
    if i >= samples.shape[0]:
        return

    A = cuda.local.array((6, 9), float64)
    N = cuda.local.array((3, 9), float64)
    d = cuda.local.array(27, float64)
    sols = cuda.local.array((15, 3), float64)
    e = cuda.local.array(9, float64)
    Rbuf = cuda.local.array((2, 3, 3), float64)

    coeffs = cuda.local.array(280, float64)
    C = cuda.local.array((31, 46), float64)
    C12 = cuda.local.array((31, 15), float64)
    RR = cuda.local.array((23, 15), float64)
    AM = cuda.local.array((15, 15), float64)
    AMp = cuda.local.array((15, 15), float64)
    coef = cuda.local.array(16, float64)
    roots = cuda.local.array(15, float64)
    row = cuda.local.array(15, float64)
    tmp_row = cuda.local.array(15, float64)
    chain = cuda.local.array((16, 16), float64)
    lo_stack = cuda.local.array(64, float64)
    hi_stack = cuda.local.array(64, float64)
    A7 = cuda.local.array((7, 7), float64)
    b7 = cuda.local.array(7, float64)
    x7 = cuda.local.array(7, float64)
    vals = cuda.local.array(8, float64)
    sol = cuda.local.array(3, float64)
    degs = cuda.local.array(16, int64)
    slo_stack = cuda.local.array(64, int64)
    shi_stack = cuda.local.array(64, int64)

    counts[i] = _solve_shared_focal_core(
        data, samples[i], models[i], params[0], params[1], params[2],
        params[3], A, N, d, sols, e, Rbuf, coeffs, C, C12, RR, AM, AMp, coef,
        roots, row, tmp_row, chain, lo_stack, hi_stack, A7, b7, x7, vals, sol,
        degs, slo_stack, shi_stack)


def _solve_batch(data, params, samples, models, counts, stream=0):
    batch = samples.shape[0]
    blocks = (batch + SOLVE_THREADS - 1) // SOLVE_THREADS
    _solve_batch_kernel[blocks, SOLVE_THREADS, stream](
        data, params, samples, models, counts)


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------

@cuda_jit()
def _prepare(model, params, dm64):
    # F = K^-T ([t]_x R) K^-1 with one focal in both intrinsics. The model
    # keeps model[12] == model[13], so the varying-focal map is exactly right
    # and there is nothing shared-focal-specific to do here.
    e = cuda.local.array(9, float64)
    a = cuda.local.array(9, float64)
    return _model_to_fundamental_core(model, params[0], params[1], params[2],
                                      params[3], dm64, e, a)


@cuda_jit(fastmath=True, inline=True)
def _score_point(dm32, m32, data, i, max_error_sq):
    # poselib's matrix overload: no cheirality check here, deliberately
    r2, den = _sampson_residual(dm32, data[0][i], data[1][i], data[2][i],
                                data[3][i])
    if den > float32(0.0) and r2 < max_error_sq * den:
        return r2 / den, True
    return float32(0.0), False


# ---------------------------------------------------------------------------
# local optimization
# ---------------------------------------------------------------------------

def build_refine_kernels(loss):
    weight_fn = loss.weight
    cost_fn = loss.cost

    @cuda_jit()
    def init_state(model_in, state):
        nrm = math.sqrt(model_in[9] * model_in[9] + model_in[10] * model_in[10]
                        + model_in[11] * model_in[11])
        if nrm < 1e-12 or model_in[12] <= 0.0:
            return False
        inv = 1.0 / nrm
        for j in range(9):
            state[j] = model_in[j]
        state[9] = model_in[9] * inv
        state[10] = model_in[10] * inv
        state[11] = model_in[11] * inv
        state[12] = math.log(model_in[12])
        return True

    @cuda_jit(inline=True)
    def state_to_model(state, model):
        for j in range(12):
            model[j] = state[j]
        f = math.exp(state[12])
        model[12] = f
        model[13] = f

    @cuda_jit()
    def model_derived(model, params, dm64):
        e = cuda.local.array(9, float64)
        a = cuda.local.array(9, float64)
        return _model_to_fundamental_core(model, params[0], params[1],
                                          params[2], params[3], dm64, e, a)

    @cuda_jit()
    def jacobian_basis(model, state, params, dm64, B64, aux):
        # pose rows exactly as in varying-focal. The focal row is the *sum* of
        # the two single-focal rows: one parameter drives both intrinsics, so
        # dF/dlog f is the sum of the two partial derivatives.
        e = cuda.local.array(9, float64)
        dE = cuda.local.array((5, 9), float64)
        a = cuda.local.array(9, float64)
        row1 = cuda.local.array(9, float64)
        row2 = cuda.local.array(9, float64)
        pp1x = params[0]
        pp1y = params[1]
        pp2x = params[2]
        pp2y = params[3]
        inv = 1.0 / model[12]
        _essential_from_pose(model, e)
        # aux[0], aux[1] are the two translation basis vectors, which the
        # retraction below picks up at this same state
        _essential_tangent_rows_core(model, e, dE, aux[0], aux[1])
        for p in range(5):
            _calibrate_epipolar_core(dE[p], pp1x, pp1y, pp2x, pp2y, inv, inv,
                                     B64[p], a)
        _log_focal_tangent_rows(dm64, pp1x, pp1y, pp2x, pp2y, row1, row2)
        for j in range(9):
            B64[5, j] = row1[j] + row2[j]

    @cuda_jit(fastmath=True, inline=True)
    def mask_point(dm32, st32, data, i, relaxed_sq):
        # poselib selects the subset with the matrix overload of get_inliers
        # here, so there is no cheirality check
        r2, den = _sampson_residual(dm32, data[0][i], data[1][i], data[2][i],
                                    data[3][i])
        return den > float32(0.0) and r2 < relaxed_sq * den

    @cuda_jit(fastmath=True, inline=True)
    def cost_point(dm32, st32, data, i, max_error_sq):
        r2, den = _sampson_residual(dm32, data[0][i], data[1][i], data[2][i],
                                    data[3][i])
        if den > float32(0.0):
            return cost_fn(r2 / den, max_error_sq)
        return cost_fn(float32(1e18), max_error_sq)

    @cuda_jit(fastmath=True)
    def accum_point(dm32, st32, B32, data, i, max_error_sq, red, tid, scratch):
        s_i, ok = _sampson_point_jacobian(dm32, data[0][i], data[1][i],
                                          data[2][i], data[3][i], scratch[0])
        if not ok:
            return 0
        w = weight_fn(s_i * s_i, max_error_sq)
        if not (w > float32(0.0)):
            return 0
        for p in range(NUM_TANGENT):
            acc = float32(0.0)
            for jj in range(9):
                acc += scratch[0, jj] * B32[p, jj]
            scratch[1, p] = acc
        idx = 0
        for p in range(NUM_TANGENT):
            for q in range(p, NUM_TANGENT):
                red[tid, idx] += float64(w * scratch[1, p] * scratch[1, q])
                idx += 1
        base = idx
        for p in range(NUM_TANGENT):
            red[tid, base + p] += float64(w * scratch[1, p] * s_i)
        return 1

    @cuda_jit()
    def retract(state, delta, state_new, aux):
        Rcur = cuda.local.array((3, 3), float64)
        Rod = cuda.local.array((3, 3), float64)
        Rnew = cuda.local.array((3, 3), float64)
        for ii in range(3):
            for jj in range(3):
                Rcur[ii, jj] = state[3 * ii + jj]
        _rodrigues(delta[0], delta[1], delta[2], Rod)
        _mat3_mul(Rcur, Rod, Rnew)
        for ii in range(3):
            for jj in range(3):
                state_new[3 * ii + jj] = Rnew[ii, jj]
        t0 = state[9] + delta[3] * aux[0, 0] + delta[4] * aux[1, 0]
        t1 = state[10] + delta[3] * aux[0, 1] + delta[4] * aux[1, 1]
        t2 = state[11] + delta[3] * aux[0, 2] + delta[4] * aux[1, 2]
        inv = 1.0 / math.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
        state_new[9] = t0 * inv
        state_new[10] = t1 * inv
        state_new[11] = t2 * inv
        state_new[12] = state[12] + delta[5]

    return {
        'init_state': init_state,
        'state_to_model': state_to_model,
        'model_derived': model_derived,
        'jacobian_basis': jacobian_basis,
        'mask_point': mask_point,
        'cost_point': cost_point,
        'accum_point': accum_point,
        'retract': retract,
    }


PROBLEM = CudaProblem(
    name='shared-focal',
    sample_size=SAMPLE_SIZE,
    num_params=NUM_PARAMS,
    max_models=MAX_MODELS,
    data_width=4,
    num_tangent=NUM_TANGENT,
    state_size=STATE_SIZE,
    derived_size=DERIVED_SIZE,
    basis_width=9,
    aux_shape=(2, 3),          # the two translation basis vectors
    scratch_shape=(2, 9),      # dsdF, then J
    min_inliers=MIN_INLIERS,
    relaxed_scale=LO_INLIER_SCALE,
    solve_batch=_solve_batch,
    score_kernels=(_prepare, _score_point),
    refine_factory=build_refine_kernels,
    default_params=(0.0, 0.0, 0.0, 0.0),   # the two principal points
)
