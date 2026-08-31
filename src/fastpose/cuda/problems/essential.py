"""Calibrated relative pose (5-point) on the GPU.

The reference port. None of the math is reimplemented here: the solver comes
from `build_five_point_kernels` in `solvers/essential.py`, the per-point
Sampson residual, the pose-to-E map and the cheirality test from
`build_sampson_point_kernels`, and the retraction, tangent basis and Sampson
jacobian from `build_refiner_primitives` - all instantiated with `cuda_jit`
instead of `njit`. What is written here is the composition: which scratch the
solve kernel allocates, and how the block-parallel scorer and LM see the
per-point kernels.

Solve: one thread per hypothesis. The solver is scalar, branchy, serial code -
there is no useful parallelism *inside* one minimal sample - so the
parallelism is purely across hypotheses, which is exactly what a RANSAC round
has thousands of. Scratch lives in `cuda.local.array`, per-thread storage
backed by global memory that CUDA interleaves across a warp, so thread-uniform
accesses (which these are: every thread walks the same elimination in
lockstep) coalesce without any manual layout work. The footprint is ~6.9 KB of
local memory per thread, dominated by the 10x20 constraint matrix and the
11x11 Sturm chain.
"""

import math

from numba import cuda, float32, float64, int64

from fastpose.cuda.backend import SOLVE_THREADS_PER_BLOCK, cuda_jit
from fastpose.cuda.problem import CudaProblem
from fastpose.cuda.problems.common import (FIVE_POINT, PRIM32, PRIM64,
                                           SAMPSON32, SAMPSON64)
from fastpose.refiners.essential import MIN_INLIERS
from fastpose.refiners.utils import LO_INLIER_SCALE
from fastpose.scorers.sampson import MIN_DEPTH

SAMPLE_SIZE = 5
NUM_PARAMS = 12
MAX_MODELS = 40
NUM_TANGENT = 5
DERIVED_SIZE = 9        # the flat essential matrix

_sampson_residual = SAMPSON32['sampson_residual']
_cheirality_ok = SAMPSON32['cheirality_ok']
_essential_from_pose = SAMPSON64['essential_from_pose']
_mat3_mul = PRIM64['mat3_mul']
_rodrigues = PRIM64['rodrigues']
_essential_tangent_rows_core = PRIM64['essential_tangent_rows_core']
_sampson_point_jacobian = PRIM32['sampson_point_jacobian']
_solve_5pt_core = FIVE_POINT['solve_5pt_core']


# ---------------------------------------------------------------------------
# minimal solver
# ---------------------------------------------------------------------------

@cuda.jit(cache=True)
def _solve_batch_kernel(data, params, samples, models, counts):
    # one thread per hypothesis; writes up to MAX_MODELS pose models [R | t]
    # into models[i] and their count into counts[i]
    i = cuda.grid(1)
    if i >= samples.shape[0]:
        return

    A = cuda.local.array((5, 9), float64)
    N = cuda.local.array((4, 9), float64)
    M = cuda.local.array((10, 20), float64)
    G = cuda.local.array((10, 10), float64)
    EEt = cuda.local.array((3, 3, 10), float64)
    tr = cuda.local.array(10, float64)
    ct = cuda.local.array((3, 10), float64)
    av = cuda.local.array(4, float64)
    bv = cuda.local.array(4, float64)
    coef = cuda.local.array(11, float64)
    row = cuda.local.array(10, float64)
    tmp_row = cuda.local.array(10, float64)
    chain = cuda.local.array((11, 11), float64)
    roots = cuda.local.array(10, float64)
    v = cuda.local.array(10, float64)
    lo_stack = cuda.local.array(64, float64)
    hi_stack = cuda.local.array(64, float64)
    e = cuda.local.array(9, float64)
    Rbuf = cuda.local.array((2, 3, 3), float64)
    degs = cuda.local.array(11, int64)
    slo_stack = cuda.local.array(64, int64)
    shi_stack = cuda.local.array(64, int64)

    counts[i] = _solve_5pt_core(
        data, samples[i], models[i],
        A, N, M, G, EEt, tr, ct, av, bv, coef, row, tmp_row, chain, roots, v,
        lo_stack, hi_stack, e, Rbuf, degs, slo_stack, shi_stack)


def _solve_batch(data, params, samples, models, counts, stream=0):
    batch = samples.shape[0]
    blocks = (batch + SOLVE_THREADS_PER_BLOCK - 1) // SOLVE_THREADS_PER_BLOCK
    _solve_batch_kernel[blocks, SOLVE_THREADS_PER_BLOCK, stream](
        data, params, samples, models, counts)


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------

@cuda_jit(inline=True)
def _prepare(model, params, dm64):
    _essential_from_pose(model, dm64)
    return True


@cuda_jit(fastmath=True, inline=True)
def _score_point(dm32, m32, data, i, max_error_sq):
    # poselib's compute_sampson_msac_score(CameraPose, ...): the truncated
    # Sampson error of E = [t]_x R, with a point counted as an inlier only if
    # it also triangulates in front of both cameras. The inlier test stays in
    # its unnormalized form, so an outlier never divides - which is what makes
    # this agree with the CPU scorer point for point.
    x = data[0][i]
    y = data[1][i]
    xp = data[2][i]
    yp = data[3][i]
    r2, den = _sampson_residual(dm32, x, y, xp, yp)
    if (den > float32(0.0) and r2 < max_error_sq * den
            and _cheirality_ok(m32, x, y, xp, yp, float32(MIN_DEPTH))):
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
        # the model already is the pose; normalize t so the translation
        # retraction stays on the unit sphere
        nrm = math.sqrt(model_in[9] * model_in[9] + model_in[10] * model_in[10]
                        + model_in[11] * model_in[11])
        if nrm < 1e-12:
            return False
        inv = 1.0 / nrm
        for j in range(9):
            state[j] = model_in[j]
        for j in range(9, 12):
            state[j] = model_in[j] * inv
        return True

    @cuda_jit(inline=True)
    def state_to_model(state, model):
        for j in range(12):
            model[j] = state[j]

    @cuda_jit(inline=True)
    def model_derived(model, params, dm64):
        _essential_from_pose(model, dm64)
        return True

    @cuda_jit()
    def jacobian_basis(model, state, params, dm64, B64, aux):
        # rows 0..4 of B are dE/dtheta; the two translation basis vectors land
        # in aux, where the retraction below picks them up at the same state
        _essential_tangent_rows_core(model, dm64, B64, aux[0], aux[1])

    @cuda_jit(fastmath=True, inline=True)
    def mask_point(dm32, st32, data, i, relaxed_sq):
        # poselib's CameraPose overload of get_inliers: the Sampson test plus
        # the cheirality check
        x = data[0][i]
        y = data[1][i]
        xp = data[2][i]
        yp = data[3][i]
        r2, den = _sampson_residual(dm32, x, y, xp, yp)
        return (den > float32(0.0) and r2 < relaxed_sq * den
                and _cheirality_ok(st32, x, y, xp, yp, float32(MIN_DEPTH)))

    @cuda_jit(fastmath=True, inline=True)
    def cost_point(dm32, st32, data, i, max_error_sq):
        # 1e18 for a degenerate denominator, matching build_sampson_cost; for
        # TruncatedLoss both branches reduce to the division-free form the CPU
        # scorer uses. Residual in float32, accumulated by the caller in
        # float64.
        r2, den = _sampson_residual(dm32, data[0][i], data[1][i], data[2][i],
                                    data[3][i])
        if den > float32(0.0):
            return cost_fn(r2 / den, max_error_sq)
        return cost_fn(float32(1e18), max_error_sq)

    @cuda_jit(fastmath=True)
    def accum_point(dm32, st32, B32, data, i, max_error_sq, red, tid, scratch):
        # per-point jacobian scratch in float32: this is the O(n) inner loop,
        # and the products below land in the float64 `red`
        s_i, ok = _sampson_point_jacobian(dm32, data[0][i], data[1][i],
                                          data[2][i], data[3][i], scratch[0])
        if not ok:
            return 0
        w = weight_fn(s_i * s_i, max_error_sq)
        # a zero weight drops the point entirely; for TruncatedLoss that is
        # exactly the hard cutoff
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
        # R_new = R exp([w]_x)
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
        # t_new = normalize(t + d3 b1 + d4 b2); b1/b2 were written by
        # jacobian_basis at this same state
        t0 = state[9] + delta[3] * aux[0, 0] + delta[4] * aux[1, 0]
        t1 = state[10] + delta[3] * aux[0, 1] + delta[4] * aux[1, 1]
        t2 = state[11] + delta[3] * aux[0, 2] + delta[4] * aux[1, 2]
        inv = 1.0 / math.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
        state_new[9] = t0 * inv
        state_new[10] = t1 * inv
        state_new[11] = t2 * inv

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
    name='essential',
    sample_size=SAMPLE_SIZE,
    num_params=NUM_PARAMS,
    max_models=MAX_MODELS,
    data_width=4,
    num_tangent=NUM_TANGENT,
    state_size=12,
    derived_size=DERIVED_SIZE,
    basis_width=9,
    aux_shape=(2, 3),          # the two translation basis vectors
    scratch_shape=(2, 9),      # dsdF, then J
    min_inliers=MIN_INLIERS,
    relaxed_scale=LO_INLIER_SCALE,
    solve_batch=_solve_batch,
    score_kernels=(_prepare, _score_point),
    refine_factory=build_refine_kernels,
)
