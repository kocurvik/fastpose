"""Homography (4-point DLT) on the GPU.

The simplest solver of all the ports - a 72-float workspace and exactly one
model per sample - paired with the only scorer here whose *derived form* is
larger than its model: the symmetric transfer error needs H^-1 as well as H,
so `derived_size` is 18 (see scorers/transfer.py for the layout, which both
backends read). Building the inverse once per model in float64 before it is
rounded is the whole reason the derived form exists; an adjugate formed in
float32 would lose digits the per-point loop never gets back.

The refiner state is the flat unit-norm H, so `state_size` and `num_params`
are both 9 - but they are still different objects: the state is normalized
onto the sphere and the tangent basis is built at *it*. There are 8 tangent
parameters, which puts `lm_threads_for` at 64 rather than 128 (NUM_ACC is 44
doubles per thread, and 128 of those would not fit the shared-memory budget).

Like the fundamental and absolute-pose problems, the CPU estimator hands local
optimization the whole correspondence set (`LMHomographyRefiner` has no
relaxed-inlier wrapper), so `relaxed_scale` is 0.0 - see
cuda/problems/absolute.py for what that convention means.
"""

from numba import cuda, float32, float64

from fastpose.cuda.backend import cuda_jit
from fastpose.cuda.problem import CudaProblem
from fastpose.cuda.problems.common import (FOUR_POINT, HOMOG32, HOMOG64,
                                           TRANSFER32, TRANSFER64)

SAMPLE_SIZE = 4
NUM_PARAMS = 9
MAX_MODELS = 1
NUM_TANGENT = 8         # the tangent space of the unit sphere in R^9
STATE_SIZE = 9          # the flat H, unit Frobenius norm
DERIVED_SIZE = 18       # H (9) followed by H^-1 (9)
NUM_RESIDUALS = 4       # two forward transfer components, then two backward

# the 4-point scratch is 72 doubles per thread, so a full block fits easily
SOLVE_THREADS = 128

_symmetric_transfer_residual = TRANSFER32['symmetric_transfer_residual']
_homography_derived = TRANSFER64['homography_derived']
_transfer_point_jacobian = HOMOG32['transfer_point_jacobian']
_sphere_init_state = HOMOG64['sphere_init_state']
_sphere_state_to_model = HOMOG64['sphere_state_to_model']
_sphere_tangent_basis_core = HOMOG64['sphere_tangent_basis_core']
_sphere_retract_core = HOMOG64['sphere_retract_core']

_solve_h4p_core = FOUR_POINT['solve_h4p_core']


# ---------------------------------------------------------------------------
# minimal solver
# ---------------------------------------------------------------------------

@cuda.jit(cache=True)
def _solve_batch_kernel(data, params, samples, models, counts):
    # one thread per hypothesis; writes at most one flat H
    i = cuda.grid(1)
    if i >= samples.shape[0]:
        return

    A = cuda.local.array((8, 9), float64)

    counts[i] = _solve_h4p_core(data, samples[i], models[i], A)


def _solve_batch(data, params, samples, models, counts, stream=0):
    batch = samples.shape[0]
    blocks = (batch + SOLVE_THREADS - 1) // SOLVE_THREADS
    _solve_batch_kernel[blocks, SOLVE_THREADS, stream](
        data, params, samples, models, counts)


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------

@cuda_jit(inline=True)
def _prepare(model, params, dm64):
    # H in 0..8 and H^-1 in 9..17; False for a singular or non-finite H, which
    # the CPU scorer reports as (1e300, 0)
    return _homography_derived(model, dm64)


@cuda_jit(fastmath=True, inline=True)
def _score_point(dm32, m32, data, i, max_error_sq):
    # truncated symmetric transfer error, with the inlier test kept in its
    # unnormalized form so it agrees point for point with the CPU scorer
    num, den = _symmetric_transfer_residual(dm32, data[0][i], data[1][i],
                                            data[2][i], data[3][i])
    if den > float32(0.0) and num < max_error_sq * den:
        return num / den, True
    return float32(0.0), False


# ---------------------------------------------------------------------------
# local optimization
# ---------------------------------------------------------------------------

def build_refine_kernels(loss):
    weight_fn = loss.weight
    cost_fn = loss.cost

    @cuda_jit()
    def init_state(model_in, state):
        return _sphere_init_state(model_in, state)

    @cuda_jit()
    def state_to_model(state, model):
        _sphere_state_to_model(state, model)

    @cuda_jit(inline=True)
    def model_derived(model, params, dm64):
        return _homography_derived(model, dm64)

    @cuda_jit()
    def jacobian_basis(model, state, params, dm64, B64, aux):
        # the Householder basis of the tangent space at the *state*, which is
        # the unit-norm H; `w` is the scratch the shared core wants pre-shaped
        w = cuda.local.array(9, float64)
        _sphere_tangent_basis_core(state, B64, w)

    @cuda_jit(fastmath=True, inline=True)
    def mask_point(dm32, st32, data, i, relaxed_sq):
        # `relaxed_sq <= 0` means "refine over every correspondence", which is
        # what the CPU homography refiner does inside RANSAC
        if relaxed_sq <= float32(0.0):
            return True
        num, den = _symmetric_transfer_residual(dm32, data[0][i], data[1][i],
                                                data[2][i], data[3][i])
        return den > float32(0.0) and num < relaxed_sq * den

    @cuda_jit(fastmath=True, inline=True)
    def cost_point(dm32, st32, data, i, max_error_sq):
        num, den = _symmetric_transfer_residual(dm32, data[0][i], data[1][i],
                                                data[2][i], data[3][i])
        if den > float32(0.0):
            return cost_fn(num / den, max_error_sq)
        return cost_fn(float32(1e18), max_error_sq)

    @cuda_jit(fastmath=True)
    def accum_point(dm32, st32, B32, data, i, max_error_sq, red, tid, scratch):
        # scratch rows 0..3 hold dr, rows 4..7 the four tangent-space jacobian
        # rows and row 8 the four residuals
        e2, ok = _transfer_point_jacobian(dm32, data[0][i], data[1][i],
                                          data[2][i], data[3][i], scratch,
                                          scratch[8])
        if not ok:
            return 0
        w = weight_fn(e2, max_error_sq)
        if not (w > float32(0.0)):
            return 0
        for k in range(NUM_RESIDUALS):
            for p in range(NUM_TANGENT):
                acc = float32(0.0)
                for jj in range(9):
                    acc += scratch[k, jj] * B32[p, jj]
                scratch[4 + k, p] = acc
        # sum the four outer products before touching shared memory: 44
        # reduction slots updated once per point rather than four times
        idx = 0
        for p in range(NUM_TANGENT):
            for q in range(p, NUM_TANGENT):
                s = float32(0.0)
                for k in range(NUM_RESIDUALS):
                    s += scratch[4 + k, p] * scratch[4 + k, q]
                red[tid, idx] += float64(w * s)
                idx += 1
        base = idx
        for p in range(NUM_TANGENT):
            s = float32(0.0)
            for k in range(NUM_RESIDUALS):
                s += scratch[4 + k, p] * scratch[8, k]
            red[tid, base + p] += float64(w * s)
        return NUM_RESIDUALS

    @cuda_jit()
    def retract(state, delta, state_new, aux):
        # h <- (h + B^T delta) / ||.||, with B rebuilt from h exactly as
        # `_apply_step` rebuilds it on the CPU
        B = cuda.local.array((NUM_TANGENT, 9), float64)
        w = cuda.local.array(9, float64)
        _sphere_retract_core(state, delta, state_new, B, w)

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
    name='homography',
    sample_size=SAMPLE_SIZE,
    num_params=NUM_PARAMS,
    max_models=MAX_MODELS,
    data_width=4,
    num_tangent=NUM_TANGENT,
    state_size=STATE_SIZE,
    derived_size=DERIVED_SIZE,
    basis_width=9,
    aux_shape=(1, 1),
    scratch_shape=(9, 9),      # dr (4), then the four jacobian rows, then res
    min_inliers=-1,            # the CPU refiner has no inlier-count gate here
    relaxed_scale=0.0,         # "keep every correspondence"
    solve_batch=_solve_batch,
    score_kernels=(_prepare, _score_point),
    refine_factory=build_refine_kernels,
)
