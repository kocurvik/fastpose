"""Fundamental matrix (7-point) on the GPU.

Solver and scorer are the simplest of all the ports - a 93-float workspace, at
most three models, and the plain Sampson scorer with no cheirality check
(poselib scores a matrix here, not a pose).

The refiner was the one genuinely blocked piece. Its state is the SVD
factorization F = U diag(1, sigma, 0) Vt, and `svd_init_state` used
`np.linalg.svd`, which is host-only - numba's LAPACK binding does not compile
for CUDA, and this is the only refiner that needs it. Rather than
reparametrize F or leave the LM on the CPU (which is O(n) per step and would
dominate at large n), `refiners/utils.py` now carries a one-sided Jacobi
`svd3` built by the same factory as the rest of the primitives, so both
backends run the same decomposition. It reconstructs and stays orthogonal to
~2e-15 on random, rank-2 and badly scaled 3x3 inputs.

Two smaller consequences of the factorized state:

- `state_size` is 19 (U, Vt, sigma) while `num_params` is 9, so the LM kernel's
  state and model really are different objects here - which is why the generic
  kernel keeps both and mirrors the model, not the state, for the per-point
  loops.
- the state's U and Vt are 3x3 but a shared array is flat, and `.reshape` does
  not compile in device code, so the thread-0 kernels copy the 18 doubles into
  per-thread local 3x3 arrays before calling the shared cores. That is O(1) per
  LM step against an O(n) accumulate.

Like the absolute-pose problems, the CPU estimator hands local optimization
the whole correspondence set (`LMFundamentalRefiner` has no relaxed-inlier
wrapper), so `relaxed_scale` is 0.0 - see cuda/problems/absolute.py for what
that convention means.
"""

from numba import cuda, float32, float64

from fastpose.cuda.backend import cuda_jit
from fastpose.cuda.problem import CudaProblem
from fastpose.cuda.problems.common import (PRIM32, PRIM64, SAMPSON32,
                                           SEVEN_POINT)

SAMPLE_SIZE = 7
NUM_PARAMS = 9
MAX_MODELS = 3
NUM_TANGENT = 7         # 3 left rotation + 3 right rotation + sigma
STATE_SIZE = 19         # U (9) | Vt (9) | sigma
DERIVED_SIZE = 9        # the flat fundamental matrix

# the 7-point scratch is 93 doubles per thread, so a full block fits easily
SOLVE_THREADS = 128

_sampson_residual = SAMPSON32['sampson_residual']
_sampson_point_jacobian = PRIM32['sampson_point_jacobian']
_mat3_mul = PRIM64['mat3_mul']
_rodrigues = PRIM64['rodrigues']
_svd_init_state_core = PRIM64['svd_init_state_core']
_factorized_f = PRIM64['factorized_f']
_epipolar_jacobian_basis_core = PRIM64['epipolar_jacobian_basis_core']

_solve_7pt_core = SEVEN_POINT['solve_7pt_core']


# ---------------------------------------------------------------------------
# minimal solver
# ---------------------------------------------------------------------------

@cuda.jit(cache=True)
def _solve_batch_kernel(data, params, samples, models, counts):
    # one thread per hypothesis; writes up to MAX_MODELS flat F matrices
    i = cuda.grid(1)
    if i >= samples.shape[0]:
        return

    A = cuda.local.array((7, 9), float64)
    f1 = cuda.local.array(9, float64)
    f2 = cuda.local.array(9, float64)
    tmp = cuda.local.array(9, float64)
    roots = cuda.local.array(3, float64)

    counts[i] = _solve_7pt_core(data, samples[i], models[i], A, f1, f2, tmp,
                                roots)


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
    # the model already is the matrix the Sampson residual reads
    for j in range(9):
        dm64[j] = model[j]
    return True


@cuda_jit(fastmath=True, inline=True)
def _score_point(dm32, m32, data, i, max_error_sq):
    # poselib's matrix overload of compute_sampson_msac_score: no cheirality
    # check, because there is no pose to check it on
    r2, den = _sampson_residual(dm32, data[0][i], data[1][i], data[2][i],
                                data[3][i])
    if den > float32(0.0) and r2 < max_error_sq * den:
        return r2 / den, True
    return float32(0.0), False


# ---------------------------------------------------------------------------
# local optimization
# ---------------------------------------------------------------------------

@cuda_jit(inline=True)
def _load_uvt(state, U, Vt):
    # the state's two 3x3 blocks, copied out of the flat shared array because
    # `.reshape` does not compile in device code
    for i in range(3):
        for j in range(3):
            U[i, j] = state[3 * i + j]
            Vt[i, j] = state[9 + 3 * i + j]


def build_refine_kernels(loss):
    weight_fn = loss.weight
    cost_fn = loss.cost

    @cuda_jit()
    def init_state(model_in, state):
        A = cuda.local.array((3, 3), float64)
        V = cuda.local.array((3, 3), float64)
        s = cuda.local.array(3, float64)
        return _svd_init_state_core(model_in, state, A, V, s)

    @cuda_jit()
    def state_to_model(state, model):
        U = cuda.local.array((3, 3), float64)
        Vt = cuda.local.array((3, 3), float64)
        _load_uvt(state, U, Vt)
        _factorized_f(U, Vt, state[18], model)

    @cuda_jit(inline=True)
    def model_derived(model, params, dm64):
        for j in range(9):
            dm64[j] = model[j]
        return True

    @cuda_jit()
    def jacobian_basis(model, state, params, dm64, B64, aux):
        U = cuda.local.array((3, 3), float64)
        Vt = cuda.local.array((3, 3), float64)
        S = cuda.local.array((3, 3), float64)
        SD = cuda.local.array((3, 3), float64)
        tmp = cuda.local.array((3, 3), float64)
        _load_uvt(state, U, Vt)
        _epipolar_jacobian_basis_core(U, Vt, state[18], B64, S, SD, tmp)

    @cuda_jit(fastmath=True, inline=True)
    def mask_point(dm32, st32, data, i, relaxed_sq):
        # `relaxed_sq <= 0` means "refine over every correspondence", which is
        # what the CPU fundamental refiner does inside RANSAC
        if relaxed_sq <= float32(0.0):
            return True
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
        # U <- U exp([d0:3]_x), Vt <- exp(-[d3:6]_x) Vt, sigma <- sigma + d6;
        # the same retraction `apply_rotation_step` applies on the CPU
        U = cuda.local.array((3, 3), float64)
        Vt = cuda.local.array((3, 3), float64)
        Rod = cuda.local.array((3, 3), float64)
        Out = cuda.local.array((3, 3), float64)
        _load_uvt(state, U, Vt)
        _rodrigues(delta[0], delta[1], delta[2], Rod)
        _mat3_mul(U, Rod, Out)
        for i in range(3):
            for j in range(3):
                state_new[3 * i + j] = Out[i, j]
        _rodrigues(-delta[3], -delta[4], -delta[5], Rod)
        _mat3_mul(Rod, Vt, Out)
        for i in range(3):
            for j in range(3):
                state_new[9 + 3 * i + j] = Out[i, j]
        state_new[18] = state[18] + delta[6]

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
    name='fundamental',
    sample_size=SAMPLE_SIZE,
    num_params=NUM_PARAMS,
    max_models=MAX_MODELS,
    data_width=4,
    num_tangent=NUM_TANGENT,
    state_size=STATE_SIZE,
    derived_size=DERIVED_SIZE,
    basis_width=9,
    aux_shape=(1, 1),
    scratch_shape=(2, 9),      # dsdF, then J
    min_inliers=-1,            # the CPU refiner has no inlier-count gate here
    relaxed_scale=0.0,         # "keep every correspondence"
    solve_batch=_solve_batch,
    score_kernels=(_prepare, _score_point),
    refine_factory=build_refine_kernels,
)
