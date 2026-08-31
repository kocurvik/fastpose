"""Calibrated absolute pose (P3P) on the GPU.

Solver, scorer and LM all come from the same factories the CPU path uses:
`build_p3p_kernels` (solvers/p3p.py), `build_reprojection_point_kernels`
(scorers/reprojection.py) and `build_reprojection_primitives`
(refiners/absolute.py), instantiated with `cuda_jit`.

The one thing worth spelling out is the LM subset. The epipolar refiners
restrict local optimization to the relaxed-threshold inliers, the way
poselib's `refine_model` does - but `LMAbsolutePoseRefiner` has no such
wrapper: the CPU estimator hands the RANSAC-internal refinement the *whole*
correspondence set, and only the final polish sees an inlier-only one. The
generic LM kernel always applies a mask, so this problem passes
`relaxed_scale = 0.0`, which `mask_point` reads as "keep everything" and is
what makes the GPU refinement the same minimization the CPU one runs. The
final polish still goes through `relaxed_scale = 1.0` and gets the inlier
subset.

Per-thread local memory for the solve kernel is ~400 bytes, two orders below
the 5-point solver's, so the block can be much larger there.
"""

from numba import cuda, float32, float64

from fastpose.cuda.backend import cuda_jit
from fastpose.cuda.problem import CudaProblem
from fastpose.cuda.problems.common import (P3P, PRIM64, REPROJ32,
                                           REPROJ_JAC32)

SAMPLE_SIZE = 3
NUM_PARAMS = 12
MAX_MODELS = 4
NUM_TANGENT = 6
DERIVED_SIZE = 12       # the pose itself; no matrix to form

# the P3P scratch is ~50 doubles per thread, so a full block fits easily
SOLVE_THREADS = 128

_reprojection_residual = REPROJ32['reprojection_residual']
_reprojection_point_jacobian = REPROJ_JAC32['reprojection_point_jacobian']
_mat3_mul = PRIM64['mat3_mul']
_rodrigues = PRIM64['rodrigues']

_solve_p3p_core = P3P['solve_p3p_core']


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

    xs = cuda.local.array((3, 3), float64)
    Xs = cuda.local.array((3, 3), float64)
    X01 = cuda.local.array(3, float64)
    X02 = cuda.local.array(3, float64)
    XXinv = cuda.local.array((3, 3), float64)
    C = cuda.local.array((3, 3), float64)
    pq = cuda.local.array((2, 3), float64)
    taus = cuda.local.array(2, float64)

    counts[i] = _solve_p3p_core(data, samples[i], models[i], xs, Xs, X01, X02,
                                XXinv, C, pq, taus)


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
    # nothing to derive: the reprojection residual reads the pose directly
    for j in range(12):
        dm64[j] = model[j]
    return True


@cuda_jit(fastmath=True, inline=True)
def _score_point(dm32, m32, data, i, max_error_sq):
    # truncated squared reprojection error; a point behind the camera is an
    # outlier. The inlier test stays unnormalized so an outlier never divides,
    # which is what makes this agree with the CPU scorer point for point.
    r2, zz = _reprojection_residual(dm32, data[2][i], data[3][i], data[4][i],
                                    data[0][i], data[1][i])
    zz_sq = zz * zz
    if zz > float32(0.0) and r2 < max_error_sq * zz_sq:
        return r2 / zz_sq, True
    return float32(0.0), False


# ---------------------------------------------------------------------------
# local optimization
# ---------------------------------------------------------------------------

def build_refine_kernels(loss):
    weight_fn = loss.weight
    cost_fn = loss.cost

    @cuda_jit(inline=True)
    def init_state(model_in, state):
        for j in range(12):
            state[j] = model_in[j]
        return True

    @cuda_jit(inline=True)
    def state_to_model(state, model):
        for j in range(12):
            model[j] = state[j]

    @cuda_jit(inline=True)
    def model_derived(model, params, dm64):
        for j in range(12):
            dm64[j] = model[j]
        return True

    @cuda_jit(inline=True)
    def jacobian_basis(model, state, params, dm64, B64, aux):
        # the reprojection jacobian is analytic per point; there is no tangent
        # basis to publish
        pass

    @cuda_jit(fastmath=True, inline=True)
    def mask_point(dm32, st32, data, i, relaxed_sq):
        # `relaxed_sq <= 0` means "refine over every correspondence", which is
        # what the CPU absolute-pose refiner does inside RANSAC; the final
        # polish passes 1.0 and gets the model's own inlier set
        if relaxed_sq <= float32(0.0):
            return True
        r2, zz = _reprojection_residual(dm32, data[2][i], data[3][i],
                                        data[4][i], data[0][i], data[1][i])
        return zz > float32(0.0) and r2 < relaxed_sq * zz * zz

    @cuda_jit(fastmath=True, inline=True)
    def cost_point(dm32, st32, data, i, max_error_sq):
        # 1e18 behind the camera, matching build_reprojection_cost; for
        # TruncatedLoss that is exactly the truncation constant
        r2, zz = _reprojection_residual(dm32, data[2][i], data[3][i],
                                        data[4][i], data[0][i], data[1][i])
        if zz > float32(0.0):
            return cost_fn(r2 / (zz * zz), max_error_sq)
        return cost_fn(float32(1e18), max_error_sq)

    @cuda_jit(fastmath=True)
    def accum_point(dm32, st32, B32, data, i, max_error_sq, red, tid, scratch):
        # two scalar residuals per point; scratch rows 0 and 1 are the two
        # jacobian rows over the 6 tangent parameters
        rx, ry, ok = _reprojection_point_jacobian(
            dm32, data[2][i], data[3][i], data[4][i], data[0][i], data[1][i],
            scratch[0], scratch[1])
        if not ok:
            return 0
        w = weight_fn(rx * rx + ry * ry, max_error_sq)
        if not (w > float32(0.0)):
            return 0
        idx = 0
        for p in range(NUM_TANGENT):
            for q in range(p, NUM_TANGENT):
                red[tid, idx] += float64(
                    w * (scratch[0, p] * scratch[0, q]
                         + scratch[1, p] * scratch[1, q]))
                idx += 1
        base = idx
        for p in range(NUM_TANGENT):
            red[tid, base + p] += float64(
                w * (scratch[0, p] * rx + scratch[1, p] * ry))
        return 2

    @cuda_jit()
    def retract(state, delta, state_new, aux):
        # R_new = R exp([w]_x), t_new = t + dt
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
        state_new[9] = state[9] + delta[3]
        state_new[10] = state[10] + delta[4]
        state_new[11] = state[11] + delta[5]

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
    name='absolute',
    sample_size=SAMPLE_SIZE,
    num_params=NUM_PARAMS,
    max_models=MAX_MODELS,
    data_width=5,
    num_tangent=NUM_TANGENT,
    state_size=12,
    derived_size=DERIVED_SIZE,
    basis_width=1,             # analytic jacobian; no basis to publish
    aux_shape=(1, 1),
    scratch_shape=(2, NUM_TANGENT),
    min_inliers=-1,            # the CPU refiner has no inlier-count gate here
    relaxed_scale=0.0,         # "keep every correspondence"; see the docstring
    solve_batch=_solve_batch,
    score_kernels=(_prepare, _score_point),
    refine_factory=build_refine_kernels,
)
