"""Absolute pose with unknown focal length (P4Pf) on the GPU.

Same shape as the calibrated absolute-pose port, with three differences:

- the model carries the focal, so the derived form is the *focal-scaled* pose
  `focal_scale_pose` produces - after which the per-point residual is the
  plain calibrated one and is literally the same device function;
- the state optimizes `log f`, so `init_state`/`state_to_model` are not the
  identity and the retraction has a seventh component;
- the solver reuses the Sturm root isolation the 5-point solver already
  provides, which is why `build_p4pf_kernels` takes it as an argument rather
  than building a second copy.

Local memory for the solve kernel is ~4.7 KB per thread (454 doubles plus 137
int64 of scratch), close to the 5-point solver's, so it keeps the same small
block size.
"""

import math

from numba import cuda, float32, float64, int64

from fastpose.cuda.backend import SOLVE_THREADS_PER_BLOCK, cuda_jit
from fastpose.cuda.problem import CudaProblem
from fastpose.cuda.problems.common import (P4PF, PRIM64, REPROJ32, REPROJ64,
                                           REPROJ_FOCAL_JAC32)

SAMPLE_SIZE = 4
NUM_PARAMS = 13
MAX_MODELS = 1
NUM_TANGENT = 7
DERIVED_SIZE = 12       # the focal-scaled pose

SOLVE_THREADS = SOLVE_THREADS_PER_BLOCK

_reprojection_residual = REPROJ32['reprojection_residual']
_focal_scale_pose = REPROJ64['focal_scale_pose']
_focal_reprojection_point_jacobian = REPROJ_FOCAL_JAC32[
    'reprojection_point_jacobian']
_mat3_mul = PRIM64['mat3_mul']
_rodrigues = PRIM64['rodrigues']

_solve_p4pf_core = P4PF['solve_p4pf_core']


# ---------------------------------------------------------------------------
# minimal solver
# ---------------------------------------------------------------------------

@cuda.jit(cache=True)
def _solve_batch_kernel(data, params, samples, models, counts):
    # one thread per hypothesis; P4Pf emits at most one model [R | t | f]
    i = cuda.grid(1)
    if i >= samples.shape[0]:
        return

    d = cuda.local.array(48, float64)
    C = cuda.local.array((4, 8), float64)
    A = cuda.local.array((4, 4), float64)
    B = cuda.local.array((4, 4), float64)
    coeffs = cuda.local.array((3, 10), float64)
    solutions = cuda.local.array((3, 8), float64)
    Pm = cuda.local.array((3, 4), float64)
    xs = cuda.local.array((2, 4), float64)
    Xs = cuda.local.array((4, 3), float64)
    q_A = cuda.local.array((3, 3), float64)
    q_P = cuda.local.array((3, 7), float64)
    q_c = cuda.local.array(9, float64)
    chain = cuda.local.array((9, 9), float64)
    roots = cuda.local.array(8, float64)
    lo_stack = cuda.local.array(64, float64)
    hi_stack = cuda.local.array(64, float64)
    degs = cuda.local.array(9, int64)
    slo_stack = cuda.local.array(64, int64)
    shi_stack = cuda.local.array(64, int64)

    counts[i] = _solve_p4pf_core(
        data, samples[i], models[i], d, C, A, B, coeffs, solutions, Pm, xs,
        Xs, q_A, q_P, q_c, chain, roots, lo_stack, hi_stack, degs, slo_stack,
        shi_stack)


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
    # fold f into the pose once per model, in float64, before it is rounded
    return _focal_scale_pose(model, dm64)


@cuda_jit(fastmath=True, inline=True)
def _score_point(dm32, m32, data, i, max_error_sq):
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

    @cuda_jit()
    def init_state(model_in, state):
        if model_in[12] <= 0.0:
            return False
        for j in range(12):
            state[j] = model_in[j]
        state[12] = math.log(model_in[12])
        return True

    @cuda_jit(inline=True)
    def state_to_model(state, model):
        for j in range(12):
            model[j] = state[j]
        model[12] = math.exp(state[12])

    @cuda_jit(inline=True)
    def model_derived(model, params, dm64):
        return _focal_scale_pose(model, dm64)

    @cuda_jit(inline=True)
    def jacobian_basis(model, state, params, dm64, B64, aux):
        # the reprojection jacobian is analytic per point; the focal enters it
        # through the model mirror, not through a tangent basis
        pass

    @cuda_jit(fastmath=True, inline=True)
    def mask_point(dm32, st32, data, i, relaxed_sq):
        # `relaxed_sq <= 0` means "refine over every correspondence", which is
        # what the CPU refiner does inside RANSAC; the final polish passes 1.0
        if relaxed_sq <= float32(0.0):
            return True
        r2, zz = _reprojection_residual(dm32, data[2][i], data[3][i],
                                        data[4][i], data[0][i], data[1][i])
        return zz > float32(0.0) and r2 < relaxed_sq * zz * zz

    @cuda_jit(fastmath=True, inline=True)
    def cost_point(dm32, st32, data, i, max_error_sq):
        r2, zz = _reprojection_residual(dm32, data[2][i], data[3][i],
                                        data[4][i], data[0][i], data[1][i])
        if zz > float32(0.0):
            return cost_fn(r2 / (zz * zz), max_error_sq)
        return cost_fn(float32(1e18), max_error_sq)

    @cuda_jit(fastmath=True)
    def accum_point(dm32, st32, B32, data, i, max_error_sq, red, tid, scratch):
        # the focal jacobian reads the *unscaled* pose plus f, so it takes the
        # model mirror rather than the derived form
        rx, ry, ok = _focal_reprojection_point_jacobian(
            st32, data[2][i], data[3][i], data[4][i], data[0][i], data[1][i],
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
        state_new[12] = state[12] + delta[6]

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
    name='absolute-focal',
    sample_size=SAMPLE_SIZE,
    num_params=NUM_PARAMS,
    max_models=MAX_MODELS,
    data_width=5,
    num_tangent=NUM_TANGENT,
    state_size=13,
    derived_size=DERIVED_SIZE,
    basis_width=1,             # analytic jacobian; no basis to publish
    aux_shape=(1, 1),
    scratch_shape=(2, NUM_TANGENT),
    min_inliers=-1,            # the CPU refiner has no inlier-count gate here
    relaxed_scale=0.0,         # "keep every correspondence"
    solve_batch=_solve_batch,
    score_kernels=(_prepare, _score_point),
    refine_factory=build_refine_kernels,
)
