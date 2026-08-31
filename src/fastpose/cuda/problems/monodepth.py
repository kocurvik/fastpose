"""Relative pose with monocular depth priors, all four variants, on the GPU.

One module, four `CudaProblem` objects, because the four share everything but
a handful of compile-time flags:

    PROBLEM_CALIBRATED     P3P on depth-induced 3D points   NUM_TANGENT = 7
    PROBLEM_SHIFT          + one depth shift per image      NUM_TANGENT = 9
    PROBLEM_SHARED_FOCAL   + one focal for both cameras     NUM_TANGENT = 8
    PROBLEM_VARYING_FOCAL  + one focal per camera           NUM_TANGENT = 9

What is new here relative to the other six ports
------------------------------------------------
**The LM minimizes a hybrid cost**, not a single residual family: a truncated
Sampson term weighted by `weight_sampson`, plus a truncated *symmetric*
reprojection term (camera 1 -> 2 and camera 2 -> 1 through the depth-induced
3D points) scaled by `scale_reproj`. So `accum_point` contributes up to three
residuals per correspondence - 2 + 2 scalar reprojection residuals and 1
Sampson - and `scratch_shape` has to hold both families' jacobian rows at
once: `(4, 9)` for dsdF, J, J0 and J1.

`scale_reproj` and `weight_sampson` are the two entries of the `params`
vector (see cuda/problem.py). They sit at indices 6 and 7 of the CPU `data`
tuple, which is why `build_monodepth_kernels` splits the six coordinate
columns off into a 6-tuple.

They also need to reach the *per-point* kernels, and those do not receive
`params` - their signature is `(dm32, st32, data, i, max_error_sq)`. So the
**derived form carries them**: `derived_size` is 11 rather than 9, the
epipolar matrix in 0..8 and the two weights in 9 and 10, written from
`params` by `prepare` and `model_derived`, which do get it. The generic
kernels mirror the whole thing to float32 for free.

**No cheirality check anywhere**, deliberately: these models carry a *metric*
translation rather than a unit one, so `MIN_DEPTH` would mean something
different than it does for the calibrated relative-pose scorer. The CPU
scorers do not apply one and neither do these.

**`relaxed_scale = 0.0`.** `LMMonoDepth*PoseRefiner` has no relaxed-inlier
wrapper - the estimators hand the RANSAC-internal refinement the whole
correspondence set - so `mask_point` keeps everything and `min_inliers` is
-1, exactly as for absolute pose and fundamental
(see cuda/problems/absolute.py for what that convention means).

The per-point maths is not written here: the two reprojection kernels come
from `build_monodepth_reproj_kernels` and the Sampson pieces from the same
`sampson_residual` / `sampson_point_jacobian` every other problem uses, all
instantiated once in cuda/problems/common.py.
"""

import math

from numba import cuda, float32, float64, int64

from fastpose.cuda.backend import cuda_jit
from fastpose.cuda.problem import CudaProblem
from fastpose.cuda.problems.common import (MONO64, MONO_REPROJ32,
                                           MONO_RESID32, P3P, P4PF, PRIM32,
                                           PRIM64, REAL_ROOTS_STURM,
                                           SAMPSON32, SAMPSON64, SHARED_FOCAL)
from fastpose.refiners.monodepth import MODEL_SIZE, STATE_SIZE
from fastpose.solvers.monodepth import build_monodepth_kernels

SAMPLE_SIZE = 3
NUM_PARAMS = MODEL_SIZE     # 15: [R | t | scale | shift1 | shift2] or
                            #     [R | t | f1 | f2 | scale]
# E (calibrated) or F (focal) in 0..8, then scale_reproj and weight_sampson;
# see the module docstring for why the weights ride here
DERIVED_SIZE = 11
W_SCALE_REPROJ = 9
W_SAMPSON = 10

# per-thread local memory: 234 doubles for the shift solver, 293 for
# shared-focal, well under the 5-point solver's ~6.9 KB, so a full block fits
SOLVE_THREADS = 128

_sampson_residual = SAMPSON32['sampson_residual']
_sampson_point_jacobian = PRIM32['sampson_point_jacobian']
_essential_from_pose32 = SAMPSON32['essential_from_pose']
_essential_from_pose = SAMPSON64['essential_from_pose']
_mat3_mul = PRIM64['mat3_mul']
_rodrigues = PRIM64['rodrigues']
_focal_fundamental = MONO64['focal_fundamental']
_essential_tangent_rows = MONO64['essential_tangent_rows']
_focal_tangent_rows = MONO64['focal_tangent_rows']

_GPU = build_monodepth_kernels(cuda_jit, REAL_ROOTS_STURM, P3P['p3p_impl'],
                               P4PF['solve_3xn_neg'],
                               P4PF['project_rotation'],
                               SHARED_FOCAL['charpoly_danilevsky_n'])


# ---------------------------------------------------------------------------
# minimal solvers - one kernel per variant, each allocating its own scratch
# ---------------------------------------------------------------------------

_solve_shift_core = _GPU['solve_monodepth_shift_3pt_core']
_solve_p3p_core = _GPU['solve_monodepth_p3p_core']
_solve_shared_focal_core = _GPU['solve_monodepth_shared_focal_3pt_core']
_solve_varying_focal_core = _GPU['solve_monodepth_varying_focal_3pt_core']


@cuda.jit(cache=True)
def _solve_shift_kernel(data, params, samples, models, counts):
    i = cuda.grid(1)
    if i >= samples.shape[0]:
        return
    d = cuda.local.array(18, float64)
    coeffs = cuda.local.array(18, float64)
    C0 = cuda.local.array((3, 3), float64)
    C1 = cuda.local.array((3, 3), float64)
    coef = cuda.local.array(5, float64)
    chain = cuda.local.array((5, 5), float64)
    roots = cuda.local.array(4, float64)
    lo_stack = cuda.local.array(64, float64)
    hi_stack = cuda.local.array(64, float64)
    A = cuda.local.array((3, 3), float64)
    B = cuda.local.array((3, 3), float64)
    degs = cuda.local.array(5, int64)
    slo_stack = cuda.local.array(64, int64)
    shi_stack = cuda.local.array(64, int64)
    counts[i] = _solve_shift_core(data, samples[i], models[i], d, coeffs, C0,
                                  C1, coef, chain, roots, lo_stack, hi_stack,
                                  A, B, degs, slo_stack, shi_stack)


@cuda.jit(cache=True)
def _solve_p3p_kernel(data, params, samples, models, counts):
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


@cuda.jit(cache=True)
def _solve_shared_focal_kernel(data, params, samples, models, counts):
    i = cuda.grid(1)
    if i >= samples.shape[0]:
        return
    a = cuda.local.array(17, float64)
    b = cuda.local.array(12, float64)
    c = cuda.local.array(18, float64)
    dd = cuda.local.array(21, float64)
    C0 = cuda.local.array((3, 3), float64)
    C1 = cuda.local.array((3, 4), float64)
    AM = cuda.local.array((4, 4), float64)
    coef = cuda.local.array(5, float64)
    chain = cuda.local.array((5, 5), float64)
    roots = cuda.local.array(4, float64)
    lo_stack = cuda.local.array(64, float64)
    hi_stack = cuda.local.array(64, float64)
    row = cuda.local.array(4, float64)
    tmp_row = cuda.local.array(4, float64)
    A = cuda.local.array((3, 3), float64)
    B = cuda.local.array((3, 3), float64)
    degs = cuda.local.array(5, int64)
    slo_stack = cuda.local.array(64, int64)
    shi_stack = cuda.local.array(64, int64)
    counts[i] = _solve_shared_focal_core(
        data, samples[i], models[i], a, b, c, dd, C0, C1, AM, coef, chain,
        roots, lo_stack, hi_stack, row, tmp_row, A, B, degs, slo_stack,
        shi_stack)


@cuda.jit(cache=True)
def _solve_varying_focal_kernel(data, params, samples, models, counts):
    i = cuda.grid(1)
    if i >= samples.shape[0]:
        return
    a = cuda.local.array(18, float64)
    b = cuda.local.array(18, float64)
    A3 = cuda.local.array((3, 3), float64)
    B3 = cuda.local.array((3, 3), float64)
    A = cuda.local.array((3, 3), float64)
    B = cuda.local.array((3, 3), float64)
    counts[i] = _solve_varying_focal_core(data, samples[i], models[i], a, b,
                                          A3, B3, A, B)


def _make_solve_batch(kernel):
    def solve_batch(data, params, samples, models, counts, stream=0):
        batch = samples.shape[0]
        blocks = (batch + SOLVE_THREADS - 1) // SOLVE_THREADS
        kernel[blocks, SOLVE_THREADS, stream](data, params, samples, models,
                                              counts)
    return solve_batch


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------

@cuda_jit()
def _prepare_calibrated(model, params, dm64):
    # E = [t]_x R; the depth parameters do not enter the scoring
    _essential_from_pose(model, dm64)
    dm64[W_SCALE_REPROJ] = params[0]
    dm64[W_SAMPSON] = params[1]
    return True


@cuda_jit()
def _prepare_focal(model, params, dm64):
    # F = diag(1,1,f2) E diag(1,1,f1), the principal points already
    # subtracted by the estimator
    e = cuda.local.array(9, float64)
    f1 = model[12]
    f2 = model[13]
    if f1 <= 0.0 or f2 <= 0.0:
        return False
    _essential_from_pose(model, e)
    _focal_fundamental(e, f1, f2, dm64)
    dm64[W_SCALE_REPROJ] = params[0]
    dm64[W_SAMPSON] = params[1]
    return True


@cuda_jit(fastmath=True, inline=True)
def _score_point(dm32, m32, data, i, max_error_sq):
    # the matrix overload, with no cheirality check - see the module docstring
    r2, den = _sampson_residual(dm32, data[0][i], data[1][i], data[2][i],
                                data[3][i])
    if den > float32(0.0) and r2 < max_error_sq * den:
        return r2 / den, True
    return float32(0.0), False


# ---------------------------------------------------------------------------
# local optimization
# ---------------------------------------------------------------------------

def _build_refine_factory(num_tangent, focal):
    """One `build_refine_kernels(loss)` for a given (num_tangent, focal)."""
    reproj = MONO_REPROJ32[(num_tangent, focal)]
    forward_point = reproj['forward_point']
    backward_point = reproj['backward_point']
    resid = MONO_RESID32[focal]
    forward_residual = resid['forward_residual']
    backward_residual = resid['backward_residual']
    NT = num_tangent
    shared = focal and num_tangent == 8

    def build_refine_kernels(loss):
        weight_fn = loss.weight
        cost_fn = loss.cost

        @cuda_jit()
        def init_state(model_in, state):
            if focal and (model_in[12] <= 0.0 or model_in[13] <= 0.0):
                return False
            for j in range(MODEL_SIZE):
                state[j] = model_in[j]
            return True

        @cuda_jit(inline=True)
        def state_to_model(state, model):
            for j in range(MODEL_SIZE):
                model[j] = state[j]

        @cuda_jit()
        def model_derived(model, params, dm64):
            if focal:
                e = cuda.local.array(9, float64)
                f1 = model[12]
                f2 = model[13]
                if f1 <= 0.0 or f2 <= 0.0:
                    return False
                _essential_from_pose(model, e)
                _focal_fundamental(e, f1, f2, dm64)
            else:
                _essential_from_pose(model, dm64)
            dm64[W_SCALE_REPROJ] = params[0]
            dm64[W_SAMPSON] = params[1]
            return True

        @cuda_jit()
        def jacobian_basis(model, state, params, dm64, B64, aux):
            # dE/dtheta (or dF/dtheta) for the six pose directions; the scale
            # and shift columns are zero because the Sampson term does not
            # see them, and the focal columns are read off E
            e = cuda.local.array(9, float64)
            _essential_from_pose(model, e)
            _essential_tangent_rows(e, model, B64, NT)
            if focal:
                _focal_tangent_rows(e, model[12], model[13], B64, shared)

        @cuda_jit(fastmath=True, inline=True)
        def mask_point(dm32, st32, data, i, relaxed_sq):
            # relaxed_scale is 0.0 for every monodepth problem: the CPU
            # refiners minimize over the whole correspondence set
            return True

        @cuda_jit(fastmath=True)
        def cost_point(dm32, st32, data, i, max_error_sq):
            # the hybrid cost of one correspondence: the Sampson term plus
            # both reprojection terms, each robustified separately so the LM
            # accept/reject test matches what accum_point weights. Uses the
            # residual-only kernels - the cost evaluation has no use for the
            # jacobian rows.
            x = data[0][i]
            y = data[1][i]
            xp = data[2][i]
            yp = data[3][i]
            scale_reproj = dm32[W_SCALE_REPROJ]
            weight_sampson = dm32[W_SAMPSON]
            total = float32(0.0)

            if weight_sampson > float32(0.0):
                r2, den = _sampson_residual(dm32, x, y, xp, yp)
                if den > float32(0.0):
                    total += weight_sampson * cost_fn(r2 / den, max_error_sq)
                else:
                    total += weight_sampson * cost_fn(float32(1e18),
                                                      max_error_sq)

            if scale_reproj > float32(0.0):
                rx, ry, ok = forward_residual(st32, x, y, xp, yp, data[4][i])
                if ok:
                    total += cost_fn(scale_reproj * (rx * rx + ry * ry),
                                     max_error_sq)
                rx, ry, ok = backward_residual(st32, x, y, xp, yp, data[5][i])
                if ok:
                    total += cost_fn(scale_reproj * (rx * rx + ry * ry),
                                     max_error_sq)
            return total

        @cuda_jit(fastmath=True)
        def accum_point(dm32, st32, B32, data, i, max_error_sq, red, tid,
                        scratch):
            # up to three residual families per correspondence; `scratch` is
            # (4, 9): dsdF, J, J0, J1
            x = data[0][i]
            y = data[1][i]
            xp = data[2][i]
            yp = data[3][i]
            scale_reproj = dm32[W_SCALE_REPROJ]
            weight_sampson = dm32[W_SAMPSON]
            nres = 0

            if scale_reproj > float32(0.0):
                for pass_ in range(2):
                    if pass_ == 0:
                        rx, ry, ok = forward_point(st32, x, y, xp, yp,
                                                   data[4][i], scratch[2],
                                                   scratch[3])
                    else:
                        rx, ry, ok = backward_point(st32, x, y, xp, yp,
                                                    data[5][i], scratch[2],
                                                    scratch[3])
                    if ok:
                        w = weight_fn(scale_reproj * (rx * rx + ry * ry),
                                      max_error_sq)
                        if w > float32(0.0):
                            nres += 2
                            wsr = w * scale_reproj
                            idx = 0
                            for p in range(NT):
                                for q in range(p, NT):
                                    red[tid, idx] += float64(
                                        wsr * (scratch[2, p] * scratch[2, q]
                                               + scratch[3, p] * scratch[3, q]))
                                    idx += 1
                            base = idx
                            for p in range(NT):
                                red[tid, base + p] += float64(
                                    wsr * (scratch[2, p] * rx
                                           + scratch[3, p] * ry))

            if weight_sampson > float32(0.0):
                s_i, ok = _sampson_point_jacobian(dm32, x, y, xp, yp,
                                                  scratch[0])
                if ok:
                    w = weight_fn(s_i * s_i, max_error_sq)
                    if w > float32(0.0):
                        nres += 1
                        for p in range(NT):
                            acc = float32(0.0)
                            for jj in range(9):
                                acc += scratch[0, jj] * B32[p, jj]
                            scratch[1, p] = acc
                        wsamp = w * weight_sampson
                        idx = 0
                        for p in range(NT):
                            for q in range(p, NT):
                                red[tid, idx] += float64(
                                    wsamp * scratch[1, p] * scratch[1, q])
                                idx += 1
                        base = idx
                        for p in range(NT):
                            red[tid, base + p] += float64(
                                wsamp * scratch[1, p] * s_i)
            return nres

        @cuda_jit()
        def retract(state, delta, state_new, aux):
            # [w(3), dt(3), dscale] then the shifts or the focals; the
            # translation is *not* on the unit sphere here - the depths fix
            # the scale - so it takes three additive parameters
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
            if focal:
                if shared:
                    f = state[12] + delta[7]
                    state_new[12] = f
                    state_new[13] = f
                else:
                    state_new[12] = state[12] + delta[7]
                    state_new[13] = state[13] + delta[8]
                state_new[14] = state[14] + delta[6]
            else:
                state_new[12] = state[12] + delta[6]
                if NT > 7:
                    state_new[13] = state[13] + delta[7]
                    state_new[14] = state[14] + delta[8]
                else:
                    state_new[13] = state[13]
                    state_new[14] = state[14]

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

    return build_refine_kernels


def _make_problem(name, kernel, max_models, num_tangent, focal):
    return CudaProblem(
        name=name,
        sample_size=SAMPLE_SIZE,
        num_params=NUM_PARAMS,
        max_models=max_models,
        data_width=6,          # x1_x, x1_y, x2_x, x2_y, d1, d2
        num_tangent=num_tangent,
        state_size=STATE_SIZE,
        derived_size=DERIVED_SIZE,
        basis_width=9,
        aux_shape=(1, 1),      # the retraction needs nothing from the basis
        scratch_shape=(4, 9),  # dsdF, J, J0, J1
        min_inliers=-1,        # the CPU refiners have no inlier-count gate
        relaxed_scale=0.0,     # "keep every correspondence"
        solve_batch=_make_solve_batch(kernel),
        score_kernels=((_prepare_focal if focal else _prepare_calibrated),
                       _score_point),
        refine_factory=_build_refine_factory(num_tangent, focal),
        default_params=(0.0, 1.0),   # scale_reproj, weight_sampson
    )


PROBLEM_CALIBRATED = _make_problem('monodepth', _solve_p3p_kernel, 4, 7, False)
PROBLEM_SHIFT = _make_problem('monodepth-shift', _solve_shift_kernel, 4, 9,
                              False)
PROBLEM_SHARED_FOCAL = _make_problem('monodepth-shared-focal',
                                     _solve_shared_focal_kernel, 4, 8, True)
PROBLEM_VARYING_FOCAL = _make_problem('monodepth-varying-focal',
                                      _solve_varying_focal_kernel, 1, 9, True)
