"""Batch-parallel local optimization (LM) on the GPU.

One block per candidate pose, and the *entire* LM loop lives inside one kernel
launch. That is the point of this module: the alternative - launching an
accumulate kernel per LM step - would pay ~5-10us of launch latency 25 times
per candidate and dominate everything else. Here the block synchronizes
internally with `cuda.syncthreads()` and the host sees a single launch that
refines every candidate of the round at once.

Mapping of the CPU refiner onto a block:

- The O(n) work - the relaxed inlier mask, the truncated Sampson cost, and the
  normal equations - is spread across the block's threads, which stride over
  the correspondences and tree-reduce in shared memory. The accumulate reduces
  20 doubles at once (the upper triangle of the 5x5 JtJ plus Jtr) so one
  reduction pass serves the whole step.
- The O(1) work - the damped Cholesky solve of the 5x5 system, the retraction,
  and the damping schedule - runs in thread 0 and is published through shared
  memory. Every branch the loop takes is therefore block-uniform, which is
  what makes the `syncthreads()` calls inside it legal.

Semantics follow `build_pose_lo_refine` + `build_lm_refine` exactly: the
relaxed inlier subset is selected once from the *initial* model at
`LO_INLIER_SCALE x max_error_sq` with the cheirality check, the LM then
minimizes the truncated Sampson cost (no cheirality, matching poselib's
bundles) over that fixed subset, and the same damping schedule and
accept/reject test apply. The subset is carried as a per-candidate mask rather
than compacted into new arrays, because device code cannot allocate.
"""

import math

import numpy as np
from numba import cuda, float32, float64, int64

from fastpose.cuda.backend import cuda_jit
from fastpose.refiners.essential import MIN_INLIERS
from fastpose.refiners.utils import LO_INLIER_SCALE, build_refiner_primitives
from fastpose.scorers.sampson import MIN_DEPTH, build_sampson_point_kernels

# float64: the retraction, the tangent basis and the pose state. All O(1) per
# LM step, and the state is what the caller ultimately gets back.
_PRIM64 = build_refiner_primitives(cuda_jit, real=float64)
_mat3_mul = _PRIM64['mat3_mul']
_rodrigues = _PRIM64['rodrigues']
_essential_tangent_rows_core = _PRIM64['essential_tangent_rows_core']

# float32: the per-point residual and jacobian, which is the O(n) work
_PRIM32 = build_refiner_primitives(cuda_jit, real=float32)
_sampson_point_jacobian = _PRIM32['sampson_point_jacobian']

_F32 = build_sampson_point_kernels(cuda_jit, real=float32)
_sampson_residual = _F32['sampson_residual']
_cheirality_ok = _F32['cheirality_ok']

_F64 = build_sampson_point_kernels(cuda_jit, real=float64)
_essential_from_pose = _F64['essential_from_pose']

# Threads per block for the LM kernel. The accumulate reduction holds
# LM_THREADS x 20 doubles in shared memory, so this trades reduction width
# against shared-memory footprint: 128 costs ~20 KB and leaves room under the
# 48 KB default limit for the state, jacobian basis and normal equations.
LM_THREADS = 128

NUM_TANGENT = 5
# upper triangle of the 5x5 JtJ (15) followed by Jtr (5)
NUM_ACC = 20

# The refined model buffer carries the 12 pose parameters plus a success flag
# in a 13th column, so the driver reads the result and its validity in one
# synchronizing copy instead of two. The batched scorer only ever indexes
# columns 0..11, so it consumes this layout unchanged.
COL_SUCCESS = 12
REFINED_WIDTH = 13


@cuda.jit(device=True, inline=True)
def _reduce_sum(red, tid, nthreads, width):
    # in-place tree reduction of red[:, :width] into row 0
    stride = nthreads // 2
    while stride > 0:
        if tid < stride:
            for q in range(width):
                red[tid, q] += red[tid + stride, q]
        cuda.syncthreads()
        stride //= 2


@cuda.jit(device=True, inline=True)
def _reduce_scalar(ss, tid, nthreads):
    stride = nthreads // 2
    while stride > 0:
        if tid < stride:
            ss[tid] += ss[tid + stride]
        cuda.syncthreads()
        stride //= 2


@cuda.jit(device=True, inline=True)
def _reduce_count(sc, tid, nthreads):
    stride = nthreads // 2
    while stride > 0:
        if tid < stride:
            sc[tid] += sc[tid + stride]
        cuda.syncthreads()
        stride //= 2


@cuda.jit(device=True)
def _solve_damped5(A, Jtr, delta):
    # Cholesky solve of A delta = -Jtr for the 5x5 damped normal equations,
    # in place on A. Mirrors _solve_damped in refiners/lm.py, including its
    # refusal to proceed on a non-positive or non-finite pivot - deliberately
    # not fastmath so the pivot check is honest about NaN.
    n = NUM_TANGENT
    for j in range(n):
        d = A[j, j]
        for k in range(j):
            d -= A[j, k] * A[j, k]
        if not (d > 0.0) or not math.isfinite(d):
            return False
        d = math.sqrt(d)
        A[j, j] = d
        inv = 1.0 / d
        for i in range(j + 1, n):
            s = A[i, j]
            for k in range(j):
                s -= A[i, k] * A[j, k]
            A[i, j] = s * inv
    for i in range(n):
        s = -Jtr[i]
        for k in range(i):
            s -= A[i, k] * delta[k]
        delta[i] = s / A[i, i]
    for i in range(n - 1, -1, -1):
        s = delta[i]
        for k in range(i + 1, n):
            s -= A[k, i] * delta[k]
        delta[i] = s / A[i, i]
    # a finite factorization can still produce a non-finite step when Jtr
    # itself carries inf
    for i in range(n):
        if not math.isfinite(delta[i]):
            return False
    return True


def build_lm_refine_kernel(loss):
    # one kernel per loss, closing over its weight/cost as compile-time
    # constants - the same pattern build_lm_refine and build_sampson_accumulate
    # use on the CPU. TruncatedLoss gives the RANSAC-internal local
    # optimization; CauchyLoss gives the final polish pass.
    weight_fn = loss.weight
    cost_fn = loss.cost

    @cuda.jit(cache=True)
    def _lm_refine_kernel(x1_x, x1_y, x2_x, x2_y, init_models, refined,
                          keep, max_error_sq, relaxed_sq, num_iterations):
        b = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        nt = cuda.blockDim.x
        n = x1_x.shape[0]

        f = cuda.shared.array(12, float64)
        f_new = cuda.shared.array(12, float64)
        f32 = cuda.shared.array(12, float32)
        e64 = cuda.shared.array(9, float64)
        e = cuda.shared.array(9, float32)
        B64 = cuda.shared.array((NUM_TANGENT, 9), float64)
        B = cuda.shared.array((NUM_TANGENT, 9), float32)
        JtJ = cuda.shared.array((NUM_TANGENT, NUM_TANGENT), float64)
        Jtr = cuda.shared.array(NUM_TANGENT, float64)
        Amat = cuda.shared.array((NUM_TANGENT, NUM_TANGENT), float64)
        delta = cuda.shared.array(NUM_TANGENT, float64)
        b1 = cuda.shared.array(3, float64)
        b2 = cuda.shared.array(3, float64)
        Rod = cuda.shared.array((3, 3), float64)
        Rcur = cuda.shared.array((3, 3), float64)
        Rnew = cuda.shared.array((3, 3), float64)
        red = cuda.shared.array((LM_THREADS, NUM_ACC), float64)
        ss = cuda.shared.array(LM_THREADS, float64)
        sc = cuda.shared.array(LM_THREADS, int64)
        sval = cuda.shared.array(4, float64)   # lam, best_cost, cost_new, step_sq
        ctl = cuda.shared.array(4, int64)      # recompute, stop, ok, nres

        mes32 = float32(max_error_sq)
        relaxed32 = float32(relaxed_sq)

        # ---- initial state: copy the model and normalize t -------------------
        if tid < 12:
            f[tid] = init_models[b, tid]
        cuda.syncthreads()
        if tid == 0:
            nrm = math.sqrt(f[9] * f[9] + f[10] * f[10] + f[11] * f[11])
            if nrm < 1e-12:
                ctl[1] = 1
            else:
                ctl[1] = 0
                inv = 1.0 / nrm
                f[9] *= inv
                f[10] *= inv
                f[11] *= inv
            _essential_from_pose(f, e64)
        cuda.syncthreads()
        if ctl[1] == 1:
            if tid == 0:
                refined[b, 0, COL_SUCCESS] = 0.0
            return
        # float32 mirrors of the pose and E for the per-point loops; refreshed
        # from the float64 state whenever it changes
        if tid < 12:
            f32[tid] = float32(f[tid])
        if tid < 9:
            e[tid] = float32(e64[tid])
        cuda.syncthreads()

        # ---- relaxed inlier subset, selected once from the initial model ------
        c = 0
        i = tid
        while i < n:
            x = x1_x[i]
            y = x1_y[i]
            xp = x2_x[i]
            yp = x2_y[i]
            r2, den = _sampson_residual(e, x, y, xp, yp)
            ok = (den > float32(0.0) and r2 < relaxed32 * den
                  and _cheirality_ok(f32, x, y, xp, yp, float32(MIN_DEPTH)))
            if ok:
                keep[b, i] = 1
                c += 1
            else:
                keep[b, i] = 0
            i += nt
        sc[tid] = c
        cuda.syncthreads()
        _reduce_count(sc, tid, nt)
        if sc[0] <= MIN_INLIERS:
            if tid == 0:
                refined[b, 0, COL_SUCCESS] = 0.0
            return

        # ---- initial cost ----------------------------------------------------
        s = 0.0
        i = tid
        while i < n:
            if keep[b, i] == 1:
                r2, den = _sampson_residual(e, x1_x[i], x1_y[i], x2_x[i], x2_y[i])
                # 1e18 for a degenerate denominator, matching
                # build_sampson_cost; for TruncatedLoss both branches reduce
                # to the division-free form the CPU scorer uses
                # residual in float32, cost accumulated in float64
                s += float64(cost_fn(r2 / den if den > float32(0.0)
                                     else float32(1e18), mes32))
            i += nt
        ss[tid] = s
        cuda.syncthreads()
        _reduce_scalar(ss, tid, nt)
        if tid == 0:
            sval[0] = 1e-4     # lam
            sval[1] = ss[0]    # best_cost
            ctl[0] = 1         # recompute jacobian
            ctl[1] = 0         # stop
        cuda.syncthreads()

        # ---- LM loop ---------------------------------------------------------
        for _ in range(num_iterations):
            if ctl[1] == 1:
                break

            if ctl[0] == 1:
                # tangent basis of E at the current pose
                if tid == 0:
                    _essential_from_pose(f, e64)
                    _essential_tangent_rows_core(f, e64, B64, b1, b2)
                cuda.syncthreads()
                if tid < 9:
                    e[tid] = float32(e64[tid])
                if tid < NUM_TANGENT * 9:
                    B[tid // 9, tid % 9] = float32(B64[tid // 9, tid % 9])
                cuda.syncthreads()

                for q in range(NUM_ACC):
                    red[tid, q] = 0.0
                # per-point jacobian scratch in float32: this is the O(n) inner
                # loop, and the products below land in the float64 `red`
                dsdF = cuda.local.array(9, float32)
                J = cuda.local.array(NUM_TANGENT, float32)
                nres = 0
                i = tid
                while i < n:
                    if keep[b, i] == 1:
                        x = x1_x[i]
                        y = x1_y[i]
                        xp = x2_x[i]
                        yp = x2_y[i]
                        s_i, ok = _sampson_point_jacobian(e, x, y, xp, yp, dsdF)
                        w = weight_fn(s_i * s_i, mes32) if ok else float32(0.0)
                        # a zero weight drops the point entirely; for
                        # TruncatedLoss that is exactly the hard cutoff
                        if w > float32(0.0):
                            nres += 1
                            for p in range(NUM_TANGENT):
                                acc = float32(0.0)
                                for jj in range(9):
                                    acc += dsdF[jj] * B[p, jj]
                                J[p] = acc
                            idx = 0
                            for p in range(NUM_TANGENT):
                                for q in range(p, NUM_TANGENT):
                                    red[tid, idx] += float64(w * J[p] * J[q])
                                    idx += 1
                            for p in range(NUM_TANGENT):
                                red[tid, 15 + p] += float64(w * J[p] * s_i)
                    i += nt
                sc[tid] = nres
                cuda.syncthreads()
                _reduce_sum(red, tid, nt, NUM_ACC)
                _reduce_count(sc, tid, nt)

                if tid == 0:
                    ctl[3] = sc[0]
                    idx = 0
                    for p in range(NUM_TANGENT):
                        for q in range(p, NUM_TANGENT):
                            JtJ[p, q] = red[0, idx]
                            JtJ[q, p] = red[0, idx]
                            idx += 1
                    for p in range(NUM_TANGENT):
                        Jtr[p] = red[0, 15 + p]
                cuda.syncthreads()
                if ctl[3] < NUM_TANGENT:
                    # too few residuals to constrain the step, as in lm.py
                    if tid == 0:
                        refined[b, 0, COL_SUCCESS] = 0.0
                    return

            # ---- damped normal equations + retraction, in thread 0 -----------
            if tid == 0:
                lam = sval[0]
                for p in range(NUM_TANGENT):
                    for q in range(NUM_TANGENT):
                        Amat[p, q] = JtJ[p, q]
                    Amat[p, p] += lam
                if not _solve_damped5(Amat, Jtr, delta):
                    ctl[2] = 0
                else:
                    ctl[2] = 1
                    step_sq = 0.0
                    for p in range(NUM_TANGENT):
                        step_sq += delta[p] * delta[p]
                    sval[3] = step_sq
                    # R_new = R exp([w]_x)
                    for ii in range(3):
                        for jj in range(3):
                            Rcur[ii, jj] = f[3 * ii + jj]
                    _rodrigues(delta[0], delta[1], delta[2], Rod)
                    _mat3_mul(Rcur, Rod, Rnew)
                    for ii in range(3):
                        for jj in range(3):
                            f_new[3 * ii + jj] = Rnew[ii, jj]
                    # t_new = normalize(t + d3 b1 + d4 b2); b1/b2 were written by
                    # essential_tangent_rows_core at this same state
                    t0 = f[9] + delta[3] * b1[0] + delta[4] * b2[0]
                    t1 = f[10] + delta[3] * b1[1] + delta[4] * b2[1]
                    t2 = f[11] + delta[3] * b1[2] + delta[4] * b2[2]
                    inv = 1.0 / math.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
                    f_new[9] = t0 * inv
                    f_new[10] = t1 * inv
                    f_new[11] = t2 * inv
                    _essential_from_pose(f_new, e64)
            cuda.syncthreads()
            # round the trial E for the float32 cost loop below. Done by the
            # whole block after the sync, so every thread sees it.
            if tid < 9:
                e[tid] = float32(e64[tid])
            cuda.syncthreads()

            if ctl[2] == 0:
                # singular or non-finite system: run the damping schedule dry
                # rather than aborting, exactly as lm.py does
                if tid == 0:
                    sval[0] *= 10.0
                    ctl[0] = 0
                    if sval[0] > 1e10:
                        ctl[1] = 1
                cuda.syncthreads()
                continue

            # ---- cost at the trial state ------------------------------------
            s = 0.0
            i = tid
            while i < n:
                if keep[b, i] == 1:
                    r2, den = _sampson_residual(e, x1_x[i], x1_y[i], x2_x[i],
                                                x2_y[i])
                    # residual in float32, cost accumulated in float64
                    s += float64(cost_fn(r2 / den if den > float32(0.0)
                                         else float32(1e18), mes32))
                i += nt
            ss[tid] = s
            cuda.syncthreads()
            _reduce_scalar(ss, tid, nt)

            if tid == 0:
                cost_new = ss[0]
                if cost_new < sval[1]:
                    sval[1] = cost_new
                    for j in range(12):
                        f[j] = f_new[j]
                    lam = sval[0] * 0.1
                    if lam < 1e-10:
                        lam = 1e-10
                    sval[0] = lam
                    ctl[0] = 1
                    if sval[3] < 1e-20:
                        ctl[1] = 1
                else:
                    sval[0] *= 10.0
                    ctl[0] = 0
                    if sval[0] > 1e10:
                        ctl[1] = 1
            cuda.syncthreads()

        if tid < 12:
            refined[b, 0, tid] = f[tid]
        if tid == 0:
            refined[b, 0, COL_SUCCESS] = 1.0

    return _lm_refine_kernel



_KERNELS = {}


def get_lm_kernel(loss):
    # memoized per loss type, so rebuilding an estimator does not recompile
    key = type(loss)
    if key not in _KERNELS:
        _KERNELS[key] = build_lm_refine_kernel(loss)
    return _KERNELS[key]


@cuda.jit(cache=True)
def _gather_candidates(models, hyp_idx, model_idx, out):
    # out[c] = models[hyp_idx[c], model_idx[c]] for the round's LO candidates
    c = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if tid < 12:
        out[c, tid] = models[hyp_idx[c], model_idx[c], tid]


class RefineBuffers():
    # per-round LM buffers, allocated once by the driver and reused
    def __init__(self, max_candidates, num_points):
        self.init_models = cuda.device_array((max_candidates, 12),
                                             dtype=np.float64)
        # shaped (k, 1, 12) so the refined models feed straight back into the
        # batched scorer, which expects a models[hypothesis, model, param]
        # table and a per-hypothesis count
        self.refined = cuda.device_array((max_candidates, 1, REFINED_WIDTH),
                                         dtype=np.float64)
        self.counts = cuda.to_device(np.ones(max_candidates, dtype=np.int64))
        self.keep = cuda.device_array((max_candidates, num_points),
                                      dtype=np.uint8)
        self.hyp_idx = cuda.device_array(max_candidates, dtype=np.int64)
        self.model_idx = cuda.device_array(max_candidates, dtype=np.int64)


def gather_candidates(models, buffers, hyp_idx, model_idx, num_candidates,
                      stream=0):
    # copies models[hyp_idx[c], model_idx[c]] into buffers.init_models[c]
    buffers.hyp_idx[:num_candidates].copy_to_device(hyp_idx[:num_candidates])
    buffers.model_idx[:num_candidates].copy_to_device(model_idx[:num_candidates])
    _gather_candidates[num_candidates, 32, stream](
        models, buffers.hyp_idx, buffers.model_idx, buffers.init_models)


def refine_batch(data, models, buffers, hyp_idx, model_idx, num_candidates,
                 max_error_sq, num_iterations, loss,
                 relaxed_scale=LO_INLIER_SCALE, stream=0):
    # gathers the round's candidates and locally optimizes all of them in one
    # launch; results land in buffers.refined, whose last column is the
    # per-candidate success flag
    gather_candidates(models, buffers, hyp_idx, model_idx, num_candidates,
                      stream)
    refine_prepared(data, buffers, num_candidates, max_error_sq,
                    num_iterations, loss, relaxed_scale, stream)


def refine_prepared(data, buffers, num_candidates, max_error_sq,
                    num_iterations, loss, relaxed_scale=LO_INLIER_SCALE,
                    stream=0):
    # same, for models already sitting in buffers.init_models (the final
    # polish pass, which refines one model the driver has already chosen)
    kernel = get_lm_kernel(loss)
    kernel[num_candidates, LM_THREADS, stream](
        data[0], data[1], data[2], data[3], buffers.init_models,
        buffers.refined, buffers.keep, max_error_sq,
        relaxed_scale * max_error_sq, num_iterations)
