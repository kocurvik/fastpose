"""Problem-agnostic batch-parallel local optimization (LM) on the GPU.

One block per candidate model, and the *entire* LM loop lives inside one
kernel launch. That is the point of this module: the alternative - launching
an accumulate kernel per LM step - would pay ~5-10us of launch latency 25
times per candidate and dominate everything else. Here the block synchronizes
internally with `cuda.syncthreads()` and the host sees a single launch that
refines every candidate of the round at once.

Mapping of the CPU refiner onto a block:

- The O(n) work - the relaxed inlier mask, the robust cost, and the normal
  equations - is spread across the block's threads, which stride over the
  correspondences and tree-reduce in shared memory. The accumulate reduces
  `NUM_ACC = T(T+1)/2 + T` doubles at once (the upper triangle of JtJ plus
  Jtr) so one reduction pass serves the whole step.
- The O(1) work - the damped Cholesky solve, the retraction and the damping
  schedule - runs in thread 0 and is published through shared memory. Every
  branch the loop takes is therefore block-uniform, which is what makes the
  `syncthreads()` calls inside it legal.

Semantics follow `build_lm_refine` plus the `build_*_lo_refine` wrappers
exactly: the relaxed inlier subset is selected once from the *initial* model
at `relaxed_scale x max_error_sq` with whatever check the CPU refiner applies,
the LM then minimizes the robust cost over that fixed subset, and the same
damping schedule and accept/reject test apply. The subset is carried as a
per-candidate `uint8` mask rather than compacted into new arrays, because
device code cannot allocate.

The shared workspace
--------------------
Four pieces, so that no device function ever has to do offset arithmetic into
one flat blob:

    dm64 / dm32   the *derived* form the per-point loop reads: E, the induced
                  F, a focal-scaled pose. `derived_size` entries.
    model / st32  the flat model itself and its float32 mirror, for the
                  per-point work that needs the pose rather than the matrix
                  (the cheirality check, the monodepth reprojection term).
    B64 / B32     the tangent basis, `(num_tangent, basis_width)`; the
                  Sampson problems put dF/dtheta here, the reprojection ones
                  have an analytic jacobian and set `basis_width = 1`.
    aux           float64 `(aux_rows, aux_cols)` scratch that must survive
                  between the thread-0 calls - the two translation basis
                  vectors the retraction shares with the jacobian, say.

What a problem supplies
-----------------------
A factory `build_refine_kernels(loss)` returning these device functions. The
float64 ones run in thread 0; the three float32 ones are the per-point hot
loop.

    init_state(model_in, state) -> bool
    state_to_model(state, model)
    model_derived(model, params, dm64) -> bool
    jacobian_basis(model, state, params, dm64, B64, aux)
    mask_point(dm32, st32, data, i, relaxed_sq) -> bool
    cost_point(dm32, st32, data, i, max_error_sq) -> float32
    accum_point(dm32, st32, B32, data, i, max_error_sq, red, tid, scratch) -> int
    retract(state, delta, state_new, aux)

Shared-memory budget
--------------------
`red` is `threads x NUM_ACC x 8` bytes and dominates; everything else is a few
KB. `lm_threads_for` picks the largest power-of-two thread count that keeps
the total under the 48 KB default limit, which is 128 up to T = 7 and 64 for
the T = 8 and T = 9 monodepth problems.
"""

import numpy as np
from numba import cuda, float32, float64, int64

from fastpose.cuda.reductions import (get_solve_damped, reduce_count,
                                      reduce_scalar, reduce_sum)

# Threads per block for the LM kernel, capped so the reduction stays shallow.
# 128 measured a reasonable default for the 5-point port; larger buys nothing
# because the tree depth grows while the per-thread work shrinks.
MAX_LM_THREADS = 128
MIN_LM_THREADS = 32

# CUDA's default shared-memory limit per block, minus a margin for everything
# that is not the reduction array. The margin is generous: the largest problem
# here needs about 1.5 KB of it.
SHARED_LIMIT = 48 * 1024
SHARED_MARGIN = 6 * 1024


def num_accumulators(num_tangent):
    # upper triangle of the T x T JtJ followed by Jtr
    return num_tangent * (num_tangent + 1) // 2 + num_tangent


def lm_threads_for(num_tangent):
    # largest power-of-two thread count whose reduction array fits the shared
    # budget. Computed rather than hard-coded because NUM_ACC grows
    # quadratically in T: 20 doubles at T=5 but 54 at T=9, where 128 threads
    # would need 55 KB and the launch would simply fail.
    acc = num_accumulators(num_tangent)
    threads = MAX_LM_THREADS
    while threads > MIN_LM_THREADS:
        if threads * acc * 8 <= SHARED_LIMIT - SHARED_MARGIN:
            break
        threads //= 2
    return threads


def build_lm_refine_kernel(fns, num_tangent, state_size, num_params,
                           derived_size, basis_width, aux_shape,
                           scratch_shape, min_inliers, threads):
    """Compile the whole LM loop for one problem and one loss.

    `fns` is the dict described in the module docstring, already specialized
    for the loss; one kernel per (problem, loss), which is what lets the
    RANSAC-internal truncated pass and the Cauchy final polish share a source.
    """
    init_state = fns['init_state']
    state_to_model = fns['state_to_model']
    model_derived = fns['model_derived']
    jacobian_basis = fns['jacobian_basis']
    mask_point = fns['mask_point']
    cost_point = fns['cost_point']
    accum_point = fns['accum_point']
    retract = fns['retract']

    NT = num_tangent
    NACC = num_accumulators(NT)
    SS_ = state_size
    NP = num_params
    DS = derived_size
    BW = basis_width
    AR, AC = aux_shape
    SR, SC = scratch_shape
    COL_OK = NP
    solve_damped = get_solve_damped(NT)

    @cuda.jit(cache=True)
    def _lm_refine_kernel(data, params, init_models, refined, keep,
                          max_error_sq, relaxed_sq, num_iterations):
        b = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        nt = cuda.blockDim.x
        n = data[0].shape[0]

        state = cuda.shared.array(SS_, float64)
        state_new = cuda.shared.array(SS_, float64)
        model = cuda.shared.array(NP, float64)
        model_new = cuda.shared.array(NP, float64)
        dm64 = cuda.shared.array(DS, float64)
        dm32 = cuda.shared.array(DS, float32)
        st32 = cuda.shared.array(NP, float32)
        B64 = cuda.shared.array((NT, BW), float64)
        B32 = cuda.shared.array((NT, BW), float32)
        aux = cuda.shared.array((AR, AC), float64)
        JtJ = cuda.shared.array((NT, NT), float64)
        Jtr = cuda.shared.array(NT, float64)
        Amat = cuda.shared.array((NT, NT), float64)
        delta = cuda.shared.array(NT, float64)
        red = cuda.shared.array((threads, NACC), float64)
        ss = cuda.shared.array(threads, float64)
        sc = cuda.shared.array(threads, int64)
        sval = cuda.shared.array(4, float64)   # lam, best_cost, _, step_sq
        ctl = cuda.shared.array(4, int64)      # recompute, stop, ok, nres

        scratch = cuda.local.array((SR, SC), float32)

        mes32 = float32(max_error_sq)
        relaxed32 = float32(relaxed_sq)

        # ---- initial state ------------------------------------------------
        if tid == 0:
            if init_state(init_models[b], state):
                state_to_model(state, model)
                ctl[2] = 1 if model_derived(model, params, dm64) else 0
            else:
                ctl[2] = 0
        cuda.syncthreads()
        if ctl[2] == 0:
            if tid == 0:
                refined[b, 0, COL_OK] = 0.0
            return
        # float32 mirrors of everything the per-point loops read; refreshed
        # from the float64 originals whenever they change
        i = tid
        while i < DS:
            dm32[i] = float32(dm64[i])
            i += nt
        i = tid
        while i < NP:
            st32[i] = float32(model[i])
            i += nt
        cuda.syncthreads()

        # ---- relaxed inlier subset, selected once from the initial model ---
        c = 0
        i = tid
        while i < n:
            if mask_point(dm32, st32, data, i, relaxed32):
                keep[b, i] = 1
                c += 1
            else:
                keep[b, i] = 0
            i += nt
        sc[tid] = c
        cuda.syncthreads()
        reduce_count(sc, tid, nt)
        if sc[0] <= min_inliers:
            if tid == 0:
                refined[b, 0, COL_OK] = 0.0
            return

        # ---- initial cost -------------------------------------------------
        s = 0.0
        i = tid
        while i < n:
            if keep[b, i] == 1:
                s += float64(cost_point(dm32, st32, data, i, mes32))
            i += nt
        ss[tid] = s
        cuda.syncthreads()
        reduce_scalar(ss, tid, nt)
        if tid == 0:
            sval[0] = 1e-4     # lam
            sval[1] = ss[0]    # best_cost
            ctl[0] = 1         # recompute jacobian
            ctl[1] = 0         # stop
        cuda.syncthreads()

        # ---- LM loop ------------------------------------------------------
        for _ in range(num_iterations):
            if ctl[1] == 1:
                break

            if ctl[0] == 1:
                # The derived form is rebuilt alongside the basis rather than
                # relied on from the trial step below: a rejected trial leaves
                # dm64 holding *its* model, and re-deriving here is O(1) and
                # removes that ordering hazard entirely.
                if tid == 0:
                    model_derived(model, params, dm64)
                    jacobian_basis(model, state, params, dm64, B64, aux)
                cuda.syncthreads()
                i = tid
                while i < DS:
                    dm32[i] = float32(dm64[i])
                    i += nt
                i = tid
                while i < NP:
                    st32[i] = float32(model[i])
                    i += nt
                i = tid
                while i < NT * BW:
                    B32[i // BW, i % BW] = float32(B64[i // BW, i % BW])
                    i += nt
                cuda.syncthreads()

                for q in range(NACC):
                    red[tid, q] = 0.0
                nres = 0
                i = tid
                while i < n:
                    if keep[b, i] == 1:
                        nres += accum_point(dm32, st32, B32, data, i, mes32,
                                            red, tid, scratch)
                    i += nt
                sc[tid] = nres
                cuda.syncthreads()
                reduce_sum(red, tid, nt, NACC)
                reduce_count(sc, tid, nt)

                if tid == 0:
                    ctl[3] = sc[0]
                    idx = 0
                    for p in range(NT):
                        for q in range(p, NT):
                            JtJ[p, q] = red[0, idx]
                            JtJ[q, p] = red[0, idx]
                            idx += 1
                    for p in range(NT):
                        Jtr[p] = red[0, NACC - NT + p]
                cuda.syncthreads()
                if ctl[3] < NT:
                    # too few residuals to constrain the step, as in lm.py
                    if tid == 0:
                        refined[b, 0, COL_OK] = 0.0
                    return

            # ---- damped normal equations + retraction, in thread 0 --------
            if tid == 0:
                lam = sval[0]
                for p in range(NT):
                    for q in range(NT):
                        Amat[p, q] = JtJ[p, q]
                    Amat[p, p] += lam
                if not solve_damped(Amat, Jtr, delta):
                    ctl[2] = 0
                else:
                    step_sq = 0.0
                    for p in range(NT):
                        step_sq += delta[p] * delta[p]
                    sval[3] = step_sq
                    retract(state, delta, state_new, aux)
                    state_to_model(state_new, model_new)
                    ctl[2] = 1 if model_derived(model_new, params, dm64) else 0
            cuda.syncthreads()
            # round the trial's derived form and model for the float32 cost
            # loop below. Done by the whole block after the sync, so every
            # thread sees it.
            i = tid
            while i < DS:
                dm32[i] = float32(dm64[i])
                i += nt
            i = tid
            while i < NP:
                st32[i] = float32(model_new[i])
                i += nt
            cuda.syncthreads()

            if ctl[2] == 0:
                # singular or non-finite system, or a state the model map
                # rejects: run the damping schedule dry rather than aborting,
                # exactly as lm.py does
                if tid == 0:
                    sval[0] *= 10.0
                    ctl[0] = 0
                    if sval[0] > 1e10:
                        ctl[1] = 1
                cuda.syncthreads()
                continue

            # ---- cost at the trial state ---------------------------------
            s = 0.0
            i = tid
            while i < n:
                if keep[b, i] == 1:
                    s += float64(cost_point(dm32, st32, data, i, mes32))
                i += nt
            ss[tid] = s
            cuda.syncthreads()
            reduce_scalar(ss, tid, nt)

            if tid == 0:
                cost_new = ss[0]
                if cost_new < sval[1]:
                    sval[1] = cost_new
                    for j in range(SS_):
                        state[j] = state_new[j]
                    for j in range(NP):
                        model[j] = model_new[j]
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

        i = tid
        while i < NP:
            refined[b, 0, i] = model[i]
            i += nt
        if tid == 0:
            refined[b, 0, COL_OK] = 1.0

    return _lm_refine_kernel


class RefineBuffers():
    """Per-round LM buffers, allocated once by the driver and reused.

    `refined` is shaped `(k, 1, num_params + 1)` so the models feed straight
    back into the batched scorer, which expects a `models[hypothesis, model,
    param]` table and a per-hypothesis count. The success flag rides in the
    extra column, so the driver reads a result and its validity in one
    synchronizing copy instead of two.
    """

    def __init__(self, max_candidates, num_points, num_params):
        self.num_params = num_params
        self.init_models = cuda.device_array((max_candidates, num_params),
                                             dtype=np.float64)
        self.refined = cuda.device_array((max_candidates, 1, num_params + 1),
                                         dtype=np.float64)
        self.counts = cuda.to_device(np.ones(max_candidates, dtype=np.int64))
        self.keep = cuda.device_array((max_candidates, num_points),
                                      dtype=np.uint8)
        self.hyp_idx = cuda.device_array(max_candidates, dtype=np.int64)
        self.model_idx = cuda.device_array(max_candidates, dtype=np.int64)


@cuda.jit(cache=True)
def _gather_candidates_kernel(models, hyp_idx, model_idx, out):
    # out[c] = models[hyp_idx[c], model_idx[c]] for the round's LO candidates
    c = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if tid < out.shape[1]:
        out[c, tid] = models[hyp_idx[c], model_idx[c], tid]


def gather_candidates(models, buffers, hyp_idx, model_idx, num_candidates,
                      stream=0):
    # copies models[hyp_idx[c], model_idx[c]] into buffers.init_models[c]
    buffers.hyp_idx[:num_candidates].copy_to_device(hyp_idx[:num_candidates])
    buffers.model_idx[:num_candidates].copy_to_device(model_idx[:num_candidates])
    _gather_candidates_kernel[num_candidates, 32, stream](
        models, buffers.hyp_idx, buffers.model_idx, buffers.init_models)
