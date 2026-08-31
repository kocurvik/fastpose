"""Problem-agnostic batch-parallel truncated (MSAC) scorer.

One block per hypothesis; the block's threads stride over the correspondences
and tree-reduce their partial (score, inlier count) in shared memory. A block
loops over the several models one minimal sample can produce, keeping its own
running best in thread 0's registers, so the host only ever reads back five
scalars per hypothesis instead of the full `batch x max_models` score table.

What a problem supplies
-----------------------
Two device functions, both built from the same factory the CPU scorer uses so
there is only ever one copy of the per-point math:

    prepare(model, params, dm64) -> bool
        float64, run in thread 0 once per model: the *derived* form the
        per-point loop reads - the epipolar matrix E = [t]_x R, the induced
        F = K2^-T E K1^-1, the focal-scaled pose. Returning False marks a
        model that cannot be scored (a non-positive focal, say), which the
        CPU scorers report as `(1e300, 0)` and this kernel reports as
        NO_MODEL.

    score_point(dm32, m32, data, i, max_error_sq) -> (r, ok)
        float32, run per correspondence, with `dm32` the rounded derived form
        and `m32` the rounded model itself (the cheirality check needs the
        pose, not just E). `r` is the *normalized* squared error and is only
        read when `ok`; an outlier is charged the truncation constant by the
        caller. Keeping the inlier test in its unnormalized form
        (`r2 < max_error_sq * den`) inside this function is what makes it
        agree point for point with the CPU scorer, which deliberately does
        not divide for outliers.

Precision is mixed exactly as in the first port: the derived form is built and
the score accumulated in float64, the rounded mirrors and every per-point
expression are float32. See Instructions.md section 4 on the `cuda-backend`
branch for the measurements behind that split.

The early bail-out is kept, which is not obvious for a block reduction: the
truncated score is a sum of non-negative terms, so any partial sum is a lower
bound on the total. The block scores a chunk, re-reduces the running total,
and every thread reads the same shared value, which makes the decision
block-uniform and the `syncthreads()` legal. Chunks grow geometrically because
a hopeless model almost always bails on the first one.

A model that bails returns a *partial* score and a *partial* inlier count.
Both are lower bounds, and a bailed-out model must not be compared on inlier
count - it isn't, because its score already exceeds the incumbent.
"""

import numpy as np
from numba import cuda, float32, float64, int64

from fastpose.cuda.backend import THREADS_PER_BLOCK

# score reported for a model slot that holds no model (m >= counts[b]) or one
# `prepare` rejected; the same 1e300 failure sentinel the CPU kernels use,
# never inf. Also the "no bail-out" bound.
NO_MODEL = 1e300

# points scored between early-bail-out checks. Each check costs one block
# reduction, so this trades bail-out latency against reduction overhead; 512
# matches the CPU scorer's SCORE_CHUNK.
SCORE_CHUNK = 512

# column layout of the packed per-hypothesis reduction output
COL_SCORE = 0
COL_IDX = 1
COL_INL = 2
COL_MAX_INL = 3
COL_MAX_IDX = 4
NUM_COLS = 5


def build_score_kernel(prepare, score_point, derived_size, num_params):
    """Compile a batched scorer for one problem.

    `derived_size` is the length of the float64 workspace `prepare` fills (and
    of its rounded float32 mirror); `num_params` the flat model width.
    """

    @cuda.jit(device=True, fastmath=True)
    def _score_one(data, dm32, m32, ss, sc, tid, nthreads, max_error_sq,
                   max_error_sq64, bound):
        # block-wide truncated score of one prepared model; returns
        # (score, num_inliers), valid in every thread after the reduction.
        #
        # `s` is float64 so the O(n) sum accumulates without drift, and the
        # truncation constant is added in float64 for the same reason - it is
        # the term most points contribute. Pass bound = NO_MODEL to disable
        # the bail-out.
        n = data[0].shape[0]
        s = 0.0
        c = 0
        start = 0
        chunk = SCORE_CHUNK
        while start < n:
            stop = start + chunk
            if stop > n:
                stop = n
            i = start + tid
            while i < stop:
                r, ok = score_point(dm32, m32, data, i, max_error_sq)
                if ok:
                    s += float64(r)
                    c += 1
                else:
                    s += max_error_sq64
                i += nthreads

            # reduce the running totals, not just this chunk's, so ss[0] is
            # the score over everything scored so far
            ss[tid] = s
            sc[tid] = c
            cuda.syncthreads()
            stride = nthreads // 2
            while stride > 0:
                if tid < stride:
                    ss[tid] += ss[tid + stride]
                    sc[tid] += sc[tid + stride]
                cuda.syncthreads()
                stride //= 2

            if ss[0] >= bound:
                return ss[0], sc[0]
            start = stop
            # Chunks grow geometrically. Each check costs a block reduction,
            # and a hopeless model overwhelmingly bails on the *first* one, so
            # a small first chunk buys almost all of the saving while doubling
            # keeps the reduction count logarithmic in n rather than linear.
            chunk += chunk
            # the next chunk overwrites ss/sc; no thread may race into that
            # while another is still reading ss[0] above
            cuda.syncthreads()
        return ss[0], sc[0]

    @cuda.jit(cache=True)
    def _score_batch_kernel(data, params, models, counts, max_error_sq, bound,
                            out):
        # grid: one block per hypothesis. Reduces each hypothesis to its
        # best-by-score model and its best-by-inlier-count model, the two
        # criteria the CPU driver tracks separately (see build_ransac's
        # best_minimal_*).
        b = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        nthreads = cuda.blockDim.x

        ss = cuda.shared.array(THREADS_PER_BLOCK, float64)
        sc = cuda.shared.array(THREADS_PER_BLOCK, int64)
        dm64 = cuda.shared.array(derived_size, float64)
        dm32 = cuda.shared.array(derived_size, float32)
        m32 = cuda.shared.array(num_params, float32)
        ctl = cuda.shared.array(1, int64)

        mes32 = float32(max_error_sq)
        num = counts[b]
        bs = NO_MODEL
        bi = -1
        bn = 0
        mi = 0
        mix = -1

        for m in range(num):
            # the derived form is built in float64 from the float64 model and
            # only then rounded: [t]_x R (and K2^-T E K1^-1) are differences
            # of products, so forming them in float32 would lose digits the
            # per-point loop never gets back
            if tid == 0:
                ctl[0] = 1 if prepare(models[b, m], params, dm64) else 0
            cuda.syncthreads()
            if ctl[0] != 0:
                i = tid
                while i < derived_size:
                    dm32[i] = float32(dm64[i])
                    i += nthreads
                i = tid
                while i < num_params:
                    m32[i] = float32(models[b, m, i])
                    i += nthreads
            cuda.syncthreads()
            if ctl[0] == 0:
                # unusable model (e.g. a non-positive focal): the CPU scorers
                # report (1e300, 0) for exactly this case
                continue

            # bail against the tighter of the round's incumbent and this
            # sample's own best so far - the CPU driver likewise updates its
            # bound between the several models one minimal sample yields
            local_bound = bs if bs < bound else bound
            score, inliers = _score_one(data, dm32, m32, ss, sc, tid, nthreads,
                                        mes32, max_error_sq, local_bound)
            # tracked in every thread, not just thread 0, so `local_bound`
            # above stays block-uniform on the next model
            if score < bs:
                bs = score
                bi = m
                bn = inliers
            if inliers > mi:
                mi = inliers
                mix = m
            # the next model overwrites dm32/m32/ss/sc; no thread may race
            # ahead into that while others are still reducing
            cuda.syncthreads()

        if tid == 0:
            out[b, COL_SCORE] = bs
            out[b, COL_IDX] = bi
            out[b, COL_INL] = bn
            out[b, COL_MAX_INL] = mi
            out[b, COL_MAX_IDX] = mix

    return _score_batch_kernel


class ScoreBuffers():
    """Per-round reduction output, allocated once by the driver and reused.

    One packed `(batch, NUM_COLS)` float64 array rather than five separate
    typed arrays. The point is the readback: five arrays meant five
    synchronizing `copy_to_host` calls per round, and on a driver that charges
    ~100us per sync (Windows WDDM) that dominated the per-round cost. Packing
    them makes it one copy of one contiguous block.

    The four integer columns are indices and inlier counts, both far below
    2**53, so float64 holds them exactly; they are cast back on the host.
    """

    def __init__(self, batch):
        self.out = cuda.device_array((batch, NUM_COLS), dtype=np.float64)

    def to_host(self, count=None):
        # one synchronizing readback per round. Slicing the leading dimension
        # keeps the copy contiguous *and* transfers only the round's own rows
        # rather than the whole batch.
        rows = self.out if count is None else self.out[:count]
        h = rows.copy_to_host()
        return (h[:, COL_SCORE],
                h[:, COL_IDX].astype(np.int64),
                h[:, COL_INL].astype(np.int64),
                h[:, COL_MAX_INL].astype(np.int64),
                h[:, COL_MAX_IDX].astype(np.int64))
