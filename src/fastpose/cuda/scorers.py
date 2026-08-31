"""Batch-parallel truncated Sampson scorer with cheirality, on the GPU.

One block per hypothesis; the block's threads stride over the correspondences
and tree-reduce their partial (score, inlier count) in shared memory. A block
loops over the several models one minimal sample can produce, keeping its own
running best in thread 0's registers, so the host only ever reads back five
scalars per hypothesis instead of the full `batch x max_models` score table.

Precision is mixed. The coordinate columns, the essential matrix and every
per-point expression are float32, because this loop is the O(iterations x
matches) hot spot and its inner arithmetic is short and well-scaled. Three
things stay float64: the pose model, the formation of E = [t]_x R from it (a
difference of products, rounded only afterwards), and the score accumulator,
so the sum over all correspondences does not drift. Measured against a full
float64 scorer on 200 real models at 16k matches, inlier counts agree on ~99.5%
of models (worst case one point) and the model ranking is unchanged.

Two further deliberate differences from `scorers/sampson.py`, both inherent to
the parallel shape rather than choices:

- **No early bail-out.** The CPU scorer stops as soon as a partial truncated
  score exceeds the incumbent, which is exact but strictly sequential. Here
  every model is scored over every point. That is the same trade
  `build_parallel_ransac` already makes and has the same consequence: a model
  the CPU driver would have abandoned mid-scan comes back with a *full* inlier
  count rather than a partial one, which can flip the `more_inliers` test and
  select a different local-optimization candidate.
- **Summation order.** A tree reduction adds the per-point contributions in a
  different order than a sequential loop, so scores differ in the last couple
  of ulps. Inlier *counts* are exact integers and agree exactly, except where
  a point sits within rounding distance of the threshold.

The per-point math itself is not reimplemented: `sampson_residual`,
`essential_from_pose` and `cheirality_ok` come from the same factory in
`scorers/sampson.py` that the CPU scorer uses, compiled here as device
functions. `tests/test_cuda.py` checks the two agree point for point.
"""

import numpy as np
from numba import cuda, float32, float64, int64

from fastpose.cuda.backend import THREADS_PER_BLOCK, cuda_jit
from fastpose.scorers.sampson import MIN_DEPTH, build_sampson_point_kernels

# float32 per-point kernels; the score itself accumulates in float64
_F32 = build_sampson_point_kernels(cuda_jit, real=float32)
_sampson_residual = _F32['sampson_residual']
_cheirality_ok = _F32['cheirality_ok']

# float64, used once per model to turn the pose into E before it is rounded
_F64 = build_sampson_point_kernels(cuda_jit, real=float64)
_essential_from_pose = _F64['essential_from_pose']

# score reported for a model slot that holds no model (m >= counts[b]); the
# same 1e300 failure sentinel the CPU kernels use, never inf
NO_MODEL = 1e300


@cuda.jit(device=True, fastmath=True)
def _score_one(x1_x, x1_y, x2_x, x2_y, pose, e, ss, sc, tid, nthreads,
               max_error_sq, max_error_sq64):
    # block-wide truncated Sampson + cheirality score of one pose model;
    # returns (score, num_inliers) valid in thread 0 only.
    #
    # Mixed precision: the coordinate columns, E, the pose and every per-point
    # expression are float32; `s` is float64, so the O(n) sum accumulates
    # without drift. The truncation constant is added in float64 for the same
    # reason - it is the term most points contribute.
    n = x1_x.shape[0]
    s = 0.0
    c = 0
    i = tid
    while i < n:
        x = x1_x[i]
        y = x1_y[i]
        xp = x2_x[i]
        yp = x2_y[i]
        r2, den = _sampson_residual(e, x, y, xp, yp)
        if (den > float32(0.0) and r2 < max_error_sq * den
                and _cheirality_ok(pose, x, y, xp, yp, float32(MIN_DEPTH))):
            s += float64(r2 / den)
            c += 1
        else:
            s += max_error_sq64
        i += nthreads
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
    return ss[0], sc[0]


# column layout of the packed per-hypothesis reduction output
COL_SCORE = 0
COL_IDX = 1
COL_INL = 2
COL_MAX_INL = 3
COL_MAX_IDX = 4
NUM_COLS = 5


@cuda.jit(cache=True)
def _score_batch_kernel(x1_x, x1_y, x2_x, x2_y, models, counts, max_error_sq,
                        out):
    # grid: one block per hypothesis. Reduces each hypothesis to its
    # best-by-score model and its best-by-inlier-count model, the two criteria
    # the CPU driver tracks separately (see build_ransac's best_minimal_*).
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x

    ss = cuda.shared.array(THREADS_PER_BLOCK, float64)
    sc = cuda.shared.array(THREADS_PER_BLOCK, int64)
    e64 = cuda.shared.array(9, float64)
    e = cuda.shared.array(9, float32)
    pose = cuda.shared.array(12, float32)

    mes32 = float32(max_error_sq)
    num = counts[b]
    bs = NO_MODEL
    bi = -1
    bn = 0
    mi = 0
    mix = -1

    for m in range(num):
        # E is formed in float64 from the float64 model and only then rounded:
        # [t]_x R is a difference of products, so forming it in float32 would
        # lose digits the per-point loop never gets back
        if tid == 0:
            _essential_from_pose(models[b, m], e64)
        cuda.syncthreads()
        if tid < 12:
            pose[tid] = float32(models[b, m, tid])
        if tid < 9:
            e[tid] = float32(e64[tid])
        cuda.syncthreads()

        score, inliers = _score_one(x1_x, x1_y, x2_x, x2_y, pose, e, ss, sc,
                                    tid, nthreads, mes32, max_error_sq)
        if tid == 0:
            if score < bs:
                bs = score
                bi = m
                bn = inliers
            if inliers > mi:
                mi = inliers
                mix = m
        # the next model overwrites `pose`/`e`/`ss`/`sc`; no thread may race
        # ahead into that while others are still reducing
        cuda.syncthreads()

    if tid == 0:
        out[b, COL_SCORE] = bs
        out[b, COL_IDX] = bi
        out[b, COL_INL] = bn
        out[b, COL_MAX_INL] = mi
        out[b, COL_MAX_IDX] = mix


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
        # keeps the copy contiguous *and* transfers only the rounds's own rows
        # rather than the whole batch.
        rows = self.out if count is None else self.out[:count]
        h = rows.copy_to_host()
        return (h[:, COL_SCORE],
                h[:, COL_IDX].astype(np.int64),
                h[:, COL_INL].astype(np.int64),
                h[:, COL_MAX_INL].astype(np.int64),
                h[:, COL_MAX_IDX].astype(np.int64))


def score_batch(data32, models, counts, max_error_sq, buffers, batch,
                stream=0):
    # launches the batched scorer over `batch` hypotheses. `data32` is the
    # float32 coordinate columns; `models` stays float64.
    _score_batch_kernel[batch, THREADS_PER_BLOCK, stream](
        data32[0], data32[1], data32[2], data32[3], models, counts,
        max_error_sq, buffers.out)
