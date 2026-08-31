"""Batched 5-point minimal solver on the GPU.

The solver math is not reimplemented here: `build_five_point_kernels` from
`solvers/essential.py` is instantiated with `cuda_jit`, so the nullspace,
constraint expansion, Gauss-Jordan reduction, Danilevsky characteristic
polynomial, Sturm root isolation and essential decomposition are compiled from
the same source the CPU path uses. Only the launch wrapper is CUDA-specific.

One thread per hypothesis. The solver is scalar, branchy, serial code - there
is no useful parallelism *inside* one minimal sample - so the parallelism is
purely across hypotheses, which is exactly what a RANSAC round has thousands
of.

Scratch lives in `cuda.local.array`. That is per-thread storage backed by
global memory, and CUDA interleaves it across the threads of a warp, so
thread-uniform accesses (which these are: every thread walks the same
elimination in lockstep) coalesce without any manual layout work. The
footprint is ~6.9 KB of local memory per thread, dominated by the 10x20
constraint matrix and the 11x11 Sturm chain.
"""

import numpy as np
from numba import cuda, float64, int64

from fastpose.cuda.backend import SOLVE_THREADS_PER_BLOCK, cuda_jit
from fastpose.solvers.essential import build_five_point_kernels

_GPU = build_five_point_kernels(cuda_jit)
_solve_5pt_core = _GPU['solve_5pt_core']

SAMPLE_SIZE = 5
NUM_PARAMS = 12
MAX_MODELS = 40


@cuda.jit(cache=True)
def _solve_batch_kernel(x1_x, x1_y, x2_x, x2_y, samples, models, counts):
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
        (x1_x, x1_y, x2_x, x2_y), samples[i], models[i],
        A, N, M, G, EEt, tr, ct, av, bv, coef, row, tmp_row, chain, roots, v,
        lo_stack, hi_stack, e, Rbuf, degs, slo_stack, shi_stack)


def solve_batch(data, samples, models, counts, stream=0):
    # launches the batched solver; all arguments are device arrays
    batch = samples.shape[0]
    blocks = (batch + SOLVE_THREADS_PER_BLOCK - 1) // SOLVE_THREADS_PER_BLOCK
    _solve_batch_kernel[blocks, SOLVE_THREADS_PER_BLOCK, stream](
        data[0], data[1], data[2], data[3], samples, models, counts)


def allocate_models(batch):
    # per-round model and count buffers, reused across rounds by the driver
    models = cuda.device_array((batch, MAX_MODELS, NUM_PARAMS), dtype=np.float64)
    counts = cuda.device_array(batch, dtype=np.int64)
    return models, counts
