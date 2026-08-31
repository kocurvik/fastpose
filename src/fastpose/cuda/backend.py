"""CUDA side of the `jit` shim in `fastpose/jit_backend.py`.

Kept in its own module so importing it is the only thing that touches
`numba.cuda`; `fastpose.cuda.__init__` stays importable (and cheap) on a
machine with no GPU.
"""

from numba import cuda

# Threads per block for the reduction kernels (scorer, LM accumulate). Must be
# a power of two - the tree reductions halve the stride - and at least a warp.
# 128 measured a reasonable default: large enough to hide the global-memory
# latency of streaming the correspondence columns, small enough that a block
# per hypothesis still fills the machine at modest batch sizes.
THREADS_PER_BLOCK = 128

# Threads per block for the one-thread-per-hypothesis solve kernel. The 5-point
# solver carries ~6.9 KB of per-thread local memory (see solvers.py), so a
# small block keeps the per-SM local-memory footprint sane.
SOLVE_THREADS_PER_BLOCK = 64


def cuda_jit(fastmath=False, inline=False):
    # GPU instantiation of a kernel written against the shim. Device functions
    # are inlined into the kernel that calls them, so there is no separate
    # on-disk cache entry for them - `cache=True` goes on the kernels instead.
    return cuda.jit(device=True, fastmath=fastmath, inline=inline)
