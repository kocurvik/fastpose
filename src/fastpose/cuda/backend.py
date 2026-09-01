"""CUDA side of the `jit` shim in `fastpose/jit_backend.py`.

Kept in its own module so importing it is the only thing that touches
`numba.cuda`; `fastpose.cuda.__init__` stays importable (and cheap) on a
machine with no GPU.
"""

import contextlib
import warnings

from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

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


@contextlib.contextmanager
def quiet_low_occupancy():
    """Silence numba's low-occupancy warning for the launches inside.

    numba warns on every launch whose grid is under 128 blocks, and several of
    this backend's launches are deliberately smaller: sampling and solving are
    one thread per hypothesis (32 and 64 blocks at the default batch), local
    optimization refines a single candidate per round (one block, see the note
    in cuda/ransac.py), and the last round of an adaptive run scores however
    many iterations are left. Only the main scorer's grid is a function of the
    batch, and that one is 4096 blocks by default. So the warning says nothing
    actionable here.

    Filtered by message rather than by category, so a NumbaPerformanceWarning
    that *is* actionable - the host-array copy one, say - still comes through.
    The leading `.*` is not decoration: numba's error classes render `str()`
    with ANSI highlighting, and the filter is matched against that, so a
    pattern anchored at 'Grid size' never fires.

    `warnings` filters are process-global state, so this is scoped to the
    driver's own launches and is one more reason `CudaRansacEstimator` is not
    thread-safe.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'.*Grid size \d+ will likely result in GPU '
                    r'under-utilization',
            category=NumbaPerformanceWarning)
        yield
