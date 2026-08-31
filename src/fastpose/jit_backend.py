"""Decorator factories that let one kernel source compile for CPU or GPU.

A kernel written against these is decorated `@jit(fastmath=..., inline=...)`
rather than `@njit(...)` directly, so the same source can be built twice: once
with `cpu_jit` (numba `njit`, on-disk cached) and once with
`fastpose.cuda.backend.cuda_jit` (`cuda.jit(device=True)`).

Only the flags the kernels here actually vary are exposed. Anything a backend
cannot express is a signal the kernel is not portable and needs a separate
implementation - the scorer and the LM accumulate are reductions on the GPU
and are written separately for that reason, not squeezed through this shim.
"""

from numba import njit


def cpu_jit(fastmath=False, inline=False):
    # CPU instantiation: njit with the on-disk cache enabled, matching what
    # every kernel in solvers/ carried before the factories existed
    kwargs = {'cache': True, 'fastmath': fastmath}
    if inline:
        kwargs['inline'] = 'always'
    return njit(**kwargs)
