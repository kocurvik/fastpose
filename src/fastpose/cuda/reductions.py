"""Block-level reduction and linear-algebra helpers shared by every kernel.

These are the pieces of the first (5-point) port that turned out to be
problem-agnostic: the three tree reductions the scorer and the LM accumulate
run, and the damped Cholesky solve of the normal equations. They are split
out of `cuda/refiners.py` so a new problem reuses them rather than growing a
second copy with a different `NUM_TANGENT` baked in.

Every reduction here is called by *all* threads of the block and contains a
`cuda.syncthreads()`, so a caller must never place one inside a branch that
only some threads take (see the block-uniformity note in Instructions.md).
"""

import math

from numba import cuda


@cuda.jit(device=True, inline=True)
def reduce_sum(red, tid, nthreads, width):
    # in-place tree reduction of red[:, :width] into row 0
    stride = nthreads // 2
    while stride > 0:
        if tid < stride:
            for q in range(width):
                red[tid, q] += red[tid + stride, q]
        cuda.syncthreads()
        stride //= 2


@cuda.jit(device=True, inline=True)
def reduce_scalar(ss, tid, nthreads):
    stride = nthreads // 2
    while stride > 0:
        if tid < stride:
            ss[tid] += ss[tid + stride]
        cuda.syncthreads()
        stride //= 2


@cuda.jit(device=True, inline=True)
def reduce_count(sc, tid, nthreads):
    stride = nthreads // 2
    while stride > 0:
        if tid < stride:
            sc[tid] += sc[tid + stride]
        cuda.syncthreads()
        stride //= 2


def build_solve_damped(n):
    # Cholesky solve of A delta = -Jtr for the n x n damped normal equations,
    # in place on A. Mirrors `_solve_damped` in refiners/lm.py, including its
    # refusal to proceed on a non-positive or non-finite pivot - deliberately
    # not fastmath so the pivot check is honest about NaN.
    #
    # `n` is a closure constant rather than an argument so numba unrolls the
    # inner loops; one kernel per NUM_TANGENT, memoized below.
    @cuda.jit(device=True)
    def _solve_damped(A, Jtr, delta):
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

    return _solve_damped


_DAMPED = {}


def get_solve_damped(n):
    # memoized so two problems with the same NUM_TANGENT share one device
    # function instead of compiling it twice
    if n not in _DAMPED:
        _DAMPED[n] = build_solve_damped(n)
    return _DAMPED[n]
