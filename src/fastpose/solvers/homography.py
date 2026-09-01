"""Minimal 4-point (DLT) solver for the homography.

The numba kernel `_solve_h4p` plugs into the generic RANSAC engine through
the `FourPointSolver` class; a pure-numpy reference implementation
(`four_point`) is kept for cross-checking and for the solver benchmarks.

`data` layout for this problem: the same tuple of four contiguous float64
columns (x1_x, x1_y, x2_x, x2_y) the epipolar problems use, so the scoring
loop stays SIMD-vectorizable.

Four correspondences give eight rows of the DLT constraint x2 x (H x1) = 0,
whose nullspace is one-dimensional, so the sample yields a single model. As
in the 7-point solver the nullspace comes from Gaussian elimination with
partial pivoting rather than a LAPACK SVD call - much faster at this size -
with the *last* unknown (H33) taken as the free variable. That choice is
degenerate only when the true H maps the first image's origin to infinity;
after the Hartley normalization every estimator applies, the origin is the
centroid of the correspondences and maps well inside the second image, so
H33 stays comfortably away from zero. The returned model is normalized to
unit Frobenius norm, which is also what keeps the scorer's inverse
well-scaled (see scorers/transfer.py).
"""

import math

import numpy as np
from numba import njit

from fastpose.jit_backend import cpu_jit


# ---------------------------------------------------------------------------
# numpy reference implementation
# ---------------------------------------------------------------------------

def four_point(p, q):
    # params:
    # p - (n, 2) array containing points x
    # q - (n, 2) array containing points x'
    # returns:
    # H - (3, 3) array with x' ~ H x, unit Frobenius norm, or None if the
    #     sample is degenerate
    n = len(p)
    A = np.zeros((2 * n, 9))
    x = p[:, 0]
    y = p[:, 1]
    xp = q[:, 0]
    yp = q[:, 1]
    A[0::2, 0] = -x
    A[0::2, 1] = -y
    A[0::2, 2] = -1.0
    A[0::2, 6] = xp * x
    A[0::2, 7] = xp * y
    A[0::2, 8] = xp
    A[1::2, 3] = -x
    A[1::2, 4] = -y
    A[1::2, 5] = -1.0
    A[1::2, 6] = yp * x
    A[1::2, 7] = yp * y
    A[1::2, 8] = yp

    _, s, vt = np.linalg.svd(A)
    # a well-posed sample leaves an 8x9 matrix of rank 8, i.e. a
    # one-dimensional nullspace; a vanishing second-smallest singular value
    # means the sample degenerated (three collinear points, say)
    if s[7] <= 1e-12 * s[0]:
        return None
    h = vt[-1, :]
    norm = np.linalg.norm(h)
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return (h / norm).reshape(3, 3)


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------

def build_four_point_kernels(jit):
    # one source, two backends: `cpu_jit` gives the njit kernels the RANSAC
    # driver calls, `cuda_jit` the device functions the GPU solve kernel
    # calls. `_solve_h4p_core` takes its scratch pre-shaped because numba's
    # runtime is host-only - see solvers/p3p.py for the full note.

    @jit(inline=True)
    def fill_homography_matrix(data, sample, A):
        # rows of the DLT constraint matrix: x2 x (H x1) = 0 linearized in the
        # 9 entries of H, two rows per sampled correspondence; A is (2k, 9)
        x1_x, x1_y, x2_x, x2_y = data
        for k in range(A.shape[0] // 2):
            i = sample[k]
            x = x1_x[i]
            y = x1_y[i]
            xp = x2_x[i]
            yp = x2_y[i]
            # x' (h6 x + h7 y + h8) - (h0 x + h1 y + h2) = 0
            A[2 * k, 0] = -x
            A[2 * k, 1] = -y
            A[2 * k, 2] = -1.0
            A[2 * k, 3] = 0.0
            A[2 * k, 4] = 0.0
            A[2 * k, 5] = 0.0
            A[2 * k, 6] = xp * x
            A[2 * k, 7] = xp * y
            A[2 * k, 8] = xp
            # y' (h6 x + h7 y + h8) - (h3 x + h4 y + h5) = 0
            A[2 * k + 1, 0] = 0.0
            A[2 * k + 1, 1] = 0.0
            A[2 * k + 1, 2] = 0.0
            A[2 * k + 1, 3] = -x
            A[2 * k + 1, 4] = -y
            A[2 * k + 1, 5] = -1.0
            A[2 * k + 1, 6] = yp * x
            A[2 * k + 1, 7] = yp * y
            A[2 * k + 1, 8] = yp

    @jit(fastmath=True)
    def _nullspace_h4p(A, h):
        # one-dimensional nullspace of the 8x9 DLT matrix via Gaussian
        # elimination with partial pivoting and back-substitution with the
        # free variable h[8] = 1, then normalized to unit Frobenius norm.
        # Returns False for a degenerate (rank-deficient) sample.
        for col in range(8):
            piv = col
            max_val = abs(A[col, col])
            for r in range(col + 1, 8):
                v = abs(A[r, col])
                if v > max_val:
                    max_val = v
                    piv = r
            if max_val < 1e-12:
                return False
            if piv != col:
                for c in range(col, 9):
                    t = A[col, c]
                    A[col, c] = A[piv, c]
                    A[piv, c] = t
            inv = 1.0 / A[col, col]
            for r in range(col + 1, 8):
                factor = A[r, col] * inv
                if factor != 0.0:
                    A[r, col] = 0.0
                    for c in range(col + 1, 9):
                        A[r, c] -= factor * A[col, c]

        h[8] = 1.0
        for r in range(7, -1, -1):
            s = A[r, 8]
            for c in range(r + 1, 8):
                s += A[r, c] * h[c]
            h[r] = -s / A[r, r]

        norm_sq = 0.0
        for j in range(9):
            norm_sq += h[j] * h[j]
        if not (norm_sq > 0.0) or not math.isfinite(norm_sq):
            return False
        inv = 1.0 / math.sqrt(norm_sq)
        for j in range(9):
            h[j] *= inv
        return True

    @jit(fastmath=True)
    def _solve_h4p_core(data, sample, models, A):
        # minimal 4-point solver; writes one model (flattened H, unit
        # Frobenius norm) into `models` and returns its count. Scratch is
        # passed in pre-shaped: see the note above the factory.
        fill_homography_matrix(data, sample, A)
        if not _nullspace_h4p(A, models[0]):
            return 0
        return 1

    return {
        'fill_homography_matrix': fill_homography_matrix,
        'nullspace_h4p': _nullspace_h4p,
        'solve_h4p_core': _solve_h4p_core,
    }


_CPU = build_four_point_kernels(cpu_jit)

# module-level names the tests import; these are the CPU kernels
fill_homography_matrix = _CPU['fill_homography_matrix']
_nullspace_h4p = _CPU['nullspace_h4p']
_solve_h4p_core = _CPU['solve_h4p_core']


@njit(cache=True, fastmath=True)
def _solve_h4p(data, sample, models, workspace):
    # RANSAC-driver entry point: carves the flat workspace into the pre-shaped
    # scratch `_solve_h4p_core` wants. CPU-only - `.reshape` does not compile
    # for CUDA, where the kernel allocates the same piece with
    # `cuda.local.array` instead.
    A = workspace[0:72].reshape(8, 9)
    return _solve_h4p_core(data, sample, models, A)


# ---------------------------------------------------------------------------
# pluggable component class
# ---------------------------------------------------------------------------

class FourPointSolver():
    # minimal DLT solver for the homography from 4 correspondences
    sample_size = 4
    num_params = 9
    max_models = 1
    workspace_size = 72
    solve = staticmethod(_solve_h4p)
