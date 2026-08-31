"""P4Pf minimal solver: absolute pose plus unknown focal length from four
2D-3D correspondences.

Port of poselib's p4pf.cc and misc/re3q3.cc (BSD-3-Clause, Copyright (c)
2020 Viktor Larsson). The first two camera rows are parametrized by the
4-dimensional nullspace of the eight cross-product constraints, the third
row follows linearly, and the three rotation-orthogonality constraints form
a 3Q3 system (three quadrics in three unknowns). The 3Q3 solver eliminates
the best-conditioned variable, reduces to the degree-8 determinant
polynomial of the resulting 3x3 system and brackets its real roots with the
shared Sturm machinery from the 5-point solver, followed by Newton polish.
The nullspace is computed by Gaussian elimination + modified Gram-Schmidt
like in the 5-point solver (no LAPACK SVD).

Like poselib with filter_solutions=true, candidate poses are
cheirality-checked on the sample and the solution whose row norms are most
consistent (fx ~ fy, square pixels) is kept, so the solver emits at most
one model `[R | t | f]` (13 flat parameters) per sample. One deviation from
poselib: the random change-of-variables fallback for degenerate 3Q3 systems
is omitted (degenerate samples are simply rejected by the scoring).

`data` layout: five contiguous float64 columns (x_x, x_y, X_x, X_y, X_z)
with the 2D points in pixel coordinates relative to the principal point.
"""

import math

import numpy as np
from numba import njit

from fastpose.jit_backend import cpu_jit
from fastpose.solvers.essential import _real_roots_sturm

MODEL_SIZE = 13


def build_p4pf_kernels(jit, real_roots_sturm):
    # `real_roots_sturm` comes from the caller rather than from a second
    # `build_five_point_kernels` call so the CPU build reuses the very kernel
    # object solvers/essential.py already instantiated (a second one would
    # duplicate its on-disk cache entry) and the CUDA build reuses the one
    # cuda/problems/common.py holds.


    @jit(inline=True, fastmath=True)
    def _det3_cols(c, j0, j1, j2):
        # determinant of the 3x3 matrix [c[:, j0] | c[:, j1] | c[:, j2]]
        return (c[0, j0] * (c[1, j1] * c[2, j2] - c[1, j2] * c[2, j1])
                - c[0, j1] * (c[1, j0] * c[2, j2] - c[1, j2] * c[2, j0])
                + c[0, j2] * (c[1, j0] * c[2, j1] - c[1, j1] * c[2, j0]))


    @jit(fastmath=True)
    def _solve_3xn_neg(A, P, n):
        # P[:, :n] <- -A^{-1} P[:, :n] in place (Gaussian elimination with
        # partial pivoting); False only for an exactly singular pivot
        for col in range(3):
            piv = col
            max_val = abs(A[col, col])
            for r in range(col + 1, 3):
                v = abs(A[r, col])
                if v > max_val:
                    max_val = v
                    piv = r
            if max_val == 0.0:
                return False
            if piv != col:
                for c in range(3):
                    t = A[col, c]
                    A[col, c] = A[piv, c]
                    A[piv, c] = t
                for c in range(n):
                    t = P[col, c]
                    P[col, c] = P[piv, c]
                    P[piv, c] = t
            inv = 1.0 / A[col, col]
            for r in range(col + 1, 3):
                factor = A[r, col] * inv
                if factor != 0.0:
                    for c in range(col + 1, 3):
                        A[r, c] -= factor * A[col, c]
                    for c in range(n):
                        P[r, c] -= factor * P[col, c]
        for r in range(2, -1, -1):
            inv = 1.0 / A[r, r]
            for c in range(n):
                # rows below r already hold the negated solution -x_k
                s = P[r, c]
                for k in range(r + 1, 3):
                    s += A[r, k] * P[k, c]
                P[r, c] = -s * inv
        return True


    @jit(fastmath=True)
    def _refine_3q3(coeffs, solutions, n_sols):
        # a few Newton steps on the three quadrics; monomial order is
        # [x^2, xy, xz, y^2, yz, z^2, x, y, z, 1]
        for i in range(n_sols):
            x = solutions[0, i]
            y = solutions[1, i]
            z = solutions[2, i]
            for _ in range(5):
                r0 = (coeffs[0, 0] * x * x + coeffs[0, 1] * x * y
                      + coeffs[0, 2] * x * z + coeffs[0, 3] * y * y
                      + coeffs[0, 4] * y * z + coeffs[0, 5] * z * z
                      + coeffs[0, 6] * x + coeffs[0, 7] * y
                      + coeffs[0, 8] * z + coeffs[0, 9])
                r1 = (coeffs[1, 0] * x * x + coeffs[1, 1] * x * y
                      + coeffs[1, 2] * x * z + coeffs[1, 3] * y * y
                      + coeffs[1, 4] * y * z + coeffs[1, 5] * z * z
                      + coeffs[1, 6] * x + coeffs[1, 7] * y
                      + coeffs[1, 8] * z + coeffs[1, 9])
                r2 = (coeffs[2, 0] * x * x + coeffs[2, 1] * x * y
                      + coeffs[2, 2] * x * z + coeffs[2, 3] * y * y
                      + coeffs[2, 4] * y * z + coeffs[2, 5] * z * z
                      + coeffs[2, 6] * x + coeffs[2, 7] * y
                      + coeffs[2, 8] * z + coeffs[2, 9])
                # spelled out rather than `max(...)`: numba 0.67 cannot
                # compile the builtin for CUDA at all (see Instructions.md
                # on the `cuda-backend` branch)
                worst = abs(r0)
                ar1 = abs(r1)
                if ar1 > worst:
                    worst = ar1
                ar2 = abs(r2)
                if ar2 > worst:
                    worst = ar2
                if worst < 1e-8:
                    break
                j00 = 2.0 * coeffs[0, 0] * x + coeffs[0, 1] * y + coeffs[0, 2] * z + coeffs[0, 6]
                j01 = coeffs[0, 1] * x + 2.0 * coeffs[0, 3] * y + coeffs[0, 4] * z + coeffs[0, 7]
                j02 = coeffs[0, 2] * x + coeffs[0, 4] * y + 2.0 * coeffs[0, 5] * z + coeffs[0, 8]
                j10 = 2.0 * coeffs[1, 0] * x + coeffs[1, 1] * y + coeffs[1, 2] * z + coeffs[1, 6]
                j11 = coeffs[1, 1] * x + 2.0 * coeffs[1, 3] * y + coeffs[1, 4] * z + coeffs[1, 7]
                j12 = coeffs[1, 2] * x + coeffs[1, 4] * y + 2.0 * coeffs[1, 5] * z + coeffs[1, 8]
                j20 = 2.0 * coeffs[2, 0] * x + coeffs[2, 1] * y + coeffs[2, 2] * z + coeffs[2, 6]
                j21 = coeffs[2, 1] * x + 2.0 * coeffs[2, 3] * y + coeffs[2, 4] * z + coeffs[2, 7]
                j22 = coeffs[2, 2] * x + coeffs[2, 4] * y + 2.0 * coeffs[2, 5] * z + coeffs[2, 8]
                det = (j00 * (j11 * j22 - j12 * j21)
                       - j01 * (j10 * j22 - j12 * j20)
                       + j02 * (j10 * j21 - j11 * j20))
                if det == 0.0:
                    break
                inv = 1.0 / det
                dx = ((j11 * j22 - j12 * j21) * r0
                      + (j02 * j21 - j01 * j22) * r1
                      + (j01 * j12 - j02 * j11) * r2) * inv
                dy = ((j12 * j20 - j10 * j22) * r0
                      + (j00 * j22 - j02 * j20) * r1
                      + (j02 * j10 - j00 * j12) * r2) * inv
                dz = ((j10 * j21 - j11 * j20) * r0
                      + (j01 * j20 - j00 * j21) * r1
                      + (j00 * j11 - j01 * j10) * r2) * inv
                x -= dx
                y -= dy
                z -= dz
            solutions[0, i] = x
            solutions[1, i] = y
            solutions[2, i] = z


    @jit(fastmath=True)
    def _re3q3_core(coeffs, solutions, A, P, c, chain, roots, lo_stack,
                    hi_stack, degs, slo_stack, shi_stack):
        # real solutions of three quadrics in three unknowns; monomial order is
        # [x^2, xy, xz, y^2, yz, z^2, x, y, z, 1]. Writes up to 8 solutions as
        # columns of `solutions` (3, 8) and returns their count. Scratch is
        # passed in pre-shaped: see the note above the factory.
        detx = abs(_det3_cols(coeffs, 3, 5, 4))
        dety = abs(_det3_cols(coeffs, 0, 5, 2))
        detz = abs(_det3_cols(coeffs, 3, 0, 1))
        elim_var = 0
        det = detx
        if det < dety:
            det = dety
            elim_var = 1
        if det < detz:
            det = detz
            elim_var = 2

        if elim_var == 0:
            for k in range(3):
                A[k, 0] = coeffs[k, 3]
                A[k, 1] = coeffs[k, 5]
                A[k, 2] = coeffs[k, 4]
                P[k, 0] = coeffs[k, 0]
                P[k, 1] = coeffs[k, 1]
                P[k, 2] = coeffs[k, 2]
                P[k, 3] = coeffs[k, 6]
                P[k, 4] = coeffs[k, 7]
                P[k, 5] = coeffs[k, 8]
                P[k, 6] = coeffs[k, 9]
        elif elim_var == 1:
            for k in range(3):
                A[k, 0] = coeffs[k, 0]
                A[k, 1] = coeffs[k, 5]
                A[k, 2] = coeffs[k, 2]
                P[k, 0] = coeffs[k, 3]
                P[k, 1] = coeffs[k, 1]
                P[k, 2] = coeffs[k, 4]
                P[k, 3] = coeffs[k, 7]
                P[k, 4] = coeffs[k, 6]
                P[k, 5] = coeffs[k, 8]
                P[k, 6] = coeffs[k, 9]
        else:
            for k in range(3):
                A[k, 0] = coeffs[k, 3]
                A[k, 1] = coeffs[k, 0]
                A[k, 2] = coeffs[k, 1]
                P[k, 0] = coeffs[k, 5]
                P[k, 1] = coeffs[k, 4]
                P[k, 2] = coeffs[k, 2]
                P[k, 3] = coeffs[k, 8]
                P[k, 4] = coeffs[k, 7]
                P[k, 5] = coeffs[k, 6]
                P[k, 6] = coeffs[k, 9]

        if not _solve_3xn_neg(A, P, 7):
            return 0

        a11 = P[0, 1] * P[2, 1] + P[0, 2] * P[1, 1] - P[2, 1] * P[0, 1] - P[2, 2] * P[2, 1] - P[2, 0]
        a12 = (P[0, 1] * P[2, 4] + P[0, 4] * P[2, 1] + P[0, 2] * P[1, 4] + P[0, 5] * P[1, 1]
               - P[2, 1] * P[0, 4] - P[2, 4] * P[0, 1] - P[2, 2] * P[2, 4] - P[2, 5] * P[2, 1] - P[2, 3])
        a13 = P[0, 4] * P[2, 4] + P[0, 5] * P[1, 4] - P[2, 4] * P[0, 4] - P[2, 5] * P[2, 4] - P[2, 6]
        a14 = P[0, 1] * P[2, 2] + P[0, 2] * P[1, 2] - P[2, 1] * P[0, 2] - P[2, 2] * P[2, 2] + P[0, 0]
        a15 = (P[0, 1] * P[2, 5] + P[0, 4] * P[2, 2] + P[0, 2] * P[1, 5] + P[0, 5] * P[1, 2]
               - P[2, 1] * P[0, 5] - P[2, 4] * P[0, 2] - P[2, 2] * P[2, 5] - P[2, 5] * P[2, 2] + P[0, 3])
        a16 = P[0, 4] * P[2, 5] + P[0, 5] * P[1, 5] - P[2, 4] * P[0, 5] - P[2, 5] * P[2, 5] + P[0, 6]
        a17 = P[0, 1] * P[2, 0] + P[0, 2] * P[1, 0] - P[2, 1] * P[0, 0] - P[2, 2] * P[2, 0]
        a18 = (P[0, 1] * P[2, 3] + P[0, 4] * P[2, 0] + P[0, 2] * P[1, 3] + P[0, 5] * P[1, 0]
               - P[2, 1] * P[0, 3] - P[2, 4] * P[0, 0] - P[2, 2] * P[2, 3] - P[2, 5] * P[2, 0])
        a19 = (P[0, 1] * P[2, 6] + P[0, 4] * P[2, 3] + P[0, 2] * P[1, 6] + P[0, 5] * P[1, 3]
               - P[2, 1] * P[0, 6] - P[2, 4] * P[0, 3] - P[2, 2] * P[2, 6] - P[2, 5] * P[2, 3])
        a110 = P[0, 4] * P[2, 6] + P[0, 5] * P[1, 6] - P[2, 4] * P[0, 6] - P[2, 5] * P[2, 6]

        a21 = P[2, 1] * P[2, 1] + P[2, 2] * P[1, 1] - P[1, 1] * P[0, 1] - P[1, 2] * P[2, 1] - P[1, 0]
        a22 = (P[2, 1] * P[2, 4] + P[2, 4] * P[2, 1] + P[2, 2] * P[1, 4] + P[2, 5] * P[1, 1]
               - P[1, 1] * P[0, 4] - P[1, 4] * P[0, 1] - P[1, 2] * P[2, 4] - P[1, 5] * P[2, 1] - P[1, 3])
        a23 = P[2, 4] * P[2, 4] + P[2, 5] * P[1, 4] - P[1, 4] * P[0, 4] - P[1, 5] * P[2, 4] - P[1, 6]
        a24 = P[2, 1] * P[2, 2] + P[2, 2] * P[1, 2] - P[1, 1] * P[0, 2] - P[1, 2] * P[2, 2] + P[2, 0]
        a25 = (P[2, 1] * P[2, 5] + P[2, 4] * P[2, 2] + P[2, 2] * P[1, 5] + P[2, 5] * P[1, 2]
               - P[1, 1] * P[0, 5] - P[1, 4] * P[0, 2] - P[1, 2] * P[2, 5] - P[1, 5] * P[2, 2] + P[2, 3])
        a26 = P[2, 4] * P[2, 5] + P[2, 5] * P[1, 5] - P[1, 4] * P[0, 5] - P[1, 5] * P[2, 5] + P[2, 6]
        a27 = P[2, 1] * P[2, 0] + P[2, 2] * P[1, 0] - P[1, 1] * P[0, 0] - P[1, 2] * P[2, 0]
        a28 = (P[2, 1] * P[2, 3] + P[2, 4] * P[2, 0] + P[2, 2] * P[1, 3] + P[2, 5] * P[1, 0]
               - P[1, 1] * P[0, 3] - P[1, 4] * P[0, 0] - P[1, 2] * P[2, 3] - P[1, 5] * P[2, 0])
        a29 = (P[2, 1] * P[2, 6] + P[2, 4] * P[2, 3] + P[2, 2] * P[1, 6] + P[2, 5] * P[1, 3]
               - P[1, 1] * P[0, 6] - P[1, 4] * P[0, 3] - P[1, 2] * P[2, 6] - P[1, 5] * P[2, 3])
        a210 = P[2, 4] * P[2, 6] + P[2, 5] * P[1, 6] - P[1, 4] * P[0, 6] - P[1, 5] * P[2, 6]

        t2 = P[2, 1] * P[2, 1]
        t3 = P[2, 2] * P[2, 2]
        t6 = P[0, 1] * P[1, 4] + P[0, 4] * P[1, 1]
        t9 = P[0, 2] * P[1, 5] + P[0, 5] * P[1, 2]
        t12 = P[0, 1] * P[1, 5] + P[0, 4] * P[1, 2]
        t15 = P[0, 2] * P[1, 4] + P[0, 5] * P[1, 1]
        t18 = P[2, 1] * P[2, 5] + P[2, 2] * P[2, 4]
        t19 = P[2, 4] * P[2, 4]
        t20 = P[2, 5] * P[2, 5]
        a31 = (P[0, 0] * P[1, 1] + P[0, 1] * P[1, 0] - P[2, 0] * P[2, 1] * 2.0 - P[0, 1] * t2 - P[1, 1] * t3
               - P[2, 2] * t2 * 2.0 + (P[0, 1] * P[0, 1]) * P[1, 1] + P[0, 2] * P[1, 1] * P[1, 2]
               + P[0, 1] * P[1, 2] * P[2, 1] + P[0, 2] * P[1, 1] * P[2, 1])
        a32 = (P[0, 0] * P[1, 4] + P[0, 1] * P[1, 3] + P[0, 3] * P[1, 1] + P[0, 4] * P[1, 0]
               - P[2, 0] * P[2, 4] * 2.0 - P[2, 1] * P[2, 3] * 2.0 - P[0, 4] * t2 + P[0, 1] * t6 - P[1, 4] * t3
               + P[1, 1] * t9 + P[2, 1] * t12 + P[2, 1] * t15 - P[2, 1] * t18 * 2.0 + P[0, 1] * P[0, 4] * P[1, 1]
               + P[0, 2] * P[1, 2] * P[1, 4] + P[0, 1] * P[1, 2] * P[2, 4] + P[0, 2] * P[1, 1] * P[2, 4]
               - P[0, 1] * P[2, 1] * P[2, 4] * 2.0 - P[1, 1] * P[2, 2] * P[2, 5] * 2.0
               - P[2, 1] * P[2, 2] * P[2, 4] * 2.0)
        a33 = (P[0, 1] * P[1, 6] + P[0, 3] * P[1, 4] + P[0, 4] * P[1, 3] + P[0, 6] * P[1, 1]
               - P[2, 1] * P[2, 6] * 2.0 - P[2, 3] * P[2, 4] * 2.0 + P[0, 4] * t6 - P[0, 1] * t19 + P[1, 4] * t9
               - P[1, 1] * t20 + P[2, 4] * t12 + P[2, 4] * t15 - P[2, 4] * t18 * 2.0 + P[0, 1] * P[0, 4] * P[1, 4]
               + P[0, 5] * P[1, 1] * P[1, 5] + P[0, 4] * P[1, 5] * P[2, 1] + P[0, 5] * P[1, 4] * P[2, 1]
               - P[0, 4] * P[2, 1] * P[2, 4] * 2.0 - P[1, 4] * P[2, 2] * P[2, 5] * 2.0
               - P[2, 1] * P[2, 4] * P[2, 5] * 2.0)
        a34 = (P[0, 4] * P[1, 6] + P[0, 6] * P[1, 4] - P[2, 4] * P[2, 6] * 2.0 - P[0, 4] * t19 - P[1, 4] * t20
               - P[2, 5] * t19 * 2.0 + (P[0, 4] * P[0, 4]) * P[1, 4] + P[0, 5] * P[1, 4] * P[1, 5]
               + P[0, 4] * P[1, 5] * P[2, 4] + P[0, 5] * P[1, 4] * P[2, 4])
        a35 = (P[0, 0] * P[1, 2] + P[0, 2] * P[1, 0] - P[2, 0] * P[2, 2] * 2.0 - P[0, 2] * t2 - P[1, 2] * t3
               - P[2, 1] * t3 * 2.0 + P[0, 2] * (P[1, 2] * P[1, 2]) + P[0, 1] * P[0, 2] * P[1, 1]
               + P[0, 1] * P[1, 2] * P[2, 2] + P[0, 2] * P[1, 1] * P[2, 2])
        a36 = (P[0, 0] * P[1, 5] + P[0, 2] * P[1, 3] + P[0, 3] * P[1, 2] + P[0, 5] * P[1, 0]
               - P[2, 0] * P[2, 5] * 2.0 - P[2, 2] * P[2, 3] * 2.0 - P[0, 5] * t2 + P[0, 2] * t6 - P[1, 5] * t3
               + P[1, 2] * t9 + P[2, 2] * t12 + P[2, 2] * t15 - P[2, 2] * t18 * 2.0 + P[0, 1] * P[0, 5] * P[1, 1]
               + P[0, 2] * P[1, 2] * P[1, 5] + P[0, 1] * P[1, 2] * P[2, 5] + P[0, 2] * P[1, 1] * P[2, 5]
               - P[0, 2] * P[2, 1] * P[2, 4] * 2.0 - P[1, 2] * P[2, 2] * P[2, 5] * 2.0
               - P[2, 1] * P[2, 2] * P[2, 5] * 2.0)
        a37 = (P[0, 2] * P[1, 6] + P[0, 3] * P[1, 5] + P[0, 5] * P[1, 3] + P[0, 6] * P[1, 2]
               - P[2, 2] * P[2, 6] * 2.0 - P[2, 3] * P[2, 5] * 2.0 + P[0, 5] * t6 - P[0, 2] * t19 + P[1, 5] * t9
               - P[1, 2] * t20 + P[2, 5] * t12 + P[2, 5] * t15 - P[2, 5] * t18 * 2.0 + P[0, 2] * P[0, 4] * P[1, 4]
               + P[0, 5] * P[1, 2] * P[1, 5] + P[0, 4] * P[1, 5] * P[2, 2] + P[0, 5] * P[1, 4] * P[2, 2]
               - P[0, 5] * P[2, 1] * P[2, 4] * 2.0 - P[1, 5] * P[2, 2] * P[2, 5] * 2.0
               - P[2, 2] * P[2, 4] * P[2, 5] * 2.0)
        a38 = (P[0, 5] * P[1, 6] + P[0, 6] * P[1, 5] - P[2, 5] * P[2, 6] * 2.0 - P[0, 5] * t19 - P[1, 5] * t20
               - P[2, 4] * t20 * 2.0 + P[0, 5] * (P[1, 5] * P[1, 5]) + P[0, 4] * P[0, 5] * P[1, 4]
               + P[0, 4] * P[1, 5] * P[2, 5] + P[0, 5] * P[1, 4] * P[2, 5])
        a39 = (P[0, 0] * P[1, 0] - P[0, 0] * t2 - P[1, 0] * t3 - P[2, 0] * P[2, 0] + P[0, 0] * P[0, 1] * P[1, 1]
               + P[0, 2] * P[1, 0] * P[1, 2] + P[0, 1] * P[1, 2] * P[2, 0] + P[0, 2] * P[1, 1] * P[2, 0]
               - P[2, 0] * P[2, 1] * P[2, 2] * 2.0)
        a310 = (P[0, 0] * P[1, 3] + P[0, 3] * P[1, 0] - P[2, 0] * P[2, 3] * 2.0 - P[0, 3] * t2 + P[0, 0] * t6
                - P[1, 3] * t3 + P[1, 0] * t9 + P[2, 0] * t12 + P[2, 0] * t15 - P[2, 0] * t18 * 2.0
                + P[0, 1] * P[0, 3] * P[1, 1] + P[0, 2] * P[1, 2] * P[1, 3] + P[0, 1] * P[1, 2] * P[2, 3]
                + P[0, 2] * P[1, 1] * P[2, 3] - P[0, 0] * P[2, 1] * P[2, 4] * 2.0
                - P[1, 0] * P[2, 2] * P[2, 5] * 2.0 - P[2, 1] * P[2, 2] * P[2, 3] * 2.0)
        a311 = (P[0, 0] * P[1, 6] + P[0, 3] * P[1, 3] + P[0, 6] * P[1, 0] - P[2, 0] * P[2, 6] * 2.0 - P[0, 6] * t2
                + P[0, 3] * t6 - P[0, 0] * t19 - P[1, 6] * t3 + P[1, 3] * t9 - P[1, 0] * t20 + P[2, 3] * t12
                + P[2, 3] * t15 - P[2, 3] * t18 * 2.0 - P[2, 3] * P[2, 3] + P[0, 0] * P[0, 4] * P[1, 4]
                + P[0, 1] * P[0, 6] * P[1, 1] + P[0, 2] * P[1, 2] * P[1, 6] + P[0, 5] * P[1, 0] * P[1, 5]
                + P[0, 1] * P[1, 2] * P[2, 6] + P[0, 2] * P[1, 1] * P[2, 6] + P[0, 4] * P[1, 5] * P[2, 0]
                + P[0, 5] * P[1, 4] * P[2, 0] - P[0, 3] * P[2, 1] * P[2, 4] * 2.0
                - P[1, 3] * P[2, 2] * P[2, 5] * 2.0 - P[2, 0] * P[2, 4] * P[2, 5] * 2.0
                - P[2, 1] * P[2, 2] * P[2, 6] * 2.0)
        a312 = (P[0, 3] * P[1, 6] + P[0, 6] * P[1, 3] - P[2, 3] * P[2, 6] * 2.0 + P[0, 6] * t6 - P[0, 3] * t19
                + P[1, 6] * t9 - P[1, 3] * t20 + P[2, 6] * t12 + P[2, 6] * t15 - P[2, 6] * t18 * 2.0
                + P[0, 3] * P[0, 4] * P[1, 4] + P[0, 5] * P[1, 3] * P[1, 5] + P[0, 4] * P[1, 5] * P[2, 3]
                + P[0, 5] * P[1, 4] * P[2, 3] - P[0, 6] * P[2, 1] * P[2, 4] * 2.0
                - P[1, 6] * P[2, 2] * P[2, 5] * 2.0 - P[2, 3] * P[2, 4] * P[2, 5] * 2.0)
        a313 = (P[0, 6] * P[1, 6] - P[0, 6] * t19 - P[1, 6] * t20 - P[2, 6] * P[2, 6]
                + P[0, 4] * P[0, 6] * P[1, 4] + P[0, 5] * P[1, 5] * P[1, 6] + P[0, 4] * P[1, 5] * P[2, 6]
                + P[0, 5] * P[1, 4] * P[2, 6] - P[2, 4] * P[2, 5] * P[2, 6] * 2.0)

        # det(M(x)): degree-8 polynomial, ascending coefficients
        c[8] = a14 * a27 * a31 - a17 * a24 * a31 - a11 * a27 * a35 + a17 * a21 * a35 + a11 * a24 * a39 - a14 * a21 * a39
        c[7] = (a14 * a27 * a32 + a14 * a28 * a31 + a15 * a27 * a31 - a17 * a24 * a32 - a17 * a25 * a31
                - a18 * a24 * a31 - a11 * a27 * a36 - a11 * a28 * a35 - a12 * a27 * a35 + a17 * a21 * a36
                + a17 * a22 * a35 + a18 * a21 * a35 + a11 * a25 * a39 + a12 * a24 * a39 - a14 * a22 * a39
                - a15 * a21 * a39 + a11 * a24 * a310 - a14 * a21 * a310)
        c[6] = (a14 * a27 * a33 + a14 * a28 * a32 + a14 * a29 * a31 + a15 * a27 * a32 + a15 * a28 * a31
                + a16 * a27 * a31 - a17 * a24 * a33 - a17 * a25 * a32 - a17 * a26 * a31 - a18 * a24 * a32
                - a18 * a25 * a31 - a19 * a24 * a31 - a11 * a27 * a37 - a11 * a28 * a36 - a11 * a29 * a35
                - a12 * a27 * a36 - a12 * a28 * a35 - a13 * a27 * a35 + a17 * a21 * a37 + a17 * a22 * a36
                + a17 * a23 * a35 + a18 * a21 * a36 + a18 * a22 * a35 + a19 * a21 * a35 + a11 * a26 * a39
                + a12 * a25 * a39 + a13 * a24 * a39 - a14 * a23 * a39 - a15 * a22 * a39 - a16 * a21 * a39
                + a11 * a24 * a311 + a11 * a25 * a310 + a12 * a24 * a310 - a14 * a21 * a311 - a14 * a22 * a310
                - a15 * a21 * a310)
        c[5] = (a14 * a27 * a34 + a14 * a28 * a33 + a14 * a29 * a32 + a15 * a27 * a33 + a15 * a28 * a32
                + a15 * a29 * a31 + a16 * a27 * a32 + a16 * a28 * a31 - a17 * a24 * a34 - a17 * a25 * a33
                - a17 * a26 * a32 - a18 * a24 * a33 - a18 * a25 * a32 - a18 * a26 * a31 - a19 * a24 * a32
                - a19 * a25 * a31 - a11 * a27 * a38 - a11 * a28 * a37 - a11 * a29 * a36 - a12 * a27 * a37
                - a12 * a28 * a36 - a12 * a29 * a35 - a13 * a27 * a36 - a13 * a28 * a35 + a17 * a21 * a38
                + a17 * a22 * a37 + a17 * a23 * a36 + a18 * a21 * a37 + a18 * a22 * a36 + a18 * a23 * a35
                + a19 * a21 * a36 + a19 * a22 * a35 + a12 * a26 * a39 + a13 * a25 * a39 - a15 * a23 * a39
                - a16 * a22 * a39 - a24 * a31 * a110 + a21 * a35 * a110 + a14 * a31 * a210 - a11 * a35 * a210
                + a11 * a24 * a312 + a11 * a25 * a311 + a11 * a26 * a310 + a12 * a24 * a311 + a12 * a25 * a310
                + a13 * a24 * a310 - a14 * a21 * a312 - a14 * a22 * a311 - a14 * a23 * a310 - a15 * a21 * a311
                - a15 * a22 * a310 - a16 * a21 * a310)
        c[4] = (a14 * a28 * a34 + a14 * a29 * a33 + a15 * a27 * a34 + a15 * a28 * a33 + a15 * a29 * a32
                + a16 * a27 * a33 + a16 * a28 * a32 + a16 * a29 * a31 - a17 * a25 * a34 - a17 * a26 * a33
                - a18 * a24 * a34 - a18 * a25 * a33 - a18 * a26 * a32 - a19 * a24 * a33 - a19 * a25 * a32
                - a19 * a26 * a31 - a11 * a28 * a38 - a11 * a29 * a37 - a12 * a27 * a38 - a12 * a28 * a37
                - a12 * a29 * a36 - a13 * a27 * a37 - a13 * a28 * a36 - a13 * a29 * a35 + a17 * a22 * a38
                + a17 * a23 * a37 + a18 * a21 * a38 + a18 * a22 * a37 + a18 * a23 * a36 + a19 * a21 * a37
                + a19 * a22 * a36 + a19 * a23 * a35 + a13 * a26 * a39 - a16 * a23 * a39 - a24 * a32 * a110
                - a25 * a31 * a110 + a21 * a36 * a110 + a22 * a35 * a110 + a14 * a32 * a210 + a15 * a31 * a210
                - a11 * a36 * a210 - a12 * a35 * a210 + a11 * a24 * a313 + a11 * a25 * a312 + a11 * a26 * a311
                + a12 * a24 * a312 + a12 * a25 * a311 + a12 * a26 * a310 + a13 * a24 * a311 + a13 * a25 * a310
                - a14 * a21 * a313 - a14 * a22 * a312 - a14 * a23 * a311 - a15 * a21 * a312 - a15 * a22 * a311
                - a15 * a23 * a310 - a16 * a21 * a311 - a16 * a22 * a310)
        c[3] = (a14 * a29 * a34 + a15 * a28 * a34 + a15 * a29 * a33 + a16 * a27 * a34 + a16 * a28 * a33
                + a16 * a29 * a32 - a17 * a26 * a34 - a18 * a25 * a34 - a18 * a26 * a33 - a19 * a24 * a34
                - a19 * a25 * a33 - a19 * a26 * a32 - a11 * a29 * a38 - a12 * a28 * a38 - a12 * a29 * a37
                - a13 * a27 * a38 - a13 * a28 * a37 - a13 * a29 * a36 + a17 * a23 * a38 + a18 * a22 * a38
                + a18 * a23 * a37 + a19 * a21 * a38 + a19 * a22 * a37 + a19 * a23 * a36 - a24 * a33 * a110
                - a25 * a32 * a110 - a26 * a31 * a110 + a21 * a37 * a110 + a22 * a36 * a110 + a23 * a35 * a110
                + a14 * a33 * a210 + a15 * a32 * a210 + a16 * a31 * a210 - a11 * a37 * a210 - a12 * a36 * a210
                - a13 * a35 * a210 + a11 * a25 * a313 + a11 * a26 * a312 + a12 * a24 * a313 + a12 * a25 * a312
                + a12 * a26 * a311 + a13 * a24 * a312 + a13 * a25 * a311 + a13 * a26 * a310 - a14 * a22 * a313
                - a14 * a23 * a312 - a15 * a21 * a313 - a15 * a22 * a312 - a15 * a23 * a311 - a16 * a21 * a312
                - a16 * a22 * a311 - a16 * a23 * a310)
        c[2] = (a15 * a29 * a34 + a16 * a28 * a34 + a16 * a29 * a33 - a18 * a26 * a34 - a19 * a25 * a34
                - a19 * a26 * a33 - a12 * a29 * a38 - a13 * a28 * a38 - a13 * a29 * a37 + a18 * a23 * a38
                + a19 * a22 * a38 + a19 * a23 * a37 - a24 * a34 * a110 - a25 * a33 * a110 - a26 * a32 * a110
                + a21 * a38 * a110 + a22 * a37 * a110 + a23 * a36 * a110 + a14 * a34 * a210 + a15 * a33 * a210
                + a16 * a32 * a210 - a11 * a38 * a210 - a12 * a37 * a210 - a13 * a36 * a210 + a11 * a26 * a313
                + a12 * a25 * a313 + a12 * a26 * a312 + a13 * a24 * a313 + a13 * a25 * a312 + a13 * a26 * a311
                - a14 * a23 * a313 - a15 * a22 * a313 - a15 * a23 * a312 - a16 * a21 * a313 - a16 * a22 * a312
                - a16 * a23 * a311)
        c[1] = (a16 * a29 * a34 - a19 * a26 * a34 - a13 * a29 * a38 + a19 * a23 * a38 - a25 * a34 * a110
                - a26 * a33 * a110 + a22 * a38 * a110 + a23 * a37 * a110 + a15 * a34 * a210 + a16 * a33 * a210
                - a12 * a38 * a210 - a13 * a37 * a210 + a12 * a26 * a313 + a13 * a25 * a313 + a13 * a26 * a312
                - a15 * a23 * a313 - a16 * a22 * a313 - a16 * a23 * a312)
        c[0] = (-a26 * a34 * a110 + a23 * a38 * a110 + a16 * a34 * a210 - a13 * a38 * a210 + a13 * a26 * a313
                - a16 * a23 * a313)

        n_roots = real_roots_sturm(c, 8, chain, degs, roots,
                                   lo_stack, hi_stack, slo_stack, shi_stack)

        for i in range(n_roots):
            xs1 = roots[i]
            xs2 = xs1 * xs1
            xs3 = xs1 * xs2

            m00 = a11 * xs2 + a12 * xs1 + a13
            m01 = a14 * xs2 + a15 * xs1 + a16
            m02 = a17 * xs3 + a18 * xs2 + a19 * xs1 + a110
            m10 = a21 * xs2 + a22 * xs1 + a23
            m11 = a24 * xs2 + a25 * xs1 + a26
            m12 = a27 * xs3 + a28 * xs2 + a29 * xs1 + a210

            solutions[0, i] = xs1
            solutions[1, i] = (m12 * m01 - m02 * m11) / (m00 * m11 - m10 * m01)
            solutions[2, i] = (m12 * m00 - m02 * m10) / (m01 * m10 - m11 * m00)

        if elim_var == 1:
            for i in range(n_roots):
                tmp = solutions[0, i]
                solutions[0, i] = solutions[1, i]
                solutions[1, i] = tmp
        elif elim_var == 2:
            for i in range(n_roots):
                tmp = solutions[0, i]
                solutions[0, i] = solutions[2, i]
                solutions[2, i] = tmp

        _refine_3q3(coeffs, solutions, n_roots)
        return n_roots


    @jit(fastmath=True)
    def _solve_4x4(A, B):
        # B <- A^{-1} B in place (Gaussian elimination with partial pivoting);
        # False only for an exactly singular pivot
        for col in range(4):
            piv = col
            max_val = abs(A[col, col])
            for r in range(col + 1, 4):
                v = abs(A[r, col])
                if v > max_val:
                    max_val = v
                    piv = r
            if max_val == 0.0:
                return False
            if piv != col:
                for c in range(4):
                    t = A[col, c]
                    A[col, c] = A[piv, c]
                    A[piv, c] = t
                    t = B[col, c]
                    B[col, c] = B[piv, c]
                    B[piv, c] = t
            inv = 1.0 / A[col, col]
            for r in range(col + 1, 4):
                factor = A[r, col] * inv
                if factor != 0.0:
                    for c in range(col + 1, 4):
                        A[r, c] -= factor * A[col, c]
                    for c in range(4):
                        B[r, c] -= factor * B[col, c]
        for r in range(3, -1, -1):
            inv = 1.0 / A[r, r]
            for c in range(4):
                s = B[r, c]
                for k in range(r + 1, 4):
                    s -= A[r, k] * B[k, c]
                B[r, c] = s * inv
        return True


    @jit(fastmath=True)
    def _nullspace_p4pf(C, d):
        # 4-dimensional nullspace of the 4x8 cross-product constraint matrix via
        # Gaussian elimination (pivots on columns 0..3, free variables 4..7) and
        # modified Gram-Schmidt, like the 5-point nullspace; basis vector j is
        # written into d[8*j : 8*j + 8]
        for col in range(4):
            piv = col
            max_val = abs(C[col, col])
            for r in range(col + 1, 4):
                v = abs(C[r, col])
                if v > max_val:
                    max_val = v
                    piv = r
            if max_val < 1e-12:
                return False
            if piv != col:
                for c in range(col, 8):
                    t = C[col, c]
                    C[col, c] = C[piv, c]
                    C[piv, c] = t
            inv = 1.0 / C[col, col]
            for r in range(col + 1, 4):
                factor = C[r, col] * inv
                if factor != 0.0:
                    C[r, col] = 0.0
                    for c in range(col + 1, 8):
                        C[r, c] -= factor * C[col, c]

        for b in range(4):
            base = 8 * b
            free_col = 4 + b
            for m in range(8):
                d[base + m] = 0.0
            d[base + free_col] = 1.0
            for r in range(3, -1, -1):
                s = C[r, free_col]
                for c in range(r + 1, 4):
                    s += C[r, c] * d[base + c]
                d[base + r] = -s / C[r, r]

        for b in range(4):
            base = 8 * b
            for k in range(b):
                kb = 8 * k
                dot = 0.0
                for m in range(8):
                    dot += d[base + m] * d[kb + m]
                for m in range(8):
                    d[base + m] -= dot * d[kb + m]
            norm = 0.0
            for m in range(8):
                norm += d[base + m] * d[base + m]
            if norm < 1e-24:
                return False
            inv = 1.0 / math.sqrt(norm)
            for m in range(8):
                d[base + m] *= inv
        return True


    @jit(fastmath=True)
    def _project_rotation(m):
        # snap the flat row-major 3x3 in m[0:9] to the nearest rotation via the
        # quaternion round-trip (Shepperd's method) - the same renormalization
        # poselib applies by storing poses as quaternions. Row rescaling leaves
        # R only approximately orthogonal under noise.
        t0 = 1.0 + m[0] + m[4] + m[8]
        t1 = 1.0 + m[0] - m[4] - m[8]
        t2 = 1.0 - m[0] + m[4] - m[8]
        t3 = 1.0 - m[0] - m[4] + m[8]
        if t0 >= t1 and t0 >= t2 and t0 >= t3:
            r = math.sqrt(t0)
            inv = 0.5 / r
            qw = 0.5 * r
            qx = (m[7] - m[5]) * inv
            qy = (m[2] - m[6]) * inv
            qz = (m[3] - m[1]) * inv
        elif t1 >= t2 and t1 >= t3:
            r = math.sqrt(t1)
            inv = 0.5 / r
            qw = (m[7] - m[5]) * inv
            qx = 0.5 * r
            qy = (m[1] + m[3]) * inv
            qz = (m[2] + m[6]) * inv
        elif t2 >= t3:
            r = math.sqrt(t2)
            inv = 0.5 / r
            qw = (m[2] - m[6]) * inv
            qx = (m[1] + m[3]) * inv
            qy = 0.5 * r
            qz = (m[5] + m[7]) * inv
        else:
            r = math.sqrt(t3)
            inv = 0.5 / r
            qw = (m[3] - m[1]) * inv
            qx = (m[2] + m[6]) * inv
            qy = (m[5] + m[7]) * inv
            qz = 0.5 * r
        inv = 1.0 / math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qw *= inv
        qx *= inv
        qy *= inv
        qz *= inv
        m[0] = 1.0 - 2.0 * (qy * qy + qz * qz)
        m[1] = 2.0 * (qx * qy - qz * qw)
        m[2] = 2.0 * (qx * qz + qy * qw)
        m[3] = 2.0 * (qx * qy + qz * qw)
        m[4] = 1.0 - 2.0 * (qx * qx + qz * qz)
        m[5] = 2.0 * (qy * qz - qx * qw)
        m[6] = 2.0 * (qx * qz - qy * qw)
        m[7] = 2.0 * (qy * qz + qx * qw)
        m[8] = 1.0 - 2.0 * (qx * qx + qy * qy)


    @jit(fastmath=True)
    def _solve_p4pf_core(data, sample, models, d, C, A, B, coeffs, solutions,
                         Pm, xs, Xs, q_A, q_P, q_c, chain, roots, lo_stack,
                         hi_stack, degs, slo_stack, shi_stack):
        # scratch is passed in pre-shaped: see the note above the factory
        x_x, x_y, X_x, X_y, X_z = data

        f0 = 0.0
        for k in range(4):
            i = sample[k]
            xs[0, k] = x_x[i]
            xs[1, k] = x_y[i]
            Xs[k, 0] = X_x[i]
            Xs[k, 1] = X_y[i]
            Xs[k, 2] = X_z[i]
            f0 += math.sqrt(xs[0, k] * xs[0, k] + xs[1, k] * xs[1, k])
        f0 *= 0.25
        if f0 <= 0.0:
            return 0
        inv_f0 = 1.0 / f0
        for k in range(4):
            xs[0, k] *= inv_f0
            xs[1, k] *= inv_f0

        # nullspace of the eight cross-product constraints on the first two
        # camera rows (interleaved as [P00 P10 P01 P11 P02 P12 P03 P13]);
        # d[0:32] = N (8x4, column-major), an orthonormal basis of the nullspace
        for k in range(4):
            x = xs[0, k]
            y = xs[1, k]
            C[k, 0] = -y * Xs[k, 0]
            C[k, 1] = x * Xs[k, 0]
            C[k, 2] = -y * Xs[k, 1]
            C[k, 3] = x * Xs[k, 1]
            C[k, 4] = -y * Xs[k, 2]
            C[k, 5] = x * Xs[k, 2]
            C[k, 6] = -y
            C[k, 7] = x
        if not _nullspace_p4pf(C, d):
            return 0

        # third camera row: [p31 p32 p33 p34] = B @ [alpha; 1]
        for k in range(4):
            if abs(xs[0, k]) < abs(xs[1, k]):
                A[k, 0] = xs[1, k] * Xs[k, 0]
                A[k, 1] = xs[1, k] * Xs[k, 1]
                A[k, 2] = xs[1, k] * Xs[k, 2]
                A[k, 3] = xs[1, k]
                for j in range(4):
                    B[k, j] = (Xs[k, 0] * d[8 * j + 1] + Xs[k, 1] * d[8 * j + 3]
                               + Xs[k, 2] * d[8 * j + 5] + d[8 * j + 7])
            else:
                A[k, 0] = xs[0, k] * Xs[k, 0]
                A[k, 1] = xs[0, k] * Xs[k, 1]
                A[k, 2] = xs[0, k] * Xs[k, 2]
                A[k, 3] = xs[0, k]
                for j in range(4):
                    B[k, j] = (Xs[k, 0] * d[8 * j] + Xs[k, 1] * d[8 * j + 2]
                               + Xs[k, 2] * d[8 * j + 4] + d[8 * j + 6])
        if not _solve_4x4(A, B):
            return 0
        # d[32:48] = B (4x4, column-major)
        for j in range(4):
            for i in range(4):
                d[32 + 4 * j + i] = B[i, j]

        # orthogonality constraints between the three rotation rows
        coeffs[0, 0] = d[0] * d[1] + d[2] * d[3] + d[4] * d[5]
        coeffs[0, 1] = d[0] * d[9] + d[1] * d[8] + d[2] * d[11] + d[3] * d[10] + d[4] * d[13] + d[5] * d[12]
        coeffs[0, 2] = d[0] * d[17] + d[1] * d[16] + d[2] * d[19] + d[3] * d[18] + d[4] * d[21] + d[5] * d[20]
        coeffs[0, 3] = d[8] * d[9] + d[10] * d[11] + d[12] * d[13]
        coeffs[0, 4] = d[8] * d[17] + d[9] * d[16] + d[10] * d[19] + d[11] * d[18] + d[12] * d[21] + d[13] * d[20]
        coeffs[0, 5] = d[16] * d[17] + d[18] * d[19] + d[20] * d[21]
        coeffs[0, 6] = d[0] * d[25] + d[1] * d[24] + d[2] * d[27] + d[3] * d[26] + d[4] * d[29] + d[5] * d[28]
        coeffs[0, 7] = d[8] * d[25] + d[9] * d[24] + d[10] * d[27] + d[11] * d[26] + d[12] * d[29] + d[13] * d[28]
        coeffs[0, 8] = d[16] * d[25] + d[17] * d[24] + d[18] * d[27] + d[19] * d[26] + d[20] * d[29] + d[21] * d[28]
        coeffs[0, 9] = d[24] * d[25] + d[26] * d[27] + d[28] * d[29]
        coeffs[1, 0] = d[0] * d[32] + d[2] * d[33] + d[4] * d[34]
        coeffs[1, 1] = d[0] * d[36] + d[2] * d[37] + d[8] * d[32] + d[4] * d[38] + d[10] * d[33] + d[12] * d[34]
        coeffs[1, 2] = d[0] * d[40] + d[2] * d[41] + d[4] * d[42] + d[16] * d[32] + d[18] * d[33] + d[20] * d[34]
        coeffs[1, 3] = d[8] * d[36] + d[10] * d[37] + d[12] * d[38]
        coeffs[1, 4] = d[8] * d[40] + d[10] * d[41] + d[16] * d[36] + d[12] * d[42] + d[18] * d[37] + d[20] * d[38]
        coeffs[1, 5] = d[16] * d[40] + d[18] * d[41] + d[20] * d[42]
        coeffs[1, 6] = d[0] * d[44] + d[2] * d[45] + d[4] * d[46] + d[24] * d[32] + d[26] * d[33] + d[28] * d[34]
        coeffs[1, 7] = d[8] * d[44] + d[10] * d[45] + d[12] * d[46] + d[24] * d[36] + d[26] * d[37] + d[28] * d[38]
        coeffs[1, 8] = d[16] * d[44] + d[18] * d[45] + d[24] * d[40] + d[20] * d[46] + d[26] * d[41] + d[28] * d[42]
        coeffs[1, 9] = d[24] * d[44] + d[26] * d[45] + d[28] * d[46]
        coeffs[2, 0] = d[1] * d[32] + d[3] * d[33] + d[5] * d[34]
        coeffs[2, 1] = d[1] * d[36] + d[3] * d[37] + d[9] * d[32] + d[5] * d[38] + d[11] * d[33] + d[13] * d[34]
        coeffs[2, 2] = d[1] * d[40] + d[3] * d[41] + d[5] * d[42] + d[17] * d[32] + d[19] * d[33] + d[21] * d[34]
        coeffs[2, 3] = d[9] * d[36] + d[11] * d[37] + d[13] * d[38]
        coeffs[2, 4] = d[9] * d[40] + d[11] * d[41] + d[17] * d[36] + d[13] * d[42] + d[19] * d[37] + d[21] * d[38]
        coeffs[2, 5] = d[17] * d[40] + d[19] * d[41] + d[21] * d[42]
        coeffs[2, 6] = d[1] * d[44] + d[3] * d[45] + d[5] * d[46] + d[25] * d[32] + d[27] * d[33] + d[29] * d[34]
        coeffs[2, 7] = d[9] * d[44] + d[11] * d[45] + d[13] * d[46] + d[25] * d[36] + d[27] * d[37] + d[29] * d[38]
        coeffs[2, 8] = d[17] * d[44] + d[19] * d[45] + d[25] * d[40] + d[21] * d[46] + d[27] * d[41] + d[29] * d[42]
        coeffs[2, 9] = d[25] * d[44] + d[27] * d[45] + d[29] * d[46]

        n_sols = _re3q3_core(coeffs, solutions, q_A, q_P, q_c, chain, roots,
                             lo_stack, hi_stack, degs, slo_stack, shi_stack)

        best_err = 1.0
        found = False
        for i in range(n_sols):
            a0 = solutions[0, i]
            a1 = solutions[1, i]
            a2 = solutions[2, i]
            # P12 = N @ [alpha; 1], interleaving rows 0 and 1 of P
            for k in range(4):
                Pm[0, k] = (d[2 * k] * a0 + d[8 + 2 * k] * a1
                            + d[16 + 2 * k] * a2 + d[24 + 2 * k])
                Pm[1, k] = (d[2 * k + 1] * a0 + d[8 + 2 * k + 1] * a1
                            + d[16 + 2 * k + 1] * a2 + d[24 + 2 * k + 1])
                Pm[2, k] = (d[32 + k] * a0 + d[36 + k] * a1
                            + d[40 + k] * a2 + d[44 + k])

            det = (Pm[0, 0] * (Pm[1, 1] * Pm[2, 2] - Pm[1, 2] * Pm[2, 1])
                   - Pm[0, 1] * (Pm[1, 0] * Pm[2, 2] - Pm[1, 2] * Pm[2, 0])
                   + Pm[0, 2] * (Pm[1, 0] * Pm[2, 1] - Pm[1, 1] * Pm[2, 0]))
            if det < 0.0:
                for r in range(3):
                    for k in range(4):
                        Pm[r, k] = -Pm[r, k]

            n2 = math.sqrt(Pm[2, 0] ** 2 + Pm[2, 1] ** 2 + Pm[2, 2] ** 2)
            if n2 <= 0.0:
                continue
            inv = 1.0 / n2
            for r in range(3):
                for k in range(4):
                    Pm[r, k] *= inv
            fx = math.sqrt(Pm[0, 0] ** 2 + Pm[0, 1] ** 2 + Pm[0, 2] ** 2)
            fy = math.sqrt(Pm[1, 0] ** 2 + Pm[1, 1] ** 2 + Pm[1, 2] ** 2)
            if fx <= 0.0 or fy <= 0.0:
                continue
            inv = 1.0 / fx
            for k in range(4):
                Pm[0, k] *= inv
            inv = 1.0 / fy
            for k in range(4):
                Pm[1, k] *= inv

            # cheirality on the sample
            ok = True
            for k in range(4):
                if (Pm[2, 0] * Xs[k, 0] + Pm[2, 1] * Xs[k, 1]
                        + Pm[2, 2] * Xs[k, 2] + Pm[2, 3]) < 0.0:
                    ok = False
                    break
            if not ok:
                continue

            # keep the solution closest to square pixels (fx ~ fy)
            a = fx / fy
            # spelled out rather than `max(...)`: numba 0.67 cannot compile
            # the builtin for CUDA at all (see Instructions.md on the
            # `cuda-backend` branch)
            err = abs(a - 1.0)
            err_inv = abs(1.0 / a - 1.0)
            if err_inv > err:
                err = err_inv
            if err < best_err:
                best_err = err
                found = True
                for r in range(3):
                    for k in range(3):
                        models[0, 3 * r + k] = Pm[r, k]
                    models[0, 9 + r] = Pm[r, 3]
                models[0, 12] = 0.5 * (fx + fy) * f0

        if not found:
            return 0

        _project_rotation(models[0])
        return 1



    return {
        'det3_cols': _det3_cols,
        'solve_3xn_neg': _solve_3xn_neg,
        'refine_3q3': _refine_3q3,
        're3q3_core': _re3q3_core,
        'solve_4x4': _solve_4x4,
        'nullspace_p4pf': _nullspace_p4pf,
        'project_rotation': _project_rotation,
        'solve_p4pf_core': _solve_p4pf_core,
    }


_CPU = build_p4pf_kernels(cpu_jit, _real_roots_sturm)

# module-level names the tests import; these are the CPU kernels, byte-for-byte
# the same source as before the factory
_det3_cols = _CPU['det3_cols']
_solve_3xn_neg = _CPU['solve_3xn_neg']
_refine_3q3 = _CPU['refine_3q3']
_re3q3_core = _CPU['re3q3_core']
_solve_4x4 = _CPU['solve_4x4']
_nullspace_p4pf = _CPU['nullspace_p4pf']
_project_rotation = _CPU['project_rotation']
_solve_p4pf_core = _CPU['solve_p4pf_core']

# scratch layout of the flat workspace, shared by the CPU wrapper below and by
# the CUDA kernel, which allocates the same pieces as per-thread local arrays
WORKSPACE_FLOATS = 454
WORKSPACE_INTS = 137


@njit(cache=True, fastmath=True)
def _solve_p4pf(data, sample, models, workspace):
    # RANSAC-driver entry point: carves the flat workspace into the pre-shaped
    # scratch `_solve_p4pf_core` wants. CPU-only - `.reshape` does not compile
    # for CUDA, where the kernel allocates the same pieces with
    # `cuda.local.array` instead.
    d = workspace[0:48]
    C = workspace[48:80].reshape(4, 8)
    A = workspace[80:96].reshape(4, 4)
    B = workspace[96:112].reshape(4, 4)
    coeffs = workspace[112:142].reshape(3, 10)
    solutions = workspace[142:166].reshape(3, 8)
    Pm = workspace[166:178].reshape(3, 4)
    xs = workspace[178:186].reshape(2, 4)
    Xs = workspace[186:198].reshape(4, 3)
    q_A = workspace[198:207].reshape(3, 3)
    q_P = workspace[207:228].reshape(3, 7)
    q_c = workspace[228:237]
    chain = workspace[237:318].reshape(9, 9)
    roots = workspace[318:326]
    lo_stack = workspace[326:390]
    hi_stack = workspace[390:454]
    iw = np.empty(WORKSPACE_INTS, dtype=np.int64)
    degs = iw[0:9]
    slo_stack = iw[9:73]
    shi_stack = iw[73:137]
    return _solve_p4pf_core(data, sample, models, d, C, A, B, coeffs,
                            solutions, Pm, xs, Xs, q_A, q_P, q_c, chain,
                            roots, lo_stack, hi_stack, degs, slo_stack,
                            shi_stack)


class P4PFSolver():
    # minimal solver for absolute pose + focal length from 4 correspondences
    sample_size = 4
    num_params = MODEL_SIZE
    max_models = 1
    workspace_size = 454
    solve = staticmethod(_solve_p4pf)
