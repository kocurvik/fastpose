"""Minimal 7-point solver for the fundamental matrix.

The numba kernel `_solve_7pt` plugs into the generic RANSAC engine through
the SevenPointSolver class; a pure-numpy reference implementation
(`seven_point`) is kept for the numpy baseline and cross-checking.

`data` layout for this problem: a tuple of four contiguous float64 columns
(x1_x, x1_y, x2_x, x2_y) so the scoring loop stays SIMD-vectorizable.
"""

import math

import numpy as np
from numba import njit

from fastpose.solvers.utils import fill_epipolar_matrix


# ---------------------------------------------------------------------------
# numpy reference implementation
# ---------------------------------------------------------------------------

def calculate_cubic_roots(F1, F2):
    # params:
    # F1 - (3, 3) array representing F_1
    # F2 - (3, 3) array representing F_2
    # returns all roots alpha of polynomial det(alpha * F_1 + (1 - alpha) * F_2) = 0
    matrices = np.stack((F2, F1, 2.0 * F2 - F1, 2.0 * F1 - F2))
    values = (
        matrices[:, 0, 0] * (matrices[:, 1, 1] * matrices[:, 2, 2] - matrices[:, 1, 2] * matrices[:, 2, 1])
        - matrices[:, 0, 1] * (matrices[:, 1, 0] * matrices[:, 2, 2] - matrices[:, 1, 2] * matrices[:, 2, 0])
        + matrices[:, 0, 2] * (matrices[:, 1, 0] * matrices[:, 2, 1] - matrices[:, 1, 1] * matrices[:, 2, 0])
    )

    value_0, value_1, value_neg_1, value_2 = values
    linear_plus_cubic = 0.5 * (value_1 - value_neg_1)
    quadratic = 0.5 * (value_1 + value_neg_1 - 2.0 * value_0)
    cubic = (value_2 - value_0 - 4.0 * quadratic - 2.0 * linear_plus_cubic) / 6.0
    linear = linear_plus_cubic - cubic

    return np.roots([cubic, quadratic, linear, value_0])


def seven_point(p, q):
    # params:
    # p - (n, 2) array containing points x
    # q - (n, 2) array containing points x'
    # returns:
    # Fs - list of (3,3) arrays representing fundamental matrices such that x'^T F x = 0
    x = p[:, 0]
    y = p[:, 1]

    xp = q[:, 0]
    yp = q[:, 1]
    A = np.column_stack([xp*x, xp*y, xp, yp*x, yp*y, yp, x, y, np.ones(len(p))])
    u, s, vt = np.linalg.svd(A)

    f_1 = vt[-2, :]
    f_2 = vt[-1, :]

    F_1 = np.reshape(f_1, (3, 3))
    F_2 = np.reshape(f_2, (3, 3))

    roots = calculate_cubic_roots(F_1, F_2)

    Fs = [np.real(alpha * F_1 + (1 - alpha) * F_2) for alpha in roots if np.isreal(alpha)]
    return Fs


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True, inline='always')
def _det3_flat(f):
    return (f[0] * (f[4] * f[8] - f[5] * f[7])
            - f[1] * (f[3] * f[8] - f[5] * f[6])
            + f[2] * (f[3] * f[7] - f[4] * f[6]))


@njit(cache=True)
def _solve_cubic_real(c3, c2, c1, c0, roots):
    # real roots of c3*x^3 + c2*x^2 + c1*x + c0, closed form (no companion
    # matrix eigendecomposition like np.roots)
    scale = abs(c2) + abs(c1) + abs(c0)
    if abs(c3) <= 1e-12 * scale or c3 == 0.0:
        if abs(c2) <= 1e-12 * (abs(c1) + abs(c0)) or c2 == 0.0:
            if c1 == 0.0:
                return 0
            roots[0] = -c0 / c1
            return 1
        disc = c1 * c1 - 4.0 * c2 * c0
        if disc < 0.0:
            return 0
        sq = math.sqrt(disc)
        roots[0] = (-c1 + sq) / (2.0 * c2)
        roots[1] = (-c1 - sq) / (2.0 * c2)
        n_roots = 2
    else:
        a = c2 / c3
        b = c1 / c3
        c = c0 / c3
        offset = a / 3.0
        p = b - a * a / 3.0
        q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c
        disc = 0.25 * q * q + p * p * p / 27.0
        if disc > 0.0:
            sq = math.sqrt(disc)
            u = -0.5 * q + sq
            v = -0.5 * q - sq
            u = math.copysign(abs(u) ** (1.0 / 3.0), u)
            v = math.copysign(abs(v) ** (1.0 / 3.0), v)
            roots[0] = u + v - offset
            n_roots = 1
        elif p == 0.0:
            roots[0] = -offset
            n_roots = 1
        else:
            r = 2.0 * math.sqrt(-p / 3.0)
            arg = 3.0 * q / (p * r)
            if arg > 1.0:
                arg = 1.0
            elif arg < -1.0:
                arg = -1.0
            phi = math.acos(arg) / 3.0
            two_pi_3 = 2.0943951023931953
            roots[0] = r * math.cos(phi) - offset
            roots[1] = r * math.cos(phi - two_pi_3) - offset
            roots[2] = r * math.cos(phi + two_pi_3) - offset
            n_roots = 3

    # one Newton polish step per root for numerical accuracy
    for i in range(n_roots):
        x = roots[i]
        fx = ((c3 * x + c2) * x + c1) * x + c0
        dfx = (3.0 * c3 * x + 2.0 * c2) * x + c1
        if dfx != 0.0:
            roots[i] = x - fx / dfx
    return n_roots


@njit(cache=True)
def _nullspace_7pt(A, f1, f2):
    # two-dimensional nullspace of the 7x9 epipolar constraint matrix via
    # Gaussian elimination with partial pivoting and back-substitution;
    # much faster than a LAPACK SVD call for a matrix this small.
    # Returns False for degenerate (rank-deficient) samples.
    for col in range(7):
        piv = col
        max_val = abs(A[col, col])
        for r in range(col + 1, 7):
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
        for r in range(col + 1, 7):
            factor = A[r, col] * inv
            if factor != 0.0:
                A[r, col] = 0.0
                for c in range(col + 1, 9):
                    A[r, c] -= factor * A[col, c]

    # back-substitute with free variables (f7, f8) = (1, 0) and (0, 1)
    for r in range(6, -1, -1):
        s1 = A[r, 7]
        s2 = A[r, 8]
        for c in range(r + 1, 7):
            s1 += A[r, c] * f1[c]
            s2 += A[r, c] * f2[c]
        inv = 1.0 / A[r, r]
        f1[r] = -s1 * inv
        f2[r] = -s2 * inv
    f1[7] = 1.0
    f1[8] = 0.0
    f2[7] = 0.0
    f2[8] = 1.0

    # normalize for a well-conditioned determinant cubic
    norm1 = 0.0
    norm2 = 0.0
    for j in range(9):
        norm1 += f1[j] * f1[j]
        norm2 += f2[j] * f2[j]
    inv1 = 1.0 / math.sqrt(norm1)
    inv2 = 1.0 / math.sqrt(norm2)
    for j in range(9):
        f1[j] *= inv1
        f2[j] *= inv2
    return True


@njit(cache=True)
def _solve_7pt(data, sample, models, workspace):
    # minimal 7-point solver; writes up to 3 models (flattened F) into
    # `models` and returns their count
    A = workspace[0:63].reshape(7, 9)
    f1 = workspace[63:72]
    f2 = workspace[72:81]
    tmp = workspace[81:90]
    roots = workspace[90:93]

    fill_epipolar_matrix(data, sample, A)

    if not _nullspace_7pt(A, f1, f2):
        return 0

    # cubic det(alpha * F1 + (1 - alpha) * F2) = 0 via 4-point evaluation
    value_1 = _det3_flat(f1)
    value_0 = _det3_flat(f2)
    for j in range(9):
        tmp[j] = 2.0 * f2[j] - f1[j]
    value_neg_1 = _det3_flat(tmp)
    for j in range(9):
        tmp[j] = 2.0 * f1[j] - f2[j]
    value_2 = _det3_flat(tmp)

    linear_plus_cubic = 0.5 * (value_1 - value_neg_1)
    quadratic = 0.5 * (value_1 + value_neg_1 - 2.0 * value_0)
    cubic = (value_2 - value_0 - 4.0 * quadratic - 2.0 * linear_plus_cubic) / 6.0
    linear = linear_plus_cubic - cubic

    n_roots = _solve_cubic_real(cubic, quadratic, linear, value_0, roots)

    for r in range(n_roots):
        alpha = roots[r]
        beta = 1.0 - alpha
        for j in range(9):
            models[r, j] = alpha * f1[j] + beta * f2[j]

    return n_roots


# ---------------------------------------------------------------------------
# pluggable component class
# ---------------------------------------------------------------------------

class SevenPointSolver():
    # minimal solver for the fundamental matrix from 7 correspondences
    sample_size = 7
    num_params = 9
    max_models = 3
    workspace_size = 93
    solve = staticmethod(_solve_7pt)
