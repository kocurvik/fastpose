"""7-point relative pose solver with two unknown focal lengths.

The minimal step estimates fundamental matrices with the standard 7-point
solver, applies the Bougnoux formula for known principal points and square
pixels, then decomposes the resulting essential matrix into a pose model
`[R | t | f1 | f2]`.
"""

import math

import numpy as np
from numba import njit

from fastpose.solvers.essential import _pose_from_essential
from fastpose.solvers.fundamental import _det3_flat, _nullspace_7pt, _solve_cubic_real

MODEL_SIZE = 14


@njit(cache=True, inline='always')
def _skew3(x, S):
    S[0, 0] = 0.0
    S[0, 1] = -x[2]
    S[0, 2] = x[1]
    S[1, 0] = x[2]
    S[1, 1] = 0.0
    S[1, 2] = -x[0]
    S[2, 0] = -x[1]
    S[2, 1] = x[0]
    S[2, 2] = 0.0


@njit(cache=True, inline='always')
def _mat3_vec(A, x, y):
    for i in range(3):
        y[i] = A[i, 0] * x[0] + A[i, 1] * x[1] + A[i, 2] * x[2]


@njit(cache=True, inline='always')
def _mat3_mul_local(A, B, C):
    for i in range(3):
        for j in range(3):
            C[i, j] = A[i, 0] * B[0, j] + A[i, 1] * B[1, j] + A[i, 2] * B[2, j]


@njit(cache=True, fastmath=True)
def bougnoux_focals_sq(F, pp1x, pp1y, pp2x, pp2y, out):
    M = F.reshape(3, 3)
    U, s, Vt = np.linalg.svd(M)
    if abs(U[2, 2]) < 1e-12 or abs(Vt[2, 2]) < 1e-12:
        return False

    e1 = np.empty(3)
    e2 = np.empty(3)
    for i in range(3):
        e1[i] = Vt[2, i] / Vt[2, 2]
        e2[i] = U[i, 2] / U[2, 2]

    p1 = np.empty(3)
    p2 = np.empty(3)
    p1[0] = pp1x
    p1[1] = pp1y
    p1[2] = 1.0
    p2[0] = pp2x
    p2[1] = pp2y
    p2[2] = 1.0

    ex1 = np.empty((3, 3))
    ex2 = np.empty((3, 3))
    _skew3(e1, ex1)
    _skew3(e2, ex2)

    IF = np.empty((3, 3))
    IFT = np.empty((3, 3))
    for j in range(3):
        IF[0, j] = M[0, j]
        IF[1, j] = M[1, j]
        IF[2, j] = 0.0
        IFT[0, j] = M[j, 0]
        IFT[1, j] = M[j, 1]
        IFT[2, j] = 0.0

    tmp3 = np.empty((3, 3))
    A = np.empty((3, 3))
    _mat3_mul_local(ex2, IF, A)
    v = np.empty(3)
    _mat3_vec(M, p1, v)
    fp = p2[0] * v[0] + p2[1] * v[1] + v[2]
    _mat3_vec(A, p1, v)
    n1 = p2[0] * v[0] + p2[1] * v[1] + v[2]
    # p2^T [e2]_x I F I F^T p2
    FIT = np.empty((3, 3))
    for j in range(3):
        FIT[0, j] = M[j, 0]
        FIT[1, j] = M[j, 1]
        FIT[2, j] = 0.0
    _mat3_mul_local(A, FIT, tmp3)
    _mat3_vec(tmp3, p2, v)
    d1 = p2[0] * v[0] + p2[1] * v[1] + v[2]

    _mat3_mul_local(ex1, IFT, A)
    _mat3_vec(M.T, p2, v)
    fp_t = p1[0] * v[0] + p1[1] * v[1] + v[2]
    _mat3_vec(A, p2, v)
    n2 = p1[0] * v[0] + p1[1] * v[1] + v[2]
    FI = np.empty((3, 3))
    for j in range(3):
        FI[0, j] = M[0, j]
        FI[1, j] = M[1, j]
        FI[2, j] = 0.0
    _mat3_mul_local(A, FI, tmp3)
    _mat3_vec(tmp3, p1, v)
    d2 = p1[0] * v[0] + p1[1] * v[1] + v[2]

    if abs(d1) < 1e-18 or abs(d2) < 1e-18:
        return False
    out[0] = -n1 * fp / d1
    out[1] = -n2 * fp_t / d2
    return out[0] > 0.0 and out[1] > 0.0


@njit(cache=True, fastmath=True)
def _solve_varying_focal_7pt(data, sample, models, workspace):
    x1_x, x1_y, x2_x, x2_y, pp1x, pp1y, pp2x, pp2y = data
    A = workspace[0:63].reshape(7, 9)
    f_a = workspace[63:72]
    f_b = workspace[72:81]
    tmp = workspace[81:90]
    roots = workspace[90:93]
    focals_sq = workspace[93:95]
    e = workspace[95:104]
    Rbuf = workspace[104:122].reshape(2, 3, 3)

    for k in range(7):
        i = sample[k]
        x = x1_x[i]
        y = x1_y[i]
        xp = x2_x[i]
        yp = x2_y[i]
        A[k, 0] = xp * x
        A[k, 1] = xp * y
        A[k, 2] = xp
        A[k, 3] = yp * x
        A[k, 4] = yp * y
        A[k, 5] = yp
        A[k, 6] = x
        A[k, 7] = y
        A[k, 8] = 1.0

    if not _nullspace_7pt(A, f_a, f_b):
        return 0

    value_1 = _det3_flat(f_a)
    value_0 = _det3_flat(f_b)
    for j in range(9):
        tmp[j] = 2.0 * f_b[j] - f_a[j]
    value_neg_1 = _det3_flat(tmp)
    for j in range(9):
        tmp[j] = 2.0 * f_a[j] - f_b[j]
    value_2 = _det3_flat(tmp)

    linear_plus_cubic = 0.5 * (value_1 - value_neg_1)
    quadratic = 0.5 * (value_1 + value_neg_1 - 2.0 * value_0)
    cubic = (value_2 - value_0 - 4.0 * quadratic - 2.0 * linear_plus_cubic) / 6.0
    linear = linear_plus_cubic - cubic
    n_roots = _solve_cubic_real(cubic, quadratic, linear, value_0, roots)

    count = 0
    for r in range(n_roots):
        if count >= models.shape[0]:
            break
        alpha = roots[r]
        beta = 1.0 - alpha
        for j in range(9):
            tmp[j] = alpha * f_a[j] + beta * f_b[j]
        if not bougnoux_focals_sq(tmp, pp1x, pp1y, pp2x, pp2y, focals_sq):
            continue
        f1 = math.sqrt(focals_sq[0])
        f2 = math.sqrt(focals_sq[1])
        # E = K2^T F K1
        a00 = f2 * tmp[0]
        a01 = f2 * tmp[1]
        a02 = f2 * tmp[2]
        a10 = f2 * tmp[3]
        a11 = f2 * tmp[4]
        a12 = f2 * tmp[5]
        a20 = pp2x * tmp[0] + pp2y * tmp[3] + tmp[6]
        a21 = pp2x * tmp[1] + pp2y * tmp[4] + tmp[7]
        a22 = pp2x * tmp[2] + pp2y * tmp[5] + tmp[8]
        e[0] = f1 * a00
        e[1] = f1 * a01
        e[2] = pp1x * a00 + pp1y * a01 + a02
        e[3] = f1 * a10
        e[4] = f1 * a11
        e[5] = pp1x * a10 + pp1y * a11 + a12
        e[6] = f1 * a20
        e[7] = f1 * a21
        e[8] = pp1x * a20 + pp1y * a21 + a22
        num_poses = _pose_from_essential(e, (x1_x, x1_y, x2_x, x2_y), sample,
                                         pp1x, pp1y, pp2x, pp2y, f1, f2,
                                         models, count, Rbuf)
        for k in range(count, count + num_poses):
            models[k, 12] = f1
            models[k, 13] = f2
        count += num_poses
    return count


class SevenPointVaryingFocalSolver():
    sample_size = 7
    num_params = MODEL_SIZE
    # up to 3 real roots of the cubic, each of which can contribute more than
    # one cheirality-consistent pose (4 in the worst case, ~1 in practice)
    max_models = 12
    workspace_size = 122
    solve = staticmethod(_solve_varying_focal_7pt)
