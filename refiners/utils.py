"""Shared machinery for the epipolar-geometry LM refiners.

Both the fundamental and the essential refiner work on the factorization
F = U diag(1, sigma, 0) Vt. The state vector layout is shared:

    state[0:9]   U   (row-major 3x3)
    state[9:18]  Vt  (row-major 3x3)
    state[18]    sigma (fixed at 1 on the essential manifold)

Tangent parameters are two rotation updates (U exp([w1]_x), exp(-[w2]_x) Vt)
plus, for the fundamental matrix only, the singular value ratio sigma.
"""

import math

import numpy as np
from numba import njit

STATE_SIZE = 19


@njit(cache=True, inline='always')
def mat3_mul(A, B, C):
    # C = A @ B for 3x3 matrices
    for i in range(3):
        for j in range(3):
            C[i, j] = A[i, 0] * B[0, j] + A[i, 1] * B[1, j] + A[i, 2] * B[2, j]


@njit(cache=True, inline='always')
def rodrigues(w0, w1, w2, R):
    # R = exp([w]_x) via the Rodrigues formula, Taylor expansion near zero
    theta_sq = w0 * w0 + w1 * w1 + w2 * w2
    theta = math.sqrt(theta_sq)
    if theta < 1e-9:
        a = 1.0 - theta_sq / 6.0
        b = 0.5 - theta_sq / 24.0
    else:
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta_sq
    R[0, 0] = 1.0 - b * (w1 * w1 + w2 * w2)
    R[0, 1] = -a * w2 + b * w0 * w1
    R[0, 2] = a * w1 + b * w0 * w2
    R[1, 0] = a * w2 + b * w0 * w1
    R[1, 1] = 1.0 - b * (w0 * w0 + w2 * w2)
    R[1, 2] = -a * w0 + b * w1 * w2
    R[2, 0] = -a * w1 + b * w0 * w2
    R[2, 1] = a * w0 + b * w1 * w2
    R[2, 2] = 1.0 - b * (w0 * w0 + w1 * w1)


@njit(cache=True, inline='always')
def factorized_f(U, Vt, sigma, f):
    # F = U @ diag(1, sigma, 0) @ Vt, written into the flat 9-vector f
    for i in range(3):
        for j in range(3):
            f[3 * i + j] = U[i, 0] * Vt[0, j] + sigma * U[i, 1] * Vt[1, j]


@njit(cache=True)
def svd_init_state(model, state):
    # decompose the flat model into the shared U/Vt/sigma state; sigma is
    # the singular value ratio s1/s0 (rank-2 projection of the input model)
    M = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            M[i, j] = model[3 * i + j]
    U, s, Vt = np.linalg.svd(M)
    if s[1] <= 0.0:
        return False
    for i in range(3):
        for j in range(3):
            state[3 * i + j] = U[i, j]
            state[9 + 3 * i + j] = Vt[i, j]
    state[18] = s[1] / s[0]
    return True


@njit(cache=True)
def state_to_model(state, f):
    U = state[0:9].reshape(3, 3)
    Vt = state[9:18].reshape(3, 3)
    factorized_f(U, Vt, state[18], f)


@njit(cache=True)
def apply_rotation_step(state, delta, state_new):
    # retraction of the two rotation tangent parameters:
    # U <- U exp([delta_0:3]_x), Vt <- exp(-[delta_3:6]_x) Vt; sigma copied
    U = state[0:9].reshape(3, 3)
    Vt = state[9:18].reshape(3, 3)
    U_new = state_new[0:9].reshape(3, 3)
    Vt_new = state_new[9:18].reshape(3, 3)
    R = np.empty((3, 3))
    rodrigues(delta[0], delta[1], delta[2], R)
    mat3_mul(U, R, U_new)
    rodrigues(-delta[3], -delta[4], -delta[5], R)
    mat3_mul(R, Vt, Vt_new)
    state_new[18] = state[18]


@njit(cache=True)
def epipolar_jacobian_basis(U, Vt, sigma, B):
    # tangent basis of the factorization F = U diag(1, sigma, 0) Vt at the
    # current state: B[k] = dF/dtheta_k as flat 9-vectors. Rows 0..2 are the
    # left rotation, rows 3..5 the right rotation; if B has a 7th row it is
    # the sigma derivative (fundamental matrix; on the essential manifold
    # sigma is fixed at 1 and B has 6 rows).
    S = np.empty((3, 3))
    SD = np.empty((3, 3))
    tmp = np.empty((3, 3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                S[i, j] = 0.0
        if k == 0:
            S[1, 2] = -1.0
            S[2, 1] = 1.0
        elif k == 1:
            S[0, 2] = 1.0
            S[2, 0] = -1.0
        else:
            S[0, 1] = -1.0
            S[1, 0] = 1.0

        # left rotation: dF/dw1_k = U @ (skew(e_k) @ D) @ Vt
        for i in range(3):
            SD[i, 0] = S[i, 0]
            SD[i, 1] = sigma * S[i, 1]
            SD[i, 2] = 0.0
        mat3_mul(SD, Vt, tmp)
        for i in range(3):
            for j in range(3):
                B[k, 3 * i + j] = (U[i, 0] * tmp[0, j] + U[i, 1] * tmp[1, j]
                                   + U[i, 2] * tmp[2, j])

        # right rotation: dF/dw2_k = -U @ (D @ skew(e_k)) @ Vt
        for j in range(3):
            SD[0, j] = S[0, j]
            SD[1, j] = sigma * S[1, j]
            SD[2, j] = 0.0
        mat3_mul(SD, Vt, tmp)
        for i in range(3):
            for j in range(3):
                B[3 + k, 3 * i + j] = -(U[i, 0] * tmp[0, j] + U[i, 1] * tmp[1, j]
                                        + U[i, 2] * tmp[2, j])

    if B.shape[0] > 6:
        # sigma: dF/dsigma = u_1 v_1^T
        for i in range(3):
            for j in range(3):
                B[6, 3 * i + j] = U[i, 1] * Vt[1, j]


@njit(cache=True, fastmath=True)
def accumulate_sampson_normal_eqs(data, f, B, JtJ, Jtr, max_error_sq):
    # normal equations of the truncated Sampson residuals s_i for the tangent
    # basis B: JtJ += J_i J_i^T, Jtr += J_i s_i with J_i[p] = ds_i/dF . B[p].
    # Points outside the threshold get zero weight (truncated loss), so the
    # minimized cost matches the MSAC score used for model selection.
    # Returns the number of contributing residuals.
    x1_x, x1_y, x2_x, x2_y = data
    n = x1_x.shape[0]
    num_tangent = B.shape[0]

    dsdF = np.empty(9)
    J = np.empty(num_tangent)
    for p in range(num_tangent):
        Jtr[p] = 0.0
        for q in range(num_tangent):
            JtJ[p, q] = 0.0

    num_residuals = 0
    for i in range(n):
        x = x1_x[i]
        y = x1_y[i]
        xp = x2_x[i]
        yp = x2_y[i]
        fx1_0 = f[0] * x + f[1] * y + f[2]
        fx1_1 = f[3] * x + f[4] * y + f[5]
        fx1_2 = f[6] * x + f[7] * y + f[8]
        ftx2_0 = f[0] * xp + f[3] * yp + f[6]
        ftx2_1 = f[1] * xp + f[4] * yp + f[7]
        residual = xp * fx1_0 + yp * fx1_1 + fx1_2
        denominator = (fx1_0 * fx1_0 + fx1_1 * fx1_1
                       + ftx2_0 * ftx2_0 + ftx2_1 * ftx2_1)
        r2 = residual * residual
        if r2 >= max_error_sq * denominator:
            continue  # truncated loss: zero weight outside threshold
        num_residuals += 1

        # s_i = residual / sqrt(denominator); ds_i/dF as flat 9-vector
        inv_sqrt_den = 1.0 / math.sqrt(denominator)
        c = residual * inv_sqrt_den / denominator
        s_i = residual * inv_sqrt_den

        dsdF[0] = inv_sqrt_den * xp * x - c * (fx1_0 * x + xp * ftx2_0)
        dsdF[1] = inv_sqrt_den * xp * y - c * (fx1_0 * y + xp * ftx2_1)
        dsdF[2] = inv_sqrt_den * xp - c * fx1_0
        dsdF[3] = inv_sqrt_den * yp * x - c * (fx1_1 * x + yp * ftx2_0)
        dsdF[4] = inv_sqrt_den * yp * y - c * (fx1_1 * y + yp * ftx2_1)
        dsdF[5] = inv_sqrt_den * yp - c * fx1_1
        dsdF[6] = inv_sqrt_den * x - c * ftx2_0
        dsdF[7] = inv_sqrt_den * y - c * ftx2_1
        dsdF[8] = inv_sqrt_den

        for p in range(num_tangent):
            acc = 0.0
            for j in range(9):
                acc += dsdF[j] * B[p, j]
            J[p] = acc
        for p in range(num_tangent):
            Jtr[p] += J[p] * s_i
            for q in range(p, num_tangent):
                JtJ[p, q] += J[p] * J[q]

    for p in range(num_tangent):
        for q in range(p):
            JtJ[p, q] = JtJ[q, p]
    return num_residuals
