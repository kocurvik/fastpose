"""Shared machinery for the epipolar-geometry LM refiners.

Both the fundamental and the essential refiner work on the factorization
F = U diag(1, sigma, 0) Vt. The state vector layout is shared:

    state[0:9]   U   (row-major 3x3)
    state[9:18]  Vt  (row-major 3x3)
    state[18]    sigma (fixed at 1 on the essential manifold)

Tangent parameters are two rotation updates (U exp([w1]_x), exp(-[w2]_x) Vt)
plus, for the fundamental matrix only, the singular value ratio sigma.

`build_pose_lo_refine` / `build_focal_lo_refine` at the bottom wrap an LM
kernel into the local-optimization refiner the relative-pose estimators hand
to the RANSAC driver: it restricts the refit to the relaxed-threshold inlier
subset the way poselib's `refine_model` does (see `LO_INLIER_SCALE`).
"""

import math

import numpy as np
from numba import njit

from fastpose.kernel_cache import stabilize
from fastpose.refiners.losses import TruncatedLoss
from fastpose.scorers.sampson import (MIN_DEPTH, cheirality_ok,
                                      essential_from_pose, model_to_fundamental)

STATE_SIZE = 19

# Poselib's relative-pose estimators do not bundle over the whole
# correspondence set during local optimization: `refine_model` first calls
# `get_inliers` at this multiple of the squared threshold and refines over
# that subset only (robust/estimators/relative_pose.cc). The truncated loss
# already zeroes the *weight* of everything past 1x, so the normal equations
# are unaffected - but the LM's accept/reject test is not. Every far outlier
# otherwise contributes a constant max_error_sq that the jacobian never
# models, and points crossing the threshold add noise that swamps the true
# cost decrease, causing spurious rejections that inflate lambda and burn the
# iteration budget. The estimators pass this as `relaxed_inlier_scale`.
LO_INLIER_SCALE = 5.0


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


@njit(cache=True)
def tangent_basis(t, b1, b2):
    # orthonormal basis (b1, b2) of the plane orthogonal to the unit vector
    # t; deterministic in t so the jacobian and the retraction always agree
    a0 = abs(t[0])
    a1 = abs(t[1])
    a2 = abs(t[2])
    # b1 = t x e_k for the axis e_k of the smallest |component|
    if a0 <= a1 and a0 <= a2:
        b1[0] = 0.0
        b1[1] = t[2]
        b1[2] = -t[1]
    elif a1 <= a2:
        b1[0] = -t[2]
        b1[1] = 0.0
        b1[2] = t[0]
    else:
        b1[0] = t[1]
        b1[1] = -t[0]
        b1[2] = 0.0
    inv = 1.0 / math.sqrt(b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2])
    b1[0] *= inv
    b1[1] *= inv
    b1[2] *= inv
    b2[0] = t[1] * b1[2] - t[2] * b1[1]
    b2[1] = t[2] * b1[0] - t[0] * b1[2]
    b2[2] = t[0] * b1[1] - t[1] * b1[0]


@njit(cache=True)
def essential_tangent_rows(pose, e, B):
    # rows 0..4 of B: dE/dtheta as flat 9-vectors for the 5 tangent
    # parameters of E = [t]_x R with t on the unit sphere, given the pose
    # [R (row-major 3x3) | t] and its flat E.
    #
    # rows 0..2 (rotation, retraction R exp([w]_x)): dE/dw_k = E skew(e_k),
    # plain column shuffles of E
    for i in range(3):
        e0 = e[3 * i]
        e1 = e[3 * i + 1]
        e2 = e[3 * i + 2]
        B[0, 3 * i] = 0.0
        B[0, 3 * i + 1] = e2
        B[0, 3 * i + 2] = -e1
        B[1, 3 * i] = -e2
        B[1, 3 * i + 1] = 0.0
        B[1, 3 * i + 2] = e0
        B[2, 3 * i] = e1
        B[2, 3 * i + 1] = -e0
        B[2, 3 * i + 2] = 0.0

    # rows 3..4 (translation): dE/dalpha_i = [b_i]_x R, since dt/dalpha_i is
    # b_i on the unit sphere (b_i is orthogonal to t, so the renormalization
    # in the retraction is second order)
    b1 = np.empty(3)
    b2 = np.empty(3)
    tangent_basis(pose[9:12], b1, b2)
    for j in range(3):
        r0 = pose[j]
        r1 = pose[3 + j]
        r2 = pose[6 + j]
        B[3, j] = -b1[2] * r1 + b1[1] * r2
        B[3, 3 + j] = b1[2] * r0 - b1[0] * r2
        B[3, 6 + j] = -b1[1] * r0 + b1[0] * r1
        B[4, j] = -b2[2] * r1 + b2[1] * r2
        B[4, 3 + j] = b2[2] * r0 - b2[0] * r2
        B[4, 6 + j] = -b2[1] * r0 + b2[0] * r1


@njit(cache=True)
def log_focal_tangent_rows(f, pp1x, pp1y, pp2x, pp2y, row1, row2):
    # dF/dlog(f1) and dF/dlog(f2) of F = K2^-T E K1^-1, as flat 9-vectors.
    #
    # f1 enters only through K1^-1, which multiplies F from the right and so
    # acts on its columns; f2 only through K2^-T, which multiplies from the
    # left and acts on its rows. In both cases the derivative of the 1/f
    # entries w.r.t. log f is just -(1/f), which folds the whole thing back
    # into F itself - no E, no focal, and nothing to differentiate
    # numerically:
    #     dF/dlog f1: columns 0, 1 -> -F, column 2 -> pp1 . (F col0, F col1)
    #     dF/dlog f2: rows    0, 1 -> -F, row    2 -> pp2 . (F row0, F row1)
    for i in range(3):
        c0 = f[3 * i]
        c1 = f[3 * i + 1]
        row1[3 * i] = -c0
        row1[3 * i + 1] = -c1
        row1[3 * i + 2] = pp1x * c0 + pp1y * c1
    for j in range(3):
        r0 = f[j]
        r1 = f[3 + j]
        row2[j] = -r0
        row2[3 + j] = -r1
        row2[6 + j] = pp2x * r0 + pp2y * r1


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


def build_sampson_accumulate(loss):
    # builds the normal equations of the Sampson residuals s_i for the
    # tangent basis B: JtJ += w_i J_i J_i^T, Jtr += w_i J_i s_i with
    # J_i[p] = ds_i/dF . B[p] and w_i = loss.weight(s_i^2, max_error_sq); a
    # zero weight drops the point entirely (for TruncatedLoss this is
    # exactly the old hard truncation, so the skip-heavy-work fast path is
    # preserved). `loss` is bound as a compile-time constant, the same
    # closure pattern build_lm_refine uses for its own kernels.
    weight_fn = loss.weight

    @njit(cache=True, fastmath=True)
    def _accumulate_sampson_normal_eqs(data, f, B, JtJ, Jtr, max_error_sq):
        # returns the number of contributing residuals
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
            if denominator <= 0.0:
                continue

            # s_i = residual / sqrt(denominator); ds_i/dF as flat 9-vector
            inv_sqrt_den = 1.0 / math.sqrt(denominator)
            s_i = residual * inv_sqrt_den
            w = weight_fn(s_i * s_i, max_error_sq)
            if w <= 0.0:
                continue
            num_residuals += 1

            c = s_i / denominator
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
                Jtr[p] += w * J[p] * s_i
                for q in range(p, num_tangent):
                    JtJ[p, q] += w * J[p] * J[q]

        for p in range(num_tangent):
            for q in range(p):
                JtJ[p, q] = JtJ[q, p]
        return num_residuals

    return _accumulate_sampson_normal_eqs


accumulate_sampson_normal_eqs = build_sampson_accumulate(TruncatedLoss())


# ---------------------------------------------------------------------------
# local-optimization wrappers: refine over the relaxed-threshold inlier subset
# only, the way poselib's *RelativePoseEstimator::refine_model does
# ---------------------------------------------------------------------------

@njit(cache=True)
def _relaxed_inlier_mask(x1_x, x1_y, x2_x, x2_y, f, threshold_sq, keep):
    # poselib's get_inliers for a flat row-major 3x3 epipolar matrix `f`:
    # marks the points whose squared Sampson error is below threshold_sq and
    # returns how many there are
    n = x1_x.shape[0]
    count = 0
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
        ok = denominator > 0.0 and residual * residual < threshold_sq * denominator
        keep[i] = ok
        if ok:
            count += 1
    return count


@njit(cache=True)
def _pose_relaxed_inlier_mask(x1_x, x1_y, x2_x, x2_y, pose, threshold_sq, keep):
    # same, for a pose model: E = [t]_x R plus the per-point cheirality check
    # that poselib's CameraPose overload of get_inliers applies
    e = np.empty(9)
    essential_from_pose(pose, e)
    count = _relaxed_inlier_mask(x1_x, x1_y, x2_x, x2_y, e, threshold_sq, keep)
    for i in range(x1_x.shape[0]):
        if keep[i] and not cheirality_ok(pose, x1_x[i], x1_y[i], x2_x[i],
                                         x2_y[i], MIN_DEPTH):
            keep[i] = False
            count -= 1
    return count


@njit(cache=True)
def _compact_columns(x1_x, x1_y, x2_x, x2_y, keep, count):
    sx1_x = np.empty(count)
    sx1_y = np.empty(count)
    sx2_x = np.empty(count)
    sx2_y = np.empty(count)
    j = 0
    for i in range(x1_x.shape[0]):
        if keep[i]:
            sx1_x[j] = x1_x[i]
            sx1_y[j] = x1_y[i]
            sx2_x[j] = x2_x[i]
            sx2_y[j] = x2_y[i]
            j += 1
    return sx1_x, sx1_y, sx2_x, sx2_y


def build_pose_lo_refine(refine_fn, min_inliers, relaxed_scale=LO_INLIER_SCALE):
    # wraps a pose-model LM kernel so it refines over the relaxed-threshold
    # inlier subset only. `min_inliers` mirrors the `num_inl <= N` early
    # return in poselib's refine_model; returning False leaves the caller
    # with the unrefined model, which is what upstream ends up scoring too.
    stabilize(refine_fn)

    @njit(cache=True)
    def refine(data, model, refined, max_error_sq, num_iterations):
        x1_x, x1_y, x2_x, x2_y = data
        keep = np.empty(x1_x.shape[0], dtype=np.bool_)
        count = _pose_relaxed_inlier_mask(x1_x, x1_y, x2_x, x2_y, model,
                                          relaxed_scale * max_error_sq, keep)
        if count <= min_inliers:
            return False
        subset = _compact_columns(x1_x, x1_y, x2_x, x2_y, keep, count)
        return refine_fn(subset, model, refined, max_error_sq, num_iterations)

    return refine


def build_focal_lo_refine(refine_fn, min_inliers, relaxed_scale=LO_INLIER_SCALE):
    # same, for the shared- and varying-focal models [R | t | f1 | f2], whose
    # data tuple carries the two principal points alongside the coordinate
    # columns. Poselib selects the subset with the matrix overload of
    # get_inliers here, so there is no cheirality check.
    stabilize(refine_fn)

    @njit(cache=True)
    def refine(data, model, refined, max_error_sq, num_iterations):
        x1_x, x1_y, x2_x, x2_y, pp1x, pp1y, pp2x, pp2y = data
        f = np.empty(9)
        if not model_to_fundamental(model, pp1x, pp1y, pp2x, pp2y, f):
            return False
        keep = np.empty(x1_x.shape[0], dtype=np.bool_)
        count = _relaxed_inlier_mask(x1_x, x1_y, x2_x, x2_y, f,
                                     relaxed_scale * max_error_sq, keep)
        if count <= min_inliers:
            return False
        sx1_x, sx1_y, sx2_x, sx2_y = _compact_columns(x1_x, x1_y, x2_x, x2_y,
                                                      keep, count)
        subset = (sx1_x, sx1_y, sx2_x, sx2_y, pp1x, pp1y, pp2x, pp2y)
        return refine_fn(subset, model, refined, max_error_sq, num_iterations)

    return refine
