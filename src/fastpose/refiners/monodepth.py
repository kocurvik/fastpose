"""Hybrid LM refiners for relative pose with monocular depth estimates.

Ports of poselib's MonoDepthRelPoseRefiner / MonoDepthSharedFocalRelPoseRefiner
/ MonoDepthVaryingFocalRelPoseRefiner (robust/optim/monodepth_relpose.h): the
minimized cost is a truncated hybrid of

- the Sampson error of E = [t]_x R (or of F = diag(1,1,f2) E diag(1,1,f1)
  for the focal problems, in centered pixel coordinates), weighted by
  `weight_sampson`, and
- the symmetric reprojection error through the monodepth-induced 3D points
  (forward: (d1 + shift1) x1h projected into image 2; backward:
  scale (d2 + shift2) x2h projected into image 1), scaled by
  sqrt(scale_reproj) so that both terms truncate at the same epipolar
  threshold. Points behind a camera contribute nothing (poselib behavior).

`scale_reproj` and `weight_sampson` ride along in the data tuple
(entries 6 and 7). All jacobians are analytic. Unlike the epipolar-only
refiners, the translation is not confined to the unit sphere - the depths
fix the scale - so t gets three additive tangent parameters, and scale /
shifts / focals are plain additive parameters like in poselib:

- calibrated:      [w(3), dt(3), dscale] (+ [dshift1, dshift2] with shift)
- shared focal:    [w(3), dt(3), dscale, df]
- varying focal:   [w(3), dt(3), dscale, df1, df2]
"""

import math

import numpy as np
from numba import njit

from fastpose.refiners.lm import build_lm_refine
from fastpose.refiners.losses import TruncatedLoss, get_loss
from fastpose.refiners.utils import mat3_mul, rodrigues
from fastpose.scorers.sampson import essential_from_pose

STATE_SIZE = 15
MODEL_SIZE = 15


@njit(cache=True)
def _init_state(model, state):
    for j in range(MODEL_SIZE):
        state[j] = model[j]
    return True


@njit(cache=True)
def _init_state_focal(model, state):
    if model[12] <= 0.0 or model[13] <= 0.0:
        return False
    for j in range(MODEL_SIZE):
        state[j] = model[j]
    return True


@njit(cache=True)
def _state_to_model(state, model):
    for j in range(MODEL_SIZE):
        model[j] = state[j]


@njit(cache=True)
def _apply_step_calibrated(state, delta, state_new):
    # [w(3), dt(3), dscale] plus [dshift1, dshift2] when delta has 9 entries
    R = state[0:9].reshape(3, 3)
    R_new = state_new[0:9].reshape(3, 3)
    Rod = np.empty((3, 3))
    rodrigues(delta[0], delta[1], delta[2], Rod)
    mat3_mul(R, Rod, R_new)
    state_new[9] = state[9] + delta[3]
    state_new[10] = state[10] + delta[4]
    state_new[11] = state[11] + delta[5]
    state_new[12] = state[12] + delta[6]
    if delta.shape[0] > 7:
        state_new[13] = state[13] + delta[7]
        state_new[14] = state[14] + delta[8]
    else:
        state_new[13] = state[13]
        state_new[14] = state[14]


@njit(cache=True)
def _apply_step_shared_focal(state, delta, state_new):
    # [w(3), dt(3), dscale, df]; model layout [R | t | f | f | scale]
    R = state[0:9].reshape(3, 3)
    R_new = state_new[0:9].reshape(3, 3)
    Rod = np.empty((3, 3))
    rodrigues(delta[0], delta[1], delta[2], Rod)
    mat3_mul(R, Rod, R_new)
    state_new[9] = state[9] + delta[3]
    state_new[10] = state[10] + delta[4]
    state_new[11] = state[11] + delta[5]
    f = state[12] + delta[7]
    state_new[12] = f
    state_new[13] = f
    state_new[14] = state[14] + delta[6]


@njit(cache=True)
def _apply_step_varying_focal(state, delta, state_new):
    # [w(3), dt(3), dscale, df1, df2]; model layout [R | t | f1 | f2 | scale]
    R = state[0:9].reshape(3, 3)
    R_new = state_new[0:9].reshape(3, 3)
    Rod = np.empty((3, 3))
    rodrigues(delta[0], delta[1], delta[2], Rod)
    mat3_mul(R, Rod, R_new)
    state_new[9] = state[9] + delta[3]
    state_new[10] = state[10] + delta[4]
    state_new[11] = state[11] + delta[5]
    state_new[12] = state[12] + delta[7]
    state_new[13] = state[13] + delta[8]
    state_new[14] = state[14] + delta[6]


@njit(cache=True, inline='always', fastmath=True)
def _focal_fundamental(e, f1, f2, f):
    # F = diag(1, 1, f2) E diag(1, 1, f1), proportional to K2^-T E K1^-1, so
    # the Sampson error matches the scorer's; this is the parametrization
    # poselib differentiates
    f[0] = e[0]
    f[1] = e[1]
    f[2] = f1 * e[2]
    f[3] = e[3]
    f[4] = e[4]
    f[5] = f1 * e[5]
    f[6] = f2 * e[6]
    f[7] = f2 * e[7]
    f[8] = f1 * f2 * e[8]


@njit(cache=True, inline='always', fastmath=True)
def _essential_tangent_rows(e, model, B, nt):
    # rows 0..2 of B: dE/dw_k = E skew(e_k) (retraction R exp([w]_x));
    # rows 3..5: dE/dt_k = skew(e_k) R; remaining rows zeroed
    for j in range(3):
        B[0, 3 * j] = 0.0
        B[0, 3 * j + 1] = e[3 * j + 2]
        B[0, 3 * j + 2] = -e[3 * j + 1]
        B[1, 3 * j] = -e[3 * j + 2]
        B[1, 3 * j + 1] = 0.0
        B[1, 3 * j + 2] = e[3 * j]
        B[2, 3 * j] = e[3 * j + 1]
        B[2, 3 * j + 1] = -e[3 * j]
        B[2, 3 * j + 2] = 0.0
    for j in range(3):
        B[3, j] = 0.0
        B[3, 3 + j] = -model[6 + j]
        B[3, 6 + j] = model[3 + j]
        B[4, j] = model[6 + j]
        B[4, 3 + j] = 0.0
        B[4, 6 + j] = -model[j]
        B[5, j] = -model[3 + j]
        B[5, 3 + j] = model[j]
        B[5, 6 + j] = 0.0
    for p in range(6, nt):
        for j in range(9):
            B[p, j] = 0.0


@njit(cache=True, inline='always', fastmath=True)
def _sampson_dsdF(x, y, xp, yp, f, dsdF):
    # Sampson residual of one correspondence for the flat 3x3 matrix f, plus
    # its gradient w.r.t. the matrix entries (row-major). Returns
    # (residual, valid); valid is False only for a degenerate denominator.
    # Callers weight the residual themselves via loss.weight(s_i^2, ...).
    fx1_0 = f[0] * x + f[1] * y + f[2]
    fx1_1 = f[3] * x + f[4] * y + f[5]
    fx1_2 = f[6] * x + f[7] * y + f[8]
    ftx2_0 = f[0] * xp + f[3] * yp + f[6]
    ftx2_1 = f[1] * xp + f[4] * yp + f[7]
    residual = xp * fx1_0 + yp * fx1_1 + fx1_2
    denominator = (fx1_0 * fx1_0 + fx1_1 * fx1_1
                   + ftx2_0 * ftx2_0 + ftx2_1 * ftx2_1)
    if denominator <= 0.0:
        return 0.0, False

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
    return s_i, True


@njit(cache=True, inline='always', fastmath=True)
def _sampson_sq(x, y, xp, yp, f):
    # squared Sampson residual (untruncated); 1e300 for a degenerate
    # denominator
    fx1_0 = f[0] * x + f[1] * y + f[2]
    fx1_1 = f[3] * x + f[4] * y + f[5]
    fx1_2 = f[6] * x + f[7] * y + f[8]
    ftx2_0 = f[0] * xp + f[3] * yp + f[6]
    ftx2_1 = f[1] * xp + f[4] * yp + f[7]
    residual = xp * fx1_0 + yp * fx1_1 + fx1_2
    denominator = (fx1_0 * fx1_0 + fx1_1 * fx1_1
                   + ftx2_0 * ftx2_0 + ftx2_1 * ftx2_1)
    if denominator <= 0.0:
        return 1e300
    return residual * residual / denominator


# ---------------------------------------------------------------------------
# hybrid cost kernels (scorer signature; the LM engine minimizes these)
# ---------------------------------------------------------------------------

def build_monodepth_hybrid_cost(loss):
    # hybrid cost for calibrated monodepth models [R | t | scale | shift1 |
    # shift2]; per-term contributions are robustified by loss.cost, so the
    # LM loop's accept/reject test matches the accumulate kernel's weighting
    cost_fn = loss.cost

    @njit(cache=True, fastmath=True)
    def _monodepth_hybrid_cost(model, data, max_error_sq, best_score):
        x1_x, x1_y, x2_x, x2_y = data[0], data[1], data[2], data[3]
        d1 = data[4]
        d2 = data[5]
        scale_reproj = data[6]
        weight_sampson = data[7]
        n = x1_x.shape[0]
        s = model[12]
        u = model[13]
        v = model[14]

        e = np.empty(9)
        essential_from_pose(model, e)

        cost = 0.0
        num_inliers = 0
        for i in range(n):
            x = x1_x[i]
            y = x1_y[i]
            xp = x2_x[i]
            yp = x2_y[i]

            if weight_sampson > 0.0:
                r2 = _sampson_sq(x, y, xp, yp, e)
                cost += weight_sampson * cost_fn(r2, max_error_sq)
                if r2 < max_error_sq:
                    num_inliers += 1

            if scale_reproj > 0.0:
                z1 = d1[i] + u
                Z1x = model[0] * z1 * x + model[1] * z1 * y + model[2] * z1 + model[9]
                Z1y = model[3] * z1 * x + model[4] * z1 * y + model[5] * z1 + model[10]
                Z1z = model[6] * z1 * x + model[7] * z1 * y + model[8] * z1 + model[11]
                if Z1z > 0.0:
                    inv_z = 1.0 / Z1z
                    rx = Z1x * inv_z - xp
                    ry = Z1y * inv_z - yp
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    cost += cost_fn(r2, max_error_sq)

                z2 = s * (d2[i] + v)
                X2x = z2 * xp - model[9]
                X2y = z2 * yp - model[10]
                X2z = z2 - model[11]
                Z2x = model[0] * X2x + model[3] * X2y + model[6] * X2z
                Z2y = model[1] * X2x + model[4] * X2y + model[7] * X2z
                Z2z = model[2] * X2x + model[5] * X2y + model[8] * X2z
                if Z2z > 0.0:
                    inv_z = 1.0 / Z2z
                    rx = Z2x * inv_z - x
                    ry = Z2y * inv_z - y
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    cost += cost_fn(r2, max_error_sq)

        return cost, num_inliers

    return _monodepth_hybrid_cost


def build_monodepth_focal_hybrid_cost(loss):
    # hybrid cost for monodepth focal models [R | t | f1 | f2 | scale] in
    # centered pixel coordinates; per-term contributions robustified by
    # loss.cost, consistent with the accumulate kernel's weighting
    cost_fn = loss.cost

    @njit(cache=True, fastmath=True)
    def _monodepth_focal_hybrid_cost(model, data, max_error_sq, best_score):
        x1_x, x1_y, x2_x, x2_y = data[0], data[1], data[2], data[3]
        d1 = data[4]
        d2 = data[5]
        scale_reproj = data[6]
        weight_sampson = data[7]
        n = x1_x.shape[0]
        f1 = model[12]
        f2 = model[13]
        s = model[14]
        if f1 <= 0.0 or f2 <= 0.0:
            return 1e300, 0
        inv_f1 = 1.0 / f1
        inv_f2 = 1.0 / f2

        e = np.empty(9)
        fmat = np.empty(9)
        essential_from_pose(model, e)
        _focal_fundamental(e, f1, f2, fmat)

        cost = 0.0
        num_inliers = 0
        for i in range(n):
            x = x1_x[i]
            y = x1_y[i]
            xp = x2_x[i]
            yp = x2_y[i]

            if weight_sampson > 0.0:
                r2 = _sampson_sq(x, y, xp, yp, fmat)
                cost += weight_sampson * cost_fn(r2, max_error_sq)
                if r2 < max_error_sq:
                    num_inliers += 1

            if scale_reproj > 0.0:
                # bearing b1 = (x/f1, y/f1, 1); forward projects with f2
                b1x = x * inv_f1
                b1y = y * inv_f1
                z1 = d1[i]
                Z1x = model[0] * z1 * b1x + model[1] * z1 * b1y + model[2] * z1 + model[9]
                Z1y = model[3] * z1 * b1x + model[4] * z1 * b1y + model[5] * z1 + model[10]
                Z1z = model[6] * z1 * b1x + model[7] * z1 * b1y + model[8] * z1 + model[11]
                if Z1z > 0.0:
                    inv_z = 1.0 / Z1z
                    rx = f2 * Z1x * inv_z - xp
                    ry = f2 * Z1y * inv_z - yp
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    cost += cost_fn(r2, max_error_sq)

                z2 = s * d2[i]
                X2x = z2 * xp * inv_f2 - model[9]
                X2y = z2 * yp * inv_f2 - model[10]
                X2z = z2 - model[11]
                Z2x = model[0] * X2x + model[3] * X2y + model[6] * X2z
                Z2y = model[1] * X2x + model[4] * X2y + model[7] * X2z
                Z2z = model[2] * X2x + model[5] * X2y + model[8] * X2z
                if Z2z > 0.0:
                    inv_z = 1.0 / Z2z
                    rx = f1 * Z2x * inv_z - x
                    ry = f1 * Z2y * inv_z - y
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    cost += cost_fn(r2, max_error_sq)

        return cost, num_inliers

    return _monodepth_focal_hybrid_cost


monodepth_hybrid_cost = build_monodepth_hybrid_cost(TruncatedLoss())
monodepth_focal_hybrid_cost = build_monodepth_focal_hybrid_cost(TruncatedLoss())


# ---------------------------------------------------------------------------
# normal-equation accumulation
# ---------------------------------------------------------------------------

def _make_accumulate_calibrated(num_tangent, loss):
    # num_tangent 7 (fixed shifts) or 9 (refined shifts)
    refine_shift = num_tangent > 7
    weight_fn = loss.weight

    @njit(cache=True, fastmath=True)
    def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
        x1_x, x1_y, x2_x, x2_y = data[0], data[1], data[2], data[3]
        d1 = data[4]
        d2 = data[5]
        scale_reproj = data[6]
        weight_sampson = data[7]
        n = x1_x.shape[0]
        s = f[12]
        u = f[13]
        v = f[14]

        for p in range(num_tangent):
            Jtr[p] = 0.0
            for q in range(num_tangent):
                JtJ[p, q] = 0.0

        e = np.empty(9)
        essential_from_pose(f, e)
        B = np.empty((num_tangent, 9))
        _essential_tangent_rows(e, f, B, num_tangent)
        dsdF = np.empty(9)
        J = np.empty(num_tangent)
        J0 = np.empty(num_tangent)
        J1 = np.empty(num_tangent)

        num_residuals = 0
        for i in range(n):
            x = x1_x[i]
            y = x1_y[i]
            xp = x2_x[i]
            yp = x2_y[i]

            if scale_reproj > 0.0:
                # forward reprojection (cam1 -> cam2)
                z1 = d1[i] + u
                X1x = z1 * x
                X1y = z1 * y
                X1z = z1
                Z1x = f[0] * X1x + f[1] * X1y + f[2] * X1z + f[9]
                Z1y = f[3] * X1x + f[4] * X1y + f[5] * X1z + f[10]
                Z1z = f[6] * X1x + f[7] * X1y + f[8] * X1z + f[11]
                if Z1z > 0.0:
                    inv_z = 1.0 / Z1z
                    px = Z1x * inv_z
                    py = Z1y * inv_z
                    rx = px - xp
                    ry = py - yp
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    w = weight_fn(r2, max_error_sq)
                    if w > 0.0:
                        num_residuals += 2
                        # Jproj = [[1, 0, -px], [0, 1, -py]] * inv_z;
                        # dZ = Jproj @ R (2x3)
                        dz00 = (f[0] - px * f[6]) * inv_z
                        dz01 = (f[1] - px * f[7]) * inv_z
                        dz02 = (f[2] - px * f[8]) * inv_z
                        dz10 = (f[3] - py * f[6]) * inv_z
                        dz11 = (f[4] - py * f[7]) * inv_z
                        dz12 = (f[5] - py * f[8]) * inv_z
                        J0[0] = -X1z * dz01 + X1y * dz02
                        J0[1] = X1z * dz00 - X1x * dz02
                        J0[2] = -X1y * dz00 + X1x * dz01
                        J1[0] = -X1z * dz11 + X1y * dz12
                        J1[1] = X1z * dz10 - X1x * dz12
                        J1[2] = -X1y * dz10 + X1x * dz11
                        J0[3] = inv_z
                        J0[4] = 0.0
                        J0[5] = -px * inv_z
                        J1[3] = 0.0
                        J1[4] = inv_z
                        J1[5] = -py * inv_z
                        J0[6] = 0.0
                        J1[6] = 0.0
                        if refine_shift:
                            # dX1/dshift1 = x1h
                            J0[7] = dz00 * x + dz01 * y + dz02
                            J1[7] = dz10 * x + dz11 * y + dz12
                            J0[8] = 0.0
                            J1[8] = 0.0
                        wsr = w * scale_reproj
                        for p in range(num_tangent):
                            Jtr[p] += wsr * (J0[p] * rx + J1[p] * ry)
                            for q in range(p, num_tangent):
                                JtJ[p, q] += wsr * (J0[p] * J0[q]
                                                    + J1[p] * J1[q])

                # backward reprojection (cam2 -> cam1)
                w2 = d2[i] + v
                X2sx = w2 * xp
                X2sy = w2 * yp
                X2sz = w2
                X2tx = s * X2sx - f[9]
                X2ty = s * X2sy - f[10]
                X2tz = s * X2sz - f[11]
                Z2x = f[0] * X2tx + f[3] * X2ty + f[6] * X2tz
                Z2y = f[1] * X2tx + f[4] * X2ty + f[7] * X2tz
                Z2z = f[2] * X2tx + f[5] * X2ty + f[8] * X2tz
                if Z2z > 0.0:
                    inv_z = 1.0 / Z2z
                    px = Z2x * inv_z
                    py = Z2y * inv_z
                    rx = px - x
                    ry = py - y
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    w = weight_fn(r2, max_error_sq)
                    if w > 0.0:
                        num_residuals += 2
                        # y_vec = R^T X2t = Z2; dZ2/dw_k = Z2 x e_k
                        # Jproj rows through R^T for translation and scale
                        # a-rows: Jproj @ R^T
                        a00 = (f[0] - px * f[2]) * inv_z
                        a01 = (f[3] - px * f[5]) * inv_z
                        a02 = (f[6] - px * f[8]) * inv_z
                        a10 = (f[1] - py * f[2]) * inv_z
                        a11 = (f[4] - py * f[5]) * inv_z
                        a12 = (f[7] - py * f[8]) * inv_z
                        # rotation: Jproj @ (Z2 x e_k)
                        # (Z2 x e_0) = (0, Z2z, -Z2y) etc.; Jproj = [[1,0,-px],
                        # [0,1,-py]] * inv_z applied to camera-frame vectors
                        J0[0] = (-px * (-Z2y)) * inv_z
                        J0[1] = (-Z2z - px * Z2x) * inv_z
                        J0[2] = Z2y * inv_z
                        J1[0] = (Z2z - py * (-Z2y)) * inv_z
                        J1[1] = (-py * Z2x) * inv_z
                        J1[2] = -Z2x * inv_z
                        # translation: -Jproj @ R^T
                        J0[3] = -a00
                        J0[4] = -a01
                        J0[5] = -a02
                        J1[3] = -a10
                        J1[4] = -a11
                        J1[5] = -a12
                        # scale: Jproj @ R^T @ X2s
                        J0[6] = a00 * X2sx + a01 * X2sy + a02 * X2sz
                        J1[6] = a10 * X2sx + a11 * X2sy + a12 * X2sz
                        if refine_shift:
                            J0[7] = 0.0
                            J1[7] = 0.0
                            # dX2/dshift2 = s * x2h
                            J0[8] = s * (a00 * xp + a01 * yp + a02)
                            J1[8] = s * (a10 * xp + a11 * yp + a12)
                        wsr = w * scale_reproj
                        for p in range(num_tangent):
                            Jtr[p] += wsr * (J0[p] * rx + J1[p] * ry)
                            for q in range(p, num_tangent):
                                JtJ[p, q] += wsr * (J0[p] * J0[q]
                                                    + J1[p] * J1[q])

            if weight_sampson > 0.0:
                s_i, valid = _sampson_dsdF(x, y, xp, yp, e, dsdF)
                if valid:
                    w = weight_fn(s_i * s_i, max_error_sq)
                    if w > 0.0:
                        num_residuals += 1
                        for p in range(6):
                            acc = 0.0
                            for j in range(9):
                                acc += dsdF[j] * B[p, j]
                            J[p] = acc
                        for p in range(6, num_tangent):
                            J[p] = 0.0
                        wsamp = w * weight_sampson
                        for p in range(num_tangent):
                            Jtr[p] += wsamp * J[p] * s_i
                            for q in range(p, num_tangent):
                                JtJ[p, q] += wsamp * J[p] * J[q]

        for p in range(num_tangent):
            for q in range(p):
                JtJ[p, q] = JtJ[q, p]
        return num_residuals

    return _accumulate


def _make_accumulate_focal(shared, loss):
    # shared focal: tangent [w, dt, dscale, df] (8); varying focal:
    # [w, dt, dscale, df1, df2] (9)
    num_tangent = 8 if shared else 9
    weight_fn = loss.weight

    @njit(cache=True, fastmath=True)
    def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
        x1_x, x1_y, x2_x, x2_y = data[0], data[1], data[2], data[3]
        d1 = data[4]
        d2 = data[5]
        scale_reproj = data[6]
        weight_sampson = data[7]
        n = x1_x.shape[0]
        f1 = f[12]
        f2 = f[13]
        s = f[14]
        if f1 <= 0.0 or f2 <= 0.0:
            return 0
        inv_f1 = 1.0 / f1
        inv_f2 = 1.0 / f2

        for p in range(num_tangent):
            Jtr[p] = 0.0
            for q in range(num_tangent):
                JtJ[p, q] = 0.0

        e = np.empty(9)
        essential_from_pose(f, e)
        fmat = np.empty(9)
        _focal_fundamental(e, f1, f2, fmat)

        # dF/dtheta rows: pose rows are the E rows scaled entrywise by the
        # diag(1,1,f2) . diag(1,1,f1) pattern; then the focal rows
        B = np.empty((num_tangent, 9))
        _essential_tangent_rows(e, f, B, num_tangent)
        for p in range(6):
            B[p, 2] *= f1
            B[p, 5] *= f1
            B[p, 6] *= f2
            B[p, 7] *= f2
            B[p, 8] *= f1 * f2
        if shared:
            # dF/df with f1 = f2 = f
            B[7, 2] = e[2]
            B[7, 5] = e[5]
            B[7, 6] = e[6]
            B[7, 7] = e[7]
            B[7, 8] = 2.0 * f1 * e[8]
        else:
            B[7, 2] = e[2]
            B[7, 5] = e[5]
            B[7, 8] = f2 * e[8]
            B[8, 6] = e[6]
            B[8, 7] = e[7]
            B[8, 8] = f1 * e[8]

        dsdF = np.empty(9)
        J = np.empty(num_tangent)
        J0 = np.empty(num_tangent)
        J1 = np.empty(num_tangent)

        num_residuals = 0
        for i in range(n):
            x = x1_x[i]
            y = x1_y[i]
            xp = x2_x[i]
            yp = x2_y[i]

            if scale_reproj > 0.0:
                # forward reprojection (cam1 -> cam2, projected with f2)
                b1x = x * inv_f1
                b1y = y * inv_f1
                z1 = d1[i]
                X1x = z1 * b1x
                X1y = z1 * b1y
                X1z = z1
                Z1x = f[0] * X1x + f[1] * X1y + f[2] * X1z + f[9]
                Z1y = f[3] * X1x + f[4] * X1y + f[5] * X1z + f[10]
                Z1z = f[6] * X1x + f[7] * X1y + f[8] * X1z + f[11]
                if Z1z > 0.0:
                    inv_z = 1.0 / Z1z
                    px = Z1x * inv_z
                    py = Z1y * inv_z
                    rx = f2 * px - xp
                    ry = f2 * py - yp
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    w = weight_fn(r2, max_error_sq)
                    if w > 0.0:
                        num_residuals += 2
                        # Jproj = f2 * [[1, 0, -px], [0, 1, -py]] * inv_z
                        c = f2 * inv_z
                        dz00 = (f[0] - px * f[6]) * c
                        dz01 = (f[1] - px * f[7]) * c
                        dz02 = (f[2] - px * f[8]) * c
                        dz10 = (f[3] - py * f[6]) * c
                        dz11 = (f[4] - py * f[7]) * c
                        dz12 = (f[5] - py * f[8]) * c
                        J0[0] = -X1z * dz01 + X1y * dz02
                        J0[1] = X1z * dz00 - X1x * dz02
                        J0[2] = -X1y * dz00 + X1x * dz01
                        J1[0] = -X1z * dz11 + X1y * dz12
                        J1[1] = X1z * dz10 - X1x * dz12
                        J1[2] = -X1y * dz10 + X1x * dz11
                        J0[3] = c
                        J0[4] = 0.0
                        J0[5] = -px * c
                        J1[3] = 0.0
                        J1[4] = c
                        J1[5] = -py * c
                        J0[6] = 0.0
                        J1[6] = 0.0
                        # dX1/df1 = z1 * (-x/f1^2, -y/f1^2, 0)
                        dxf = -z1 * b1x * inv_f1
                        dyf = -z1 * b1y * inv_f1
                        df10 = dz00 * dxf + dz01 * dyf
                        df11 = dz10 * dxf + dz11 * dyf
                        if shared:
                            J0[7] = px + df10
                            J1[7] = py + df11
                        else:
                            J0[7] = df10
                            J1[7] = df11
                            J0[8] = px
                            J1[8] = py
                        wsr = w * scale_reproj
                        for p in range(num_tangent):
                            Jtr[p] += wsr * (J0[p] * rx + J1[p] * ry)
                            for q in range(p, num_tangent):
                                JtJ[p, q] += wsr * (J0[p] * J0[q]
                                                    + J1[p] * J1[q])

                # backward reprojection (cam2 -> cam1, projected with f1)
                b2x = xp * inv_f2
                b2y = yp * inv_f2
                w2 = d2[i]
                X2sx = w2 * b2x
                X2sy = w2 * b2y
                X2sz = w2
                X2tx = s * X2sx - f[9]
                X2ty = s * X2sy - f[10]
                X2tz = s * X2sz - f[11]
                Z2x = f[0] * X2tx + f[3] * X2ty + f[6] * X2tz
                Z2y = f[1] * X2tx + f[4] * X2ty + f[7] * X2tz
                Z2z = f[2] * X2tx + f[5] * X2ty + f[8] * X2tz
                if Z2z > 0.0:
                    inv_z = 1.0 / Z2z
                    px = Z2x * inv_z
                    py = Z2y * inv_z
                    rx = f1 * px - x
                    ry = f1 * py - y
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    w = weight_fn(r2, max_error_sq)
                    if w > 0.0:
                        num_residuals += 2
                        c = f1 * inv_z
                        a00 = (f[0] - px * f[2]) * c
                        a01 = (f[3] - px * f[5]) * c
                        a02 = (f[6] - px * f[8]) * c
                        a10 = (f[1] - py * f[2]) * c
                        a11 = (f[4] - py * f[5]) * c
                        a12 = (f[7] - py * f[8]) * c
                        # rotation: Jproj @ (Z2 x e_k), Jproj scaled by f1
                        J0[0] = (-px * (-Z2y)) * c
                        J0[1] = (-Z2z - px * Z2x) * c
                        J0[2] = Z2y * c
                        J1[0] = (Z2z - py * (-Z2y)) * c
                        J1[1] = (-py * Z2x) * c
                        J1[2] = -Z2x * c
                        J0[3] = -a00
                        J0[4] = -a01
                        J0[5] = -a02
                        J1[3] = -a10
                        J1[4] = -a11
                        J1[5] = -a12
                        J0[6] = a00 * X2sx + a01 * X2sy + a02 * X2sz
                        J1[6] = a10 * X2sx + a11 * X2sy + a12 * X2sz
                        # dX2/df2 = s * w2 * (-xp/f2^2, -yp/f2^2, 0)
                        dxf = -s * w2 * b2x * inv_f2
                        dyf = -s * w2 * b2y * inv_f2
                        df20 = a00 * dxf + a01 * dyf
                        df21 = a10 * dxf + a11 * dyf
                        if shared:
                            J0[7] = px + df20
                            J1[7] = py + df21
                        else:
                            J0[7] = px
                            J1[7] = py
                            J0[8] = df20
                            J1[8] = df21
                        wsr = w * scale_reproj
                        for p in range(num_tangent):
                            Jtr[p] += wsr * (J0[p] * rx + J1[p] * ry)
                            for q in range(p, num_tangent):
                                JtJ[p, q] += wsr * (J0[p] * J0[q]
                                                    + J1[p] * J1[q])

            if weight_sampson > 0.0:
                s_i, valid = _sampson_dsdF(x, y, xp, yp, fmat, dsdF)
                if valid:
                    w = weight_fn(s_i * s_i, max_error_sq)
                    if w > 0.0:
                        num_residuals += 1
                        for p in range(num_tangent):
                            if p == 6:
                                J[p] = 0.0  # Sampson does not depend on scale
                                continue
                            acc = 0.0
                            for j in range(9):
                                acc += dsdF[j] * B[p, j]
                            J[p] = acc
                        wsamp = w * weight_sampson
                        for p in range(num_tangent):
                            Jtr[p] += wsamp * J[p] * s_i
                            for q in range(p, num_tangent):
                                JtJ[p, q] += wsamp * J[p] * J[q]

        for p in range(num_tangent):
            for q in range(p):
                JtJ[p, q] = JtJ[q, p]
        return num_residuals

    return _accumulate


_accumulate_calibrated = _make_accumulate_calibrated(7, TruncatedLoss())
_accumulate_calibrated_shift = _make_accumulate_calibrated(9, TruncatedLoss())
_accumulate_shared_focal = _make_accumulate_focal(True, TruncatedLoss())
_accumulate_varying_focal = _make_accumulate_focal(False, TruncatedLoss())

_refine_monodepth_lm = build_lm_refine(
    _init_state, _state_to_model, _accumulate_calibrated,
    _apply_step_calibrated, monodepth_hybrid_cost,
    STATE_SIZE, 7, MODEL_SIZE)
_refine_monodepth_shift_lm = build_lm_refine(
    _init_state, _state_to_model, _accumulate_calibrated_shift,
    _apply_step_calibrated, monodepth_hybrid_cost,
    STATE_SIZE, 9, MODEL_SIZE)
_refine_monodepth_shared_focal_lm = build_lm_refine(
    _init_state_focal, _state_to_model, _accumulate_shared_focal,
    _apply_step_shared_focal, monodepth_focal_hybrid_cost,
    STATE_SIZE, 8, MODEL_SIZE)
_refine_monodepth_varying_focal_lm = build_lm_refine(
    _init_state_focal, _state_to_model, _accumulate_varying_focal,
    _apply_step_varying_focal, monodepth_focal_hybrid_cost,
    STATE_SIZE, 9, MODEL_SIZE)


_final_kernels_calibrated = {}


def _get_final_refine_calibrated(loss):
    # lazily compiles (and caches) the LM kernel for a non-default loss; see
    # essential.py's _get_final_refine for the general rationale
    key = type(loss)
    if key not in _final_kernels_calibrated:
        accumulate_final = _make_accumulate_calibrated(7, loss)
        cost_final = build_monodepth_hybrid_cost(loss)
        _final_kernels_calibrated[key] = build_lm_refine(
            _init_state, _state_to_model, accumulate_final,
            _apply_step_calibrated, cost_final, STATE_SIZE, 7, MODEL_SIZE)
    return _final_kernels_calibrated[key]


_final_kernels_calibrated_shift = {}


def _get_final_refine_calibrated_shift(loss):
    key = type(loss)
    if key not in _final_kernels_calibrated_shift:
        accumulate_final = _make_accumulate_calibrated(9, loss)
        cost_final = build_monodepth_hybrid_cost(loss)
        _final_kernels_calibrated_shift[key] = build_lm_refine(
            _init_state, _state_to_model, accumulate_final,
            _apply_step_calibrated, cost_final, STATE_SIZE, 9, MODEL_SIZE)
    return _final_kernels_calibrated_shift[key]


_final_kernels_shared_focal = {}


def _get_final_refine_shared_focal(loss):
    key = type(loss)
    if key not in _final_kernels_shared_focal:
        accumulate_final = _make_accumulate_focal(True, loss)
        cost_final = build_monodepth_focal_hybrid_cost(loss)
        _final_kernels_shared_focal[key] = build_lm_refine(
            _init_state_focal, _state_to_model, accumulate_final,
            _apply_step_shared_focal, cost_final, STATE_SIZE, 8, MODEL_SIZE)
    return _final_kernels_shared_focal[key]


_final_kernels_varying_focal = {}


def _get_final_refine_varying_focal(loss):
    key = type(loss)
    if key not in _final_kernels_varying_focal:
        accumulate_final = _make_accumulate_focal(False, loss)
        cost_final = build_monodepth_focal_hybrid_cost(loss)
        _final_kernels_varying_focal[key] = build_lm_refine(
            _init_state_focal, _state_to_model, accumulate_final,
            _apply_step_varying_focal, cost_final, STATE_SIZE, 9, MODEL_SIZE)
    return _final_kernels_varying_focal[key]


class LMMonoDepthPoseRefiner():
    # calibrated monodepth: pose + scale (7 tangent parameters). `loss`
    # selects the robust cost/weighting, either as a Loss object or as one
    # of the names in refiners/losses.py's LOSSES ('truncated', 'cauchy',
    # 'truncated_cauchy'). TruncatedLoss is the default, matching every
    # RANSAC-internal use; the others are meant for a final polish pass on
    # an inlier-only subset.
    def __init__(self, num_iterations=25, loss=TruncatedLoss()):
        loss = get_loss(loss)
        self.num_iterations = num_iterations
        self.loss = loss
        if not isinstance(loss, TruncatedLoss):
            self.refine = _get_final_refine_calibrated(loss)

    refine = staticmethod(_refine_monodepth_lm)


class LMMonoDepthShiftPoseRefiner():
    # calibrated monodepth: pose + scale + both shifts (9 tangent parameters)
    def __init__(self, num_iterations=25, loss=TruncatedLoss()):
        loss = get_loss(loss)
        self.num_iterations = num_iterations
        self.loss = loss
        if not isinstance(loss, TruncatedLoss):
            self.refine = _get_final_refine_calibrated_shift(loss)

    refine = staticmethod(_refine_monodepth_shift_lm)


class LMMonoDepthSharedFocalPoseRefiner():
    # monodepth pose + scale + shared focal (8 tangent parameters)
    def __init__(self, num_iterations=25, loss=TruncatedLoss()):
        loss = get_loss(loss)
        self.num_iterations = num_iterations
        self.loss = loss
        if not isinstance(loss, TruncatedLoss):
            self.refine = _get_final_refine_shared_focal(loss)

    refine = staticmethod(_refine_monodepth_shared_focal_lm)


class LMMonoDepthVaryingFocalPoseRefiner():
    # monodepth pose + scale + two focals (9 tangent parameters)
    def __init__(self, num_iterations=25, loss=TruncatedLoss()):
        loss = get_loss(loss)
        self.num_iterations = num_iterations
        self.loss = loss
        if not isinstance(loss, TruncatedLoss):
            self.refine = _get_final_refine_varying_focal(loss)

    refine = staticmethod(_refine_monodepth_varying_focal_lm)
