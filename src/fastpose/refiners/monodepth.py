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

The per-point maths - the two reprojection residuals and their jacobian rows,
the E -> F map and the tangent basis - is built by
`build_monodepth_primitives` / `build_monodepth_reproj_kernels` so the same
source compiles for the CPU (`njit`) and for CUDA
(`cuda.jit(device=True)`); the accumulate kernels below are thin loops over
them. The Sampson residual and its jacobian are *not* redefined here: they
are the same `sampson_residual` / `sampson_point_jacobian` every other
refiner uses.
"""

import math

import numpy as np
from numba import float64, njit

from fastpose.jit_backend import cpu_jit
from fastpose.refiners.lm import build_lm_refine
from fastpose.refiners.losses import TruncatedLoss, get_loss
from fastpose.refiners.utils import (mat3_mul, rodrigues,
                                     sampson_point_jacobian)
from fastpose.scorers.sampson import essential_from_pose, sampson_residual

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


def build_monodepth_primitives(jit, real=float64):
    # `real` types the float literals, so the CUDA refiner can build these in
    # float32 without a bare `1.0` silently promoting the chain back to
    # float64; see build_sampson_point_kernels for the full rationale.
    @jit(fastmath=True, inline=True)
    def focal_fundamental(e, f1, f2, f):
        # F = diag(1, 1, f2) E diag(1, 1, f1), proportional to K2^-T E K1^-1,
        # so the Sampson error matches the scorer's; this is the
        # parametrization poselib differentiates
        f[0] = e[0]
        f[1] = e[1]
        f[2] = f1 * e[2]
        f[3] = e[3]
        f[4] = e[4]
        f[5] = f1 * e[5]
        f[6] = f2 * e[6]
        f[7] = f2 * e[7]
        f[8] = f1 * f2 * e[8]

    @jit(fastmath=True, inline=True)
    def essential_tangent_rows(e, model, B, nt):
        # rows 0..2 of B: dE/dw_k = E skew(e_k) (retraction R exp([w]_x));
        # rows 3..5: dE/dt_k = skew(e_k) R; remaining rows zeroed.
        #
        # Not the same map as refiners/utils.py's essential_tangent_rows_core:
        # there the translation is confined to the unit sphere and gets two
        # tangent directions, here the depths fix the scale and t gets three
        # additive ones.
        zero = real(0.0)
        for j in range(3):
            B[0, 3 * j] = zero
            B[0, 3 * j + 1] = e[3 * j + 2]
            B[0, 3 * j + 2] = -e[3 * j + 1]
            B[1, 3 * j] = -e[3 * j + 2]
            B[1, 3 * j + 1] = zero
            B[1, 3 * j + 2] = e[3 * j]
            B[2, 3 * j] = e[3 * j + 1]
            B[2, 3 * j + 1] = -e[3 * j]
            B[2, 3 * j + 2] = zero
        for j in range(3):
            B[3, j] = zero
            B[3, 3 + j] = -model[6 + j]
            B[3, 6 + j] = model[3 + j]
            B[4, j] = model[6 + j]
            B[4, 3 + j] = zero
            B[4, 6 + j] = -model[j]
            B[5, j] = -model[3 + j]
            B[5, 3 + j] = model[j]
            B[5, 6 + j] = zero
        for p in range(6, nt):
            for j in range(9):
                B[p, j] = zero

    @jit(fastmath=True, inline=True)
    def focal_tangent_rows(e, f1, f2, B, shared):
        # the focal columns of dF/dtheta, and the diag(1,1,f2).diag(1,1,f1)
        # rescaling the pose columns pick up on the way from E to F
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
            B[7, 8] = real(2.0) * f1 * e[8]
        else:
            B[7, 2] = e[2]
            B[7, 5] = e[5]
            B[7, 8] = f2 * e[8]
            B[8, 6] = e[6]
            B[8, 7] = e[7]
            B[8, 8] = f1 * e[8]

    return {
        'focal_fundamental': focal_fundamental,
        'essential_tangent_rows': essential_tangent_rows,
        'focal_tangent_rows': focal_tangent_rows,
    }


def build_monodepth_residual_kernels(jit, real=float64, focal=False):
    """The two monodepth reprojection residuals, without their jacobians.

    The cost evaluation is O(n) per LM trial step and needs only `(rx, ry)`,
    so it does not pay for the jacobian rows. Same split as
    `sampson_residual` versus `sampson_point_jacobian`.
    """

    @jit(fastmath=True, inline=True)
    def forward_residual(m, x, y, xp, yp, di):
        zero = real(0.0)
        if focal:
            inv_f1 = real(1.0) / m[12]
            X1x = di * x * inv_f1
            X1y = di * y * inv_f1
            X1z = di
        else:
            z1 = di + m[13]
            X1x = z1 * x
            X1y = z1 * y
            X1z = z1
        Z1x = m[0] * X1x + m[1] * X1y + m[2] * X1z + m[9]
        Z1y = m[3] * X1x + m[4] * X1y + m[5] * X1z + m[10]
        Z1z = m[6] * X1x + m[7] * X1y + m[8] * X1z + m[11]
        if Z1z <= zero:
            return zero, zero, False
        inv_z = real(1.0) / Z1z
        if focal:
            f2 = m[13]
            return (f2 * Z1x * inv_z - xp, f2 * Z1y * inv_z - yp, True)
        return Z1x * inv_z - xp, Z1y * inv_z - yp, True

    @jit(fastmath=True, inline=True)
    def backward_residual(m, x, y, xp, yp, di):
        zero = real(0.0)
        if focal:
            inv_f2 = real(1.0) / m[13]
            s = m[14]
            X2sx = di * xp * inv_f2
            X2sy = di * yp * inv_f2
            X2sz = di
        else:
            s = m[12]
            w2 = di + m[14]
            X2sx = w2 * xp
            X2sy = w2 * yp
            X2sz = w2
        X2tx = s * X2sx - m[9]
        X2ty = s * X2sy - m[10]
        X2tz = s * X2sz - m[11]
        Z2x = m[0] * X2tx + m[3] * X2ty + m[6] * X2tz
        Z2y = m[1] * X2tx + m[4] * X2ty + m[7] * X2tz
        Z2z = m[2] * X2tx + m[5] * X2ty + m[8] * X2tz
        if Z2z <= zero:
            return zero, zero, False
        inv_z = real(1.0) / Z2z
        if focal:
            f1 = m[12]
            return (f1 * Z2x * inv_z - x, f1 * Z2y * inv_z - y, True)
        return Z2x * inv_z - x, Z2y * inv_z - y, True

    return {'forward_residual': forward_residual,
            'backward_residual': backward_residual}


def build_monodepth_reproj_kernels(jit, real=float64, num_tangent=7,
                                   focal=False):
    """The two monodepth reprojection residuals and their jacobian rows.

    `num_tangent` and `focal` are compile-time flags, exactly as they are for
    the accumulate kernels built from them: 7 or 9 for the calibrated problem
    (fixed or refined shifts), 8 or 9 for the focal one (shared or varying).

    Like `reprojection_point_jacobian` in refiners/absolute.py, each kernel
    returns `(rx, ry, ok)` and fills `J0`/`J1` in the same call; the caller
    applies `loss.weight` afterwards and drops the point when it is zero.
    """
    refine_shift = (not focal) and num_tangent > 7
    shared = focal and num_tangent == 8

    @jit(fastmath=True, inline=True)
    def forward_point(m, x, y, xp, yp, di, J0, J1):
        # camera 1 -> camera 2: the 3D point the depth induces in camera 1,
        # projected into image 2
        zero = real(0.0)
        if focal:
            inv_f1 = real(1.0) / m[12]
            f2 = m[13]
            X1x = di * x * inv_f1
            X1y = di * y * inv_f1
            X1z = di
        else:
            z1 = di + m[13]
            X1x = z1 * x
            X1y = z1 * y
            X1z = z1
        Z1x = m[0] * X1x + m[1] * X1y + m[2] * X1z + m[9]
        Z1y = m[3] * X1x + m[4] * X1y + m[5] * X1z + m[10]
        Z1z = m[6] * X1x + m[7] * X1y + m[8] * X1z + m[11]
        if Z1z <= zero:
            return zero, zero, False
        inv_z = real(1.0) / Z1z
        px = Z1x * inv_z
        py = Z1y * inv_z
        if focal:
            rx = f2 * px - xp
            ry = f2 * py - yp
            c = f2 * inv_z
        else:
            rx = px - xp
            ry = py - yp
            c = inv_z

        # Jproj = c * [[1, 0, -px], [0, 1, -py]]; dz = Jproj @ R (2x3)
        dz00 = (m[0] - px * m[6]) * c
        dz01 = (m[1] - px * m[7]) * c
        dz02 = (m[2] - px * m[8]) * c
        dz10 = (m[3] - py * m[6]) * c
        dz11 = (m[4] - py * m[7]) * c
        dz12 = (m[5] - py * m[8]) * c
        J0[0] = -X1z * dz01 + X1y * dz02
        J0[1] = X1z * dz00 - X1x * dz02
        J0[2] = -X1y * dz00 + X1x * dz01
        J1[0] = -X1z * dz11 + X1y * dz12
        J1[1] = X1z * dz10 - X1x * dz12
        J1[2] = -X1y * dz10 + X1x * dz11
        J0[3] = c
        J0[4] = zero
        J0[5] = -px * c
        J1[3] = zero
        J1[4] = c
        J1[5] = -py * c
        J0[6] = zero
        J1[6] = zero
        if focal:
            # dX1/df1 = di * (-x/f1^2, -y/f1^2, 0)
            dxf = -X1x * inv_f1
            dyf = -X1y * inv_f1
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
        elif refine_shift:
            # dX1/dshift1 = x1h
            J0[7] = dz00 * x + dz01 * y + dz02
            J1[7] = dz10 * x + dz11 * y + dz12
            J0[8] = zero
            J1[8] = zero
        return rx, ry, True

    @jit(fastmath=True, inline=True)
    def backward_point(m, x, y, xp, yp, di, J0, J1):
        # camera 2 -> camera 1: the scaled 3D point the depth induces in
        # camera 2, mapped back through R^T and projected into image 1
        zero = real(0.0)
        if focal:
            f1 = m[12]
            inv_f2 = real(1.0) / m[13]
            s = m[14]
            w2 = di
            X2sx = w2 * xp * inv_f2
            X2sy = w2 * yp * inv_f2
        else:
            s = m[12]
            w2 = di + m[14]
            X2sx = w2 * xp
            X2sy = w2 * yp
        X2sz = w2
        X2tx = s * X2sx - m[9]
        X2ty = s * X2sy - m[10]
        X2tz = s * X2sz - m[11]
        Z2x = m[0] * X2tx + m[3] * X2ty + m[6] * X2tz
        Z2y = m[1] * X2tx + m[4] * X2ty + m[7] * X2tz
        Z2z = m[2] * X2tx + m[5] * X2ty + m[8] * X2tz
        if Z2z <= zero:
            return zero, zero, False
        inv_z = real(1.0) / Z2z
        px = Z2x * inv_z
        py = Z2y * inv_z
        if focal:
            rx = f1 * px - x
            ry = f1 * py - y
            c = f1 * inv_z
        else:
            rx = px - x
            ry = py - y
            c = inv_z

        # a-rows: Jproj @ R^T
        a00 = (m[0] - px * m[2]) * c
        a01 = (m[3] - px * m[5]) * c
        a02 = (m[6] - px * m[8]) * c
        a10 = (m[1] - py * m[2]) * c
        a11 = (m[4] - py * m[5]) * c
        a12 = (m[7] - py * m[8]) * c
        # rotation: Jproj @ (Z2 x e_k), with y_vec = R^T X2t = Z2
        J0[0] = (-px * (-Z2y)) * c
        J0[1] = (-Z2z - px * Z2x) * c
        J0[2] = Z2y * c
        J1[0] = (Z2z - py * (-Z2y)) * c
        J1[1] = (-py * Z2x) * c
        J1[2] = -Z2x * c
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
        if focal:
            # dX2/df2 = s * w2 * (-xp/f2^2, -yp/f2^2, 0)
            dxf = -s * X2sx * inv_f2
            dyf = -s * X2sy * inv_f2
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
        elif refine_shift:
            J0[7] = zero
            J1[7] = zero
            # dX2/dshift2 = s * x2h
            J0[8] = s * (a00 * xp + a01 * yp + a02)
            J1[8] = s * (a10 * xp + a11 * yp + a12)
        return rx, ry, True

    return {'forward_point': forward_point, 'backward_point': backward_point}


_CPU_PRIM = build_monodepth_primitives(cpu_jit)
_focal_fundamental = _CPU_PRIM['focal_fundamental']
_essential_tangent_rows = _CPU_PRIM['essential_tangent_rows']
_focal_tangent_rows = _CPU_PRIM['focal_tangent_rows']

_CPU_REPROJ = {
    (nt, focal): build_monodepth_reproj_kernels(cpu_jit, num_tangent=nt,
                                                focal=focal)
    for nt, focal in ((7, False), (9, False), (8, True), (9, True))
}
_CPU_RESID = {focal: build_monodepth_residual_kernels(cpu_jit, focal=focal)
              for focal in (False, True)}


@njit(cache=True, inline='always', fastmath=True)
def _sampson_sq(x, y, xp, yp, f):
    # squared Sampson residual (untruncated); 1e300 for a degenerate
    # denominator
    r2, den = sampson_residual(f, x, y, xp, yp)
    if den <= 0.0:
        return 1e300
    return r2 / den


# ---------------------------------------------------------------------------
# hybrid cost kernels (scorer signature; the LM engine minimizes these)
# ---------------------------------------------------------------------------

def build_monodepth_hybrid_cost(loss):
    # hybrid cost for calibrated monodepth models [R | t | scale | shift1 |
    # shift2]; per-term contributions are robustified by loss.cost, so the
    # LM loop's accept/reject test matches the accumulate kernel's weighting
    cost_fn = loss.cost
    _forward = _CPU_RESID[False]['forward_residual']
    _backward = _CPU_RESID[False]['backward_residual']

    @njit(cache=True, fastmath=True)
    def _monodepth_hybrid_cost(model, data, max_error_sq, best_score):
        x1_x, x1_y, x2_x, x2_y = data[0], data[1], data[2], data[3]
        d1 = data[4]
        d2 = data[5]
        scale_reproj = data[6]
        weight_sampson = data[7]
        n = x1_x.shape[0]

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
                rx, ry, ok = _forward(model, x, y, xp, yp, d1[i])
                if ok:
                    cost += cost_fn(scale_reproj * (rx * rx + ry * ry),
                                    max_error_sq)
                rx, ry, ok = _backward(model, x, y, xp, yp, d2[i])
                if ok:
                    cost += cost_fn(scale_reproj * (rx * rx + ry * ry),
                                    max_error_sq)

        return cost, num_inliers

    return _monodepth_hybrid_cost


def build_monodepth_focal_hybrid_cost(loss):
    # hybrid cost for monodepth focal models [R | t | f1 | f2 | scale] in
    # centered pixel coordinates; per-term contributions robustified by
    # loss.cost, consistent with the accumulate kernel's weighting
    cost_fn = loss.cost
    _forward = _CPU_RESID[True]['forward_residual']
    _backward = _CPU_RESID[True]['backward_residual']

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
        if f1 <= 0.0 or f2 <= 0.0:
            return 1e300, 0

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
                rx, ry, ok = _forward(model, x, y, xp, yp, d1[i])
                if ok:
                    cost += cost_fn(scale_reproj * (rx * rx + ry * ry),
                                    max_error_sq)
                rx, ry, ok = _backward(model, x, y, xp, yp, d2[i])
                if ok:
                    cost += cost_fn(scale_reproj * (rx * rx + ry * ry),
                                    max_error_sq)

        return cost, num_inliers

    return _monodepth_focal_hybrid_cost


monodepth_hybrid_cost = build_monodepth_hybrid_cost(TruncatedLoss())
monodepth_focal_hybrid_cost = build_monodepth_focal_hybrid_cost(TruncatedLoss())


# ---------------------------------------------------------------------------
# normal-equation accumulation
# ---------------------------------------------------------------------------

def _make_accumulate_calibrated(num_tangent, loss):
    # num_tangent 7 (fixed shifts) or 9 (refined shifts)
    weight_fn = loss.weight
    _forward = _CPU_REPROJ[(num_tangent, False)]['forward_point']
    _backward = _CPU_REPROJ[(num_tangent, False)]['backward_point']

    @njit(cache=True, fastmath=True)
    def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
        x1_x, x1_y, x2_x, x2_y = data[0], data[1], data[2], data[3]
        d1 = data[4]
        d2 = data[5]
        scale_reproj = data[6]
        weight_sampson = data[7]
        n = x1_x.shape[0]
        # the scale and the two shifts are read off `f` inside the point
        # kernels, which is why they are not unpacked here

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
                rx, ry, ok = _forward(f, x, y, xp, yp, d1[i], J0, J1)
                if ok:
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    w = weight_fn(r2, max_error_sq)
                    if w > 0.0:
                        num_residuals += 2
                        wsr = w * scale_reproj
                        for p in range(num_tangent):
                            Jtr[p] += wsr * (J0[p] * rx + J1[p] * ry)
                            for q in range(p, num_tangent):
                                JtJ[p, q] += wsr * (J0[p] * J0[q]
                                                    + J1[p] * J1[q])

                # backward reprojection (cam2 -> cam1)
                rx, ry, ok = _backward(f, x, y, xp, yp, d2[i], J0, J1)
                if ok:
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    w = weight_fn(r2, max_error_sq)
                    if w > 0.0:
                        num_residuals += 2
                        wsr = w * scale_reproj
                        for p in range(num_tangent):
                            Jtr[p] += wsr * (J0[p] * rx + J1[p] * ry)
                            for q in range(p, num_tangent):
                                JtJ[p, q] += wsr * (J0[p] * J0[q]
                                                    + J1[p] * J1[q])

            if weight_sampson > 0.0:
                s_i, valid = sampson_point_jacobian(e, x, y, xp, yp, dsdF)
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
    _forward = _CPU_REPROJ[(num_tangent, True)]['forward_point']
    _backward = _CPU_REPROJ[(num_tangent, True)]['backward_point']

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
        if f1 <= 0.0 or f2 <= 0.0:
            return 0
        # the scale and the two reciprocals are read off `f` inside the point
        # kernels, which is why they are not unpacked here

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
        _focal_tangent_rows(e, f1, f2, B, shared)

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
                rx, ry, ok = _forward(f, x, y, xp, yp, d1[i], J0, J1)
                if ok:
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    w = weight_fn(r2, max_error_sq)
                    if w > 0.0:
                        num_residuals += 2
                        wsr = w * scale_reproj
                        for p in range(num_tangent):
                            Jtr[p] += wsr * (J0[p] * rx + J1[p] * ry)
                            for q in range(p, num_tangent):
                                JtJ[p, q] += wsr * (J0[p] * J0[q]
                                                    + J1[p] * J1[q])

                # backward reprojection (cam2 -> cam1, projected with f1)
                rx, ry, ok = _backward(f, x, y, xp, yp, d2[i], J0, J1)
                if ok:
                    r2 = scale_reproj * (rx * rx + ry * ry)
                    w = weight_fn(r2, max_error_sq)
                    if w > 0.0:
                        num_residuals += 2
                        wsr = w * scale_reproj
                        for p in range(num_tangent):
                            Jtr[p] += wsr * (J0[p] * rx + J1[p] * ry)
                            for q in range(p, num_tangent):
                                JtJ[p, q] += wsr * (J0[p] * J0[q]
                                                    + J1[p] * J1[q])

            if weight_sampson > 0.0:
                s_i, valid = sampson_point_jacobian(fmat, x, y, xp, yp, dsdF)
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
