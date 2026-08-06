"""Minimal solvers for relative pose with monocular depth estimates.

Ports of PoseLib's monodepth relative pose solvers (BSD-3-Clause):

- `relpose_monodepth_3pt.cc`: calibrated relative pose + depth scale and
  per-image depth shifts from 3 points (scale/affine-invariant depths).
  The problem reduces to a quartic whose roots are polished with a few
  Gauss-Newton steps on the three distance constraints.
- P3P variant (poselib's `RelativePoseMonoDepthEstimator` without shift):
  3D points from camera-1 depths, absolute pose of camera 2 via the shared
  P3P kernel, depth scale from the first correspondence.
- `relpose_monodepth_3pt_shared_focal.cc`: relative pose + scale + one
  shared focal length; the third camera-2 depth is treated as unknown and
  found as an eigenvalue of a 4x4 action matrix (Danilevsky + Sturm here).
- `relpose_monodepth_3pt_varying_focal.cc`: relative pose + scale + two
  focal lengths from a single 3x3 linear system.

Depth convention (poselib's `MonoDepthTwoViewGeometry`): the 3D point of
correspondence i is `(d1_i + shift1) * x1h_i` in camera 1 and
`scale * (d2_i + shift2) * x2h_i` in camera 2, with `x2h = R x1h`-side
points in homogeneous (calibrated or centered-pixel) coordinates. The
solvers without shift assume shift1 = shift2 = 0.

Model layouts (15 flat parameters each):
- calibrated: `[R | t | scale | shift1 | shift2]`
- focal:      `[R | t | f1 | f2 | scale]` (shared focal keeps f1 == f2),
  matching the varying-focal relative pose layout in the first 14 entries.

`data` layout for all monodepth problems: six contiguous float64 columns
(x1_x, x1_y, x2_x, x2_y, d1, d2) followed by the two scalar hybrid
refinement weights (scale_reproj, weight_sampson); the solvers and scorers
only use the columns. Calibrated solvers expect calibrated coordinates,
the focal solvers centered pixel coordinates (known principal points).
"""

import math

import numpy as np
from numba import njit

from fastpose.solvers.essential import _real_roots_sturm
from fastpose.solvers.p3p import _p3p_impl
from fastpose.solvers.p4pf import _project_rotation, _solve_3xn_neg
from fastpose.solvers.shared_focal import _charpoly_danilevsky_n

MODEL_SIZE = 15


@njit(cache=True, fastmath=True)
def _pose_from_point_triplets(A, B, model):
    # rigid motion mapping the three 3D points in the rows of A onto the
    # rows of B: R = [b01 | b02 | b01 x b02] [a01 | a02 | a01 x a02]^-1
    # (poselib's Y X^-1 construction), t = B0 - R A0; R is snapped back to
    # SO(3) via the quaternion round-trip like poselib's pose constructor.
    # Writes [R | t] into model[0:12]; False for collinear points.
    a10 = A[0, 0] - A[1, 0]
    a11 = A[0, 1] - A[1, 1]
    a12 = A[0, 2] - A[1, 2]
    a20 = A[0, 0] - A[2, 0]
    a21 = A[0, 1] - A[2, 1]
    a22 = A[0, 2] - A[2, 2]
    ac0 = a11 * a22 - a12 * a21
    ac1 = a12 * a20 - a10 * a22
    ac2 = a10 * a21 - a11 * a20
    det = ac0 * ac0 + ac1 * ac1 + ac2 * ac2
    if det <= 0.0:
        return False
    inv_det = 1.0 / det

    # rows of [a01 | a02 | a01 x a02]^-1 (cross products of the other two
    # columns, like the P3P kernel)
    i00 = (a21 * ac2 - a22 * ac1) * inv_det
    i01 = (a22 * ac0 - a20 * ac2) * inv_det
    i02 = (a20 * ac1 - a21 * ac0) * inv_det
    i10 = (ac1 * a12 - ac2 * a11) * inv_det
    i11 = (ac2 * a10 - ac0 * a12) * inv_det
    i12 = (ac0 * a11 - ac1 * a10) * inv_det
    i20 = ac0 * inv_det
    i21 = ac1 * inv_det
    i22 = ac2 * inv_det

    b10 = B[0, 0] - B[1, 0]
    b11 = B[0, 1] - B[1, 1]
    b12 = B[0, 2] - B[1, 2]
    b20 = B[0, 0] - B[2, 0]
    b21 = B[0, 1] - B[2, 1]
    b22 = B[0, 2] - B[2, 2]
    bc0 = b11 * b22 - b12 * b21
    bc1 = b12 * b20 - b10 * b22
    bc2 = b10 * b21 - b11 * b20

    # R = [b01 | b02 | b01 x b02] @ inv (columns times rows of inv)
    model[0] = b10 * i00 + b20 * i10 + bc0 * i20
    model[1] = b10 * i01 + b20 * i11 + bc0 * i21
    model[2] = b10 * i02 + b20 * i12 + bc0 * i22
    model[3] = b11 * i00 + b21 * i10 + bc1 * i20
    model[4] = b11 * i01 + b21 * i11 + bc1 * i21
    model[5] = b11 * i02 + b21 * i12 + bc1 * i22
    model[6] = b12 * i00 + b22 * i10 + bc2 * i20
    model[7] = b12 * i01 + b22 * i11 + bc2 * i21
    model[8] = b12 * i02 + b22 * i12 + bc2 * i22

    # t = B0 - R A0 with the unprojected R, like poselib
    model[9] = B[0, 0] - (model[0] * A[0, 0] + model[1] * A[0, 1] + model[2] * A[0, 2])
    model[10] = B[0, 1] - (model[3] * A[0, 0] + model[4] * A[0, 1] + model[5] * A[0, 2])
    model[11] = B[0, 2] - (model[6] * A[0, 0] + model[7] * A[0, 1] + model[8] * A[0, 2])

    _project_rotation(model)
    return True


@njit(cache=True, fastmath=True)
def _refine_suv(coeffs, s2, u, v):
    # Gauss-Newton polish of (s^2, u, v) on the three pairwise distance
    # constraints (poselib's refine_suv); coeffs is the 18-vector of
    # quadratic coefficients, monomials [s*v^2, u^2, s*v, s, u, 1] per row
    for _ in range(5):
        r0 = (coeffs[0] * s2 * v * v + coeffs[1] * u * u + coeffs[2] * s2 * v
              + coeffs[3] * s2 + coeffs[4] * u + coeffs[5])
        r1 = (coeffs[6] * s2 * v * v + coeffs[7] * u * u + coeffs[8] * s2 * v
              + coeffs[9] * s2 + coeffs[10] * u + coeffs[11])
        r2 = (coeffs[12] * s2 * v * v + coeffs[13] * u * u + coeffs[14] * s2 * v
              + coeffs[15] * s2 + coeffs[16] * u + coeffs[17])
        if abs(r0) + abs(r1) + abs(r2) < 1e-10:
            break
        j00 = coeffs[0] * v * v + coeffs[2] * v + coeffs[3]
        j01 = 2.0 * coeffs[1] * u + coeffs[4]
        j02 = 2.0 * coeffs[0] * s2 * v + coeffs[2] * s2
        j10 = coeffs[6] * v * v + coeffs[8] * v + coeffs[9]
        j11 = 2.0 * coeffs[7] * u + coeffs[10]
        j12 = 2.0 * coeffs[6] * s2 * v + coeffs[8] * s2
        j20 = coeffs[12] * v * v + coeffs[14] * v + coeffs[15]
        j21 = 2.0 * coeffs[13] * u + coeffs[16]
        j22 = 2.0 * coeffs[12] * s2 * v + coeffs[14] * s2
        det = (j00 * (j11 * j22 - j12 * j21)
               - j01 * (j10 * j22 - j12 * j20)
               + j02 * (j10 * j21 - j11 * j20))
        if det == 0.0:
            break
        inv = 1.0 / det
        s2 -= ((j11 * j22 - j12 * j21) * r0
               + (j02 * j21 - j01 * j22) * r1
               + (j01 * j12 - j02 * j11) * r2) * inv
        u -= ((j12 * j20 - j10 * j22) * r0
              + (j00 * j22 - j02 * j20) * r1
              + (j02 * j10 - j00 * j12) * r2) * inv
        v -= ((j10 * j21 - j11 * j20) * r0
              + (j01 * j20 - j00 * j21) * r1
              + (j00 * j11 - j01 * j10) * r2) * inv
    return s2, u, v


@njit(cache=True, fastmath=True)
def _solve_monodepth_shift_3pt(data, sample, models, workspace):
    # calibrated relative pose + depth scale + two depth shifts from three
    # points; port of poselib's relpose_monodepth_3pt (solver_p3p_mono_3d)
    x1_x, x1_y, x2_x, x2_y, dd1, dd2 = data[0], data[1], data[2], data[3], data[4], data[5]

    o = 0
    d = workspace[o:o + 18]
    o += 18
    coeffs = workspace[o:o + 18]
    o += 18
    C0 = workspace[o:o + 9].reshape(3, 3)
    o += 9
    C1 = workspace[o:o + 9].reshape(3, 3)
    o += 9
    coef = workspace[o:o + 5]
    o += 5
    chain = workspace[o:o + 25].reshape(5, 5)
    o += 25
    roots = workspace[o:o + 4]
    o += 4
    lo_stack = workspace[o:o + 64]
    o += 64
    hi_stack = workspace[o:o + 64]
    o += 64
    A = workspace[o:o + 9].reshape(3, 3)
    o += 9
    B = workspace[o:o + 9].reshape(3, 3)
    o += 9

    # data vector layout of the generated solver:
    # [x1_x(3), x1_y(3), x2_x(3), x2_y(3), d1(3), d2(3)]
    for k in range(3):
        i = sample[k]
        d[k] = x1_x[i]
        d[3 + k] = x1_y[i]
        d[6 + k] = x2_x[i]
        d[9 + k] = x2_y[i]
        d[12 + k] = dd1[i]
        d[15 + k] = dd2[i]

    coeffs[0] = d[6] ** 2 - 2 * d[6] * d[7] + d[7] ** 2 + d[9] ** 2 - 2 * d[9] * d[10] + d[10] ** 2
    coeffs[1] = -d[0] ** 2 + 2 * d[0] * d[1] - d[1] ** 2 - d[3] ** 2 + 2 * d[3] * d[4] - d[4] ** 2
    coeffs[2] = (2 * d[6] ** 2 * d[15] - 2 * d[6] * d[7] * d[15] + 2 * d[9] ** 2 * d[15]
                 - 2 * d[9] * d[10] * d[15] - 2 * d[6] * d[7] * d[16] + 2 * d[7] ** 2 * d[16]
                 - 2 * d[9] * d[10] * d[16] + 2 * d[10] ** 2 * d[16])
    coeffs[3] = (d[6] ** 2 * d[15] ** 2 + d[9] ** 2 * d[15] ** 2
                 - 2 * d[6] * d[7] * d[15] * d[16] - 2 * d[9] * d[10] * d[15] * d[16]
                 + d[7] ** 2 * d[16] ** 2 + d[10] ** 2 * d[16] ** 2 + d[15] ** 2
                 - 2 * d[15] * d[16] + d[16] ** 2)
    coeffs[4] = (-2 * d[0] ** 2 * d[12] + 2 * d[0] * d[1] * d[12] - 2 * d[3] ** 2 * d[12]
                 + 2 * d[3] * d[4] * d[12] + 2 * d[0] * d[1] * d[13] - 2 * d[1] ** 2 * d[13]
                 + 2 * d[3] * d[4] * d[13] - 2 * d[4] ** 2 * d[13])
    coeffs[5] = (-d[0] ** 2 * d[12] ** 2 - d[3] ** 2 * d[12] ** 2
                 + 2 * d[0] * d[1] * d[12] * d[13] + 2 * d[3] * d[4] * d[12] * d[13]
                 - d[1] ** 2 * d[13] ** 2 - d[4] ** 2 * d[13] ** 2 - d[12] ** 2
                 + 2 * d[12] * d[13] - d[13] ** 2)
    coeffs[6] = d[6] ** 2 - 2 * d[6] * d[8] + d[8] ** 2 + d[9] ** 2 - 2 * d[9] * d[11] + d[11] ** 2
    coeffs[7] = -d[0] ** 2 + 2 * d[0] * d[2] - d[2] ** 2 - d[3] ** 2 + 2 * d[3] * d[5] - d[5] ** 2
    coeffs[8] = (2 * d[6] ** 2 * d[15] - 2 * d[6] * d[8] * d[15] + 2 * d[9] ** 2 * d[15]
                 - 2 * d[9] * d[11] * d[15] - 2 * d[6] * d[8] * d[17] + 2 * d[8] ** 2 * d[17]
                 - 2 * d[9] * d[11] * d[17] + 2 * d[11] ** 2 * d[17])
    coeffs[9] = (d[6] ** 2 * d[15] ** 2 + d[9] ** 2 * d[15] ** 2
                 - 2 * d[6] * d[8] * d[15] * d[17] - 2 * d[9] * d[11] * d[15] * d[17]
                 + d[8] ** 2 * d[17] ** 2 + d[11] ** 2 * d[17] ** 2 + d[15] ** 2
                 - 2 * d[15] * d[17] + d[17] ** 2)
    coeffs[10] = (-2 * d[0] ** 2 * d[12] + 2 * d[0] * d[2] * d[12] - 2 * d[3] ** 2 * d[12]
                  + 2 * d[3] * d[5] * d[12] + 2 * d[0] * d[2] * d[14] - 2 * d[2] ** 2 * d[14]
                  + 2 * d[3] * d[5] * d[14] - 2 * d[5] ** 2 * d[14])
    coeffs[11] = (-d[0] ** 2 * d[12] ** 2 - d[3] ** 2 * d[12] ** 2
                  + 2 * d[0] * d[2] * d[12] * d[14] + 2 * d[3] * d[5] * d[12] * d[14]
                  - d[2] ** 2 * d[14] ** 2 - d[5] ** 2 * d[14] ** 2 - d[12] ** 2
                  + 2 * d[12] * d[14] - d[14] ** 2)
    coeffs[12] = d[7] ** 2 - 2 * d[7] * d[8] + d[8] ** 2 + d[10] ** 2 - 2 * d[10] * d[11] + d[11] ** 2
    coeffs[13] = -d[1] ** 2 + 2 * d[1] * d[2] - d[2] ** 2 - d[4] ** 2 + 2 * d[4] * d[5] - d[5] ** 2
    coeffs[14] = (2 * d[7] ** 2 * d[16] - 2 * d[7] * d[8] * d[16] + 2 * d[10] ** 2 * d[16]
                  - 2 * d[10] * d[11] * d[16] - 2 * d[7] * d[8] * d[17] + 2 * d[8] ** 2 * d[17]
                  - 2 * d[10] * d[11] * d[17] + 2 * d[11] ** 2 * d[17])
    coeffs[15] = (d[7] ** 2 * d[16] ** 2 + d[10] ** 2 * d[16] ** 2
                  - 2 * d[7] * d[8] * d[16] * d[17] - 2 * d[10] * d[11] * d[16] * d[17]
                  + d[8] ** 2 * d[17] ** 2 + d[11] ** 2 * d[17] ** 2 + d[16] ** 2
                  - 2 * d[16] * d[17] + d[17] ** 2)
    coeffs[16] = (-2 * d[1] ** 2 * d[13] + 2 * d[1] * d[2] * d[13] - 2 * d[4] ** 2 * d[13]
                  + 2 * d[4] * d[5] * d[13] + 2 * d[1] * d[2] * d[14] - 2 * d[2] ** 2 * d[14]
                  + 2 * d[4] * d[5] * d[14] - 2 * d[5] ** 2 * d[14])
    coeffs[17] = (-d[1] ** 2 * d[13] ** 2 - d[4] ** 2 * d[13] ** 2
                  + 2 * d[1] * d[2] * d[13] * d[14] + 2 * d[4] * d[5] * d[13] * d[14]
                  - d[2] ** 2 * d[14] ** 2 - d[5] ** 2 * d[14] ** 2 - d[13] ** 2
                  + 2 * d[13] * d[14] - d[14] ** 2)

    # C2 = -C0^-1 C1, monomials [s^2, s^2 v, s^2 v^2] vs [1, u, u^2]
    C0[0, 0] = coeffs[0]
    C0[0, 1] = coeffs[2]
    C0[0, 2] = coeffs[3]
    C0[1, 0] = coeffs[6]
    C0[1, 1] = coeffs[8]
    C0[1, 2] = coeffs[9]
    C0[2, 0] = coeffs[12]
    C0[2, 1] = coeffs[14]
    C0[2, 2] = coeffs[15]
    C1[0, 0] = coeffs[1]
    C1[0, 1] = coeffs[4]
    C1[0, 2] = coeffs[5]
    C1[1, 0] = coeffs[7]
    C1[1, 1] = coeffs[10]
    C1[1, 2] = coeffs[11]
    C1[2, 0] = coeffs[13]
    C1[2, 1] = coeffs[16]
    C1[2, 2] = coeffs[17]
    if not _solve_3xn_neg(C0, C1, 3):
        return 0
    k0 = C1[0, 0]
    k1 = C1[0, 1]
    k2 = C1[0, 2]
    k3 = C1[1, 0]
    k4 = C1[1, 1]
    k5 = C1[1, 2]
    k6 = C1[2, 0]
    k7 = C1[2, 1]
    k8 = C1[2, 2]

    den = k3 * k3 - k0 * k6
    if den == 0.0:
        return 0
    c4 = 1.0 / den
    coef[4] = 1.0
    coef[3] = c4 * (2 * k3 * k4 - k1 * k6 - k0 * k7)
    coef[2] = c4 * (k4 * k4 - k0 * k8 - k1 * k7 - k2 * k6 + 2 * k3 * k5)
    coef[1] = c4 * (2 * k4 * k5 - k2 * k7 - k1 * k8)
    coef[0] = c4 * (k5 * k5 - k2 * k8)

    iw = np.empty(133, dtype=np.int64)
    degs = iw[0:5]
    slo_stack = iw[5:69]
    shi_stack = iw[69:133]
    n_roots = _real_roots_sturm(coef, 4, chain, degs, roots,
                                lo_stack, hi_stack, slo_stack, shi_stack)

    count = 0
    for ri in range(n_roots):
        u = roots[ri]
        ss = k6 * u * u + k7 * u + k8  # s^2
        if ss < 0.001:
            continue
        v = (k3 * u * u + k4 * u + k5) / ss

        ok = True
        for k in range(3):
            if d[15 + k] + v <= 0.0 or d[12 + k] + u <= 0.0:
                ok = False
                break
        if not ok:
            continue

        ss, u, v = _refine_suv(coeffs, ss, u, v)
        if ss <= 0.0:
            continue
        s = math.sqrt(ss)

        for k in range(3):
            z1 = d[12 + k] + u
            A[k, 0] = z1 * d[k]
            A[k, 1] = z1 * d[3 + k]
            A[k, 2] = z1
            z2 = s * (d[15 + k] + v)
            B[k, 0] = z2 * d[6 + k]
            B[k, 1] = z2 * d[9 + k]
            B[k, 2] = z2
        if not _pose_from_point_triplets(A, B, models[count]):
            continue
        models[count, 12] = s
        models[count, 13] = u
        models[count, 14] = v
        count += 1

    return count


@njit(cache=True, fastmath=True)
def _solve_monodepth_p3p(data, sample, models, workspace):
    # calibrated relative pose + depth scale, shifts fixed at zero: 3D
    # points from the camera-1 depths, absolute pose of camera 2 via P3P,
    # scale from the first correspondence (poselib's monodepth estimator
    # with estimate_shift=false; the scale is read off the z-row instead of
    # the x-row - identical for P3P poses, but immune to x2_x ~ 0)
    x1_x, x1_y, x2_x, x2_y, dd1, dd2 = data[0], data[1], data[2], data[3], data[4], data[5]
    xs = workspace[0:9].reshape(3, 3)
    Xs = workspace[9:18].reshape(3, 3)
    X01 = workspace[18:21]
    X02 = workspace[21:24]
    XXinv = workspace[24:33].reshape(3, 3)
    C = workspace[33:42].reshape(3, 3)
    pq = workspace[42:48].reshape(2, 3)
    taus = workspace[48:50]

    i0 = sample[0]
    d2_0 = dd2[i0]
    if d2_0 <= 0.0:
        return 0
    X0_x = dd1[i0] * x1_x[i0]
    X0_y = dd1[i0] * x1_y[i0]
    X0_z = dd1[i0]

    for k in range(3):
        i = sample[k]
        nx = x2_x[i]
        ny = x2_y[i]
        inv = 1.0 / math.sqrt(nx * nx + ny * ny + 1.0)
        xs[k, 0] = nx * inv
        xs[k, 1] = ny * inv
        xs[k, 2] = inv
        z1 = dd1[i]
        Xs[k, 0] = z1 * x1_x[i]
        Xs[k, 1] = z1 * x1_y[i]
        Xs[k, 2] = z1

    count = _p3p_impl(xs, Xs, X01, X02, XXinv, C, pq, taus, models)
    inv_d2 = 1.0 / d2_0
    for m in range(count):
        # R X0 + t = z2 * (x2, y2, 1), so the z-row gives z2 directly
        models[m, 12] = (models[m, 6] * X0_x + models[m, 7] * X0_y
                         + models[m, 8] * X0_z + models[m, 11]) * inv_d2
        models[m, 13] = 0.0
        models[m, 14] = 0.0
    return count


@njit(cache=True, fastmath=True)
def _solve_monodepth_shared_focal_3pt(data, sample, models, workspace):
    # relative pose + depth scale + shared focal length from three points in
    # centered pixel coordinates; port of relpose_monodepth_3pt_shared_focal.
    # The depth of the third point in image 2 is an unknown, found as a real
    # eigenvalue of a 4x4 action matrix (charpoly + Sturm instead of Eigen's
    # eigensolver).
    x1_x, x1_y, x2_x, x2_y, dd1, dd2 = data[0], data[1], data[2], data[3], data[4], data[5]

    o = 0
    a = workspace[o:o + 17]
    o += 17
    b = workspace[o:o + 12]
    o += 12
    c = workspace[o:o + 18]
    o += 18
    dd = workspace[o:o + 21]
    o += 21
    C0 = workspace[o:o + 9].reshape(3, 3)
    o += 9
    C1 = workspace[o:o + 12].reshape(3, 4)
    o += 12
    AM = workspace[o:o + 16].reshape(4, 4)
    o += 16
    coef = workspace[o:o + 5]
    o += 5
    chain = workspace[o:o + 25].reshape(5, 5)
    o += 25
    roots = workspace[o:o + 4]
    o += 4
    lo_stack = workspace[o:o + 64]
    o += 64
    hi_stack = workspace[o:o + 64]
    o += 64
    row = workspace[o:o + 4]
    o += 4
    tmp_row = workspace[o:o + 4]
    o += 4
    A = workspace[o:o + 9].reshape(3, 3)
    o += 9
    B = workspace[o:o + 9].reshape(3, 3)
    o += 9

    i0 = sample[0]
    i1 = sample[1]
    i2 = sample[2]

    # X1 = [d1_k * x1h_k], X2 = [d2_0 * x2h_0 | d2_1 * x2h_1 | x2h_2]
    a[0] = dd1[i0] * x1_x[i0]
    a[1] = dd1[i1] * x1_x[i1]
    a[2] = dd1[i2] * x1_x[i2]
    a[3] = dd1[i0] * x1_y[i0]
    a[4] = dd1[i1] * x1_y[i1]
    a[5] = dd1[i2] * x1_y[i2]
    a[6] = dd1[i0]
    a[7] = dd1[i1]
    a[8] = dd1[i2]
    a[9] = dd2[i0] * x2_x[i0]
    a[10] = dd2[i1] * x2_x[i1]
    a[11] = x2_x[i2]
    a[12] = dd2[i0] * x2_y[i0]
    a[13] = dd2[i1] * x2_y[i1]
    a[14] = x2_y[i2]
    a[15] = dd2[i0]
    a[16] = dd2[i1]

    b[0] = a[0] - a[1]
    b[1] = a[3] - a[4]
    b[2] = a[6] - a[7]
    b[3] = a[0] - a[2]
    b[4] = a[3] - a[5]
    b[5] = a[6] - a[8]
    b[6] = a[1] - a[2]
    b[7] = a[4] - a[5]
    b[8] = a[7] - a[8]
    b[9] = a[9] - a[10]
    b[10] = a[12] - a[13]
    b[11] = a[15] - a[16]

    c[0] = -b[11] * b[11]
    c[1] = b[2] * b[2]
    c[2] = -b[9] * b[9] - b[10] * b[10]
    c[3] = b[0] * b[0] + b[1] * b[1]
    c[4] = -1.0
    c[5] = 2 * a[15]
    c[6] = -a[15] * a[15]
    c[7] = b[5] * b[5]
    c[8] = -a[11] * a[11] - a[14] * a[14]
    c[9] = 2 * a[9] * a[11] + 2 * a[12] * a[14]
    c[10] = -a[9] * a[9] - a[12] * a[12]
    c[11] = b[3] * b[3] + b[4] * b[4]
    c[12] = 2 * a[16] - 2 * a[15]
    c[13] = a[15] * a[15] - a[16] * a[16]
    c[14] = b[8] * b[8] - b[5] * b[5]
    c[15] = 2 * a[10] * a[11] - 2 * a[9] * a[11] - 2 * a[12] * a[14] + 2 * a[13] * a[14]
    c[16] = a[9] * a[9] - a[10] * a[10] + a[12] * a[12] - a[13] * a[13]
    c[17] = -b[3] * b[3] - b[4] * b[4] + b[6] * b[6] + b[7] * b[7]

    den1 = a[6] - a[7]
    den2 = 2 * (a[6] - a[7]) * (a[15] - a[16])
    den3 = a[6] + a[7] - 2 * a[8]
    if den1 == 0.0 or den2 == 0.0 or den3 == 0.0:
        return 0
    dd[6] = 1.0 / den1
    dd[0] = (-c[3] * c[8]) * dd[6]
    dd[1] = (-c[3] * c[9]) * dd[6]
    dd[2] = (c[2] * c[11] - c[3] * c[10]) * dd[6]
    dd[3] = (-c[3] * c[4] - c[1] * c[8]) * dd[6]
    dd[4] = (-c[3] * c[5] - c[1] * c[9]) * dd[6]
    dd[5] = (c[2] * c[7] - c[3] * c[6] + c[0] * c[11] - c[1] * c[10]) * dd[6]
    dd[7] = (a[6] * a[16] - 2 * a[6] * a[15] + a[7] * a[15] + a[8] * a[15] - a[8] * a[16]) * dd[6]

    dd[8] = 1.0 / den2
    dd[9] = (-c[3] * c[15]) * dd[8]
    dd[10] = (c[2] * c[17] - c[3] * c[16]) * dd[8]
    dd[11] = (-c[3] * c[12] - c[1] * c[15]) * dd[8]
    dd[12] = (c[2] * c[14] - c[3] * c[13] + c[0] * c[17] - c[1] * c[16]) * dd[8]

    dd[13] = 1.0 / den3
    dd[14] = (a[8] * a[15] - a[7] * a[15] - a[6] * a[16] + a[8] * a[16]) * dd[13]
    dd[15] = (c[8] * c[17]) * dd[13]
    dd[16] = (c[9] * c[17] - c[11] * c[15]) * dd[13]
    dd[17] = (c[10] * c[17] - c[11] * c[16]) * dd[13]
    dd[18] = (c[4] * c[17] + c[8] * c[14]) * dd[13]
    dd[19] = (c[5] * c[17] - c[7] * c[15] + c[9] * c[14] - c[11] * c[12]) * dd[13]
    dd[20] = (c[6] * c[17] - c[7] * c[16] + c[10] * c[14] - c[11] * c[13]) * dd[13]

    C0[0, 0] = dd[2]
    C0[0, 1] = dd[5]
    C0[0, 2] = dd[7]
    C0[1, 0] = dd[10]
    C0[1, 1] = dd[12]
    C0[1, 2] = 1.0
    C0[2, 0] = dd[17]
    C0[2, 1] = dd[20]
    C0[2, 2] = dd[14]
    C1[0, 0] = dd[0] - dd[9]
    C1[0, 1] = dd[3] - dd[11]
    C1[0, 2] = dd[1] - dd[10]
    C1[0, 3] = dd[4] - dd[12]
    C1[1, 0] = 0.0
    C1[1, 1] = 0.0
    C1[1, 2] = dd[9]
    C1[1, 3] = dd[11]
    C1[2, 0] = dd[15] - dd[9]
    C1[2, 1] = dd[18] - dd[11]
    C1[2, 2] = dd[16] - dd[10]
    C1[2, 3] = dd[19] - dd[12]
    if not _solve_3xn_neg(C0, C1, 4):
        return 0

    # action matrix; its real eigenvalues are the inverse third depths
    for r in range(4):
        for cc in range(4):
            AM[r, cc] = 0.0
    AM[0, 2] = 1.0
    AM[1, 3] = 1.0
    for cc in range(4):
        AM[2, cc] = C1[0, cc]
        AM[3, cc] = C1[1, cc]

    if not _charpoly_danilevsky_n(AM, coef, row, tmp_row, 4):
        return 0
    iw = np.empty(133, dtype=np.int64)
    degs = iw[0:5]
    slo_stack = iw[5:69]
    shi_stack = iw[69:133]
    n_roots = _real_roots_sturm(coef, 4, chain, degs, roots,
                                lo_stack, hi_stack, slo_stack, shi_stack)

    count = 0
    for ri in range(n_roots):
        lam = roots[ri]
        if lam < 1e-12:
            continue
        d3 = 1.0 / lam

        a00 = (dd[3] - dd[11]) * d3 * d3 + (dd[4] - dd[12]) * d3 + dd[5]
        a01 = dd[7]
        a10 = dd[12] + dd[11] * d3
        a11 = 1.0
        r0 = (dd[0] - dd[9]) * d3 * d3 + (dd[1] - dd[10]) * d3 + dd[2]
        r1 = dd[10] + dd[9] * d3
        det = a00 * a11 - a01 * a10
        if det == 0.0:
            continue
        inv = 1.0 / det
        f_sq = -(a11 * r0 - a01 * r1) * inv
        if f_sq < 0.0:
            continue

        den = c[0] * f_sq + c[2]
        if den == 0.0:
            continue
        s2 = -(c[1] * f_sq + c[3]) / den
        if s2 < 0.001:
            continue
        s = math.sqrt(s2)
        f = math.sqrt(f_sq)
        inv_f = 1.0 / f

        # camera-frame point triplets with Kinv = diag(1/f, 1/f, 1)
        A[0, 0] = a[0] * inv_f
        A[0, 1] = a[3] * inv_f
        A[0, 2] = a[6]
        A[1, 0] = a[1] * inv_f
        A[1, 1] = a[4] * inv_f
        A[1, 2] = a[7]
        A[2, 0] = a[2] * inv_f
        A[2, 1] = a[5] * inv_f
        A[2, 2] = a[8]
        B[0, 0] = s * a[9] * inv_f
        B[0, 1] = s * a[12] * inv_f
        B[0, 2] = s * a[15]
        B[1, 0] = s * a[10] * inv_f
        B[1, 1] = s * a[13] * inv_f
        B[1, 2] = s * a[16]
        B[2, 0] = s * d3 * a[11] * inv_f
        B[2, 1] = s * d3 * a[14] * inv_f
        B[2, 2] = s * d3
        if not _pose_from_point_triplets(A, B, models[count]):
            continue
        models[count, 12] = f
        models[count, 13] = f
        models[count, 14] = s
        count += 1

    return count


@njit(cache=True, fastmath=True)
def _solve_monodepth_varying_focal_3pt(data, sample, models, workspace):
    # relative pose + depth scale + two focal lengths from three points in
    # centered pixel coordinates; port of relpose_monodepth_3pt_varying_focal
    # (a single 3x3 linear system in (1/f1^2, s^2/f2^2, s^2))
    x1_x, x1_y, x2_x, x2_y, dd1, dd2 = data[0], data[1], data[2], data[3], data[4], data[5]

    a = workspace[0:18]
    b = workspace[18:36]
    A3 = workspace[36:45].reshape(3, 3)
    B3 = workspace[45:54].reshape(3, 3)
    A = workspace[54:63].reshape(3, 3)
    B = workspace[63:72].reshape(3, 3)

    i0 = sample[0]
    i1 = sample[1]
    i2 = sample[2]
    a[0] = x1_x[i0] * dd1[i0]
    a[1] = x1_x[i1] * dd1[i1]
    a[2] = x1_x[i2] * dd1[i2]
    a[3] = x1_y[i0] * dd1[i0]
    a[4] = x1_y[i1] * dd1[i1]
    a[5] = x1_y[i2] * dd1[i2]
    a[6] = dd1[i0]
    a[7] = dd1[i1]
    a[8] = dd1[i2]
    a[9] = x2_x[i0] * dd2[i0]
    a[10] = x2_x[i1] * dd2[i1]
    a[11] = x2_x[i2] * dd2[i2]
    a[12] = x2_y[i0] * dd2[i0]
    a[13] = x2_y[i1] * dd2[i1]
    a[14] = x2_y[i2] * dd2[i2]
    a[15] = dd2[i0]
    a[16] = dd2[i1]
    a[17] = dd2[i2]

    b[0] = a[0] - a[1]
    b[1] = a[3] - a[4]
    b[2] = a[6] - a[7]
    b[3] = a[0] - a[2]
    b[4] = a[3] - a[5]
    b[5] = a[6] - a[8]
    b[6] = a[1] - a[2]
    b[7] = a[4] - a[5]
    b[8] = a[7] - a[8]
    b[9] = a[9] - a[10]
    b[10] = a[12] - a[13]
    b[11] = a[15] - a[16]
    b[12] = a[9] - a[11]
    b[13] = a[12] - a[14]
    b[14] = a[15] - a[17]
    b[15] = a[10] - a[11]
    b[16] = a[13] - a[14]
    b[17] = a[16] - a[17]

    A3[0, 0] = b[0] * b[0] + b[1] * b[1]
    A3[0, 1] = -b[9] * b[9] - b[10] * b[10]
    A3[0, 2] = -b[11] * b[11]
    A3[1, 0] = b[3] * b[3] + b[4] * b[4]
    A3[1, 1] = -b[12] * b[12] - b[13] * b[13]
    A3[1, 2] = -b[14] * b[14]
    A3[2, 0] = b[6] * b[6] + b[7] * b[7]
    A3[2, 1] = -b[15] * b[15] - b[16] * b[16]
    A3[2, 2] = -b[17] * b[17]
    B3[0, 0] = b[2] * b[2]
    B3[1, 0] = b[5] * b[5]
    B3[2, 0] = b[8] * b[8]
    if not _solve_3xn_neg(A3, B3, 1):
        return 0
    if B3[0, 0] <= 0.0 or B3[1, 0] <= 0.0 or B3[2, 0] <= 0.0:
        return 0

    inv_f1 = math.sqrt(B3[0, 0])   # 1 / f1
    s = math.sqrt(B3[2, 0])
    inv_f2 = math.sqrt(B3[1, 0] / B3[2, 0])  # 1 / f2

    for k in range(3):
        A[k, 0] = a[k] * inv_f1
        A[k, 1] = a[3 + k] * inv_f1
        A[k, 2] = a[6 + k]
        B[k, 0] = s * a[9 + k] * inv_f2
        B[k, 1] = s * a[12 + k] * inv_f2
        B[k, 2] = s * a[15 + k]
    if not _pose_from_point_triplets(A, B, models[0]):
        return 0
    models[0, 12] = 1.0 / inv_f1
    models[0, 13] = 1.0 / inv_f2
    models[0, 14] = s
    return 1


# ---------------------------------------------------------------------------
# pluggable component classes
# ---------------------------------------------------------------------------

class MonoDepthShiftSolver():
    # calibrated relative pose + depth scale + two depth shifts from 3
    # points; models [R | t | scale | shift1 | shift2]
    sample_size = 3
    num_params = MODEL_SIZE
    max_models = 4
    workspace_size = 234
    solve = staticmethod(_solve_monodepth_shift_3pt)


class MonoDepthP3PSolver():
    # calibrated relative pose + depth scale (shifts fixed at zero) via P3P;
    # models [R | t | scale | 0 | 0]
    sample_size = 3
    num_params = MODEL_SIZE
    max_models = 4
    workspace_size = 50
    solve = staticmethod(_solve_monodepth_p3p)


class MonoDepthSharedFocalSolver():
    # relative pose + depth scale + shared focal length from 3 points in
    # centered pixel coordinates; models [R | t | f | f | scale]
    sample_size = 3
    num_params = MODEL_SIZE
    max_models = 4
    workspace_size = 293
    solve = staticmethod(_solve_monodepth_shared_focal_3pt)


class MonoDepthVaryingFocalSolver():
    # relative pose + depth scale + two focal lengths from 3 points in
    # centered pixel coordinates; models [R | t | f1 | f2 | scale]
    sample_size = 3
    num_params = MODEL_SIZE
    max_models = 1
    workspace_size = 72
    solve = staticmethod(_solve_monodepth_varying_focal_3pt)
