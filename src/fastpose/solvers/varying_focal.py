"""7-point relative pose solver with two unknown focal lengths.

The minimal step estimates fundamental matrices with the standard 7-point
solver, recovers the focal lengths with Rybkin's closed-form formula for
known principal points and square pixels (an SVD-free equivalent of the
Bougnoux formula), then decomposes the resulting essential matrix into a
pose model `[R | t | f1 | f2]`.
"""

import math

from numba import njit

from fastpose.solvers.essential import _pose_from_essential
from fastpose.solvers.fundamental import _det3_flat, _nullspace_7pt, _solve_cubic_real

MODEL_SIZE = 14


@njit(cache=True, inline='always')
def _rybkin_f_sq(f11, f12, f13, f21, f22, f23, f31, f32, f33):
    # Rybkin's formula: squared focal length of the *first* camera from the
    # entries of a fundamental matrix with both principal points at the
    # origin (called with the transposed entries it yields the second
    # camera's). Algebraically equivalent to the Bougnoux formula but a pure
    # polynomial ratio - no epipoles, no SVD. A zero denominator (the same
    # optical-axes-intersect degeneracy Bougnoux has) yields inf/nan, which
    # the caller's positivity-and-finiteness check rejects.
    den = (f11 * f12 * f31 * f33 - f11 * f13 * f31 * f32
           + f12 * f12 * f32 * f33 - f12 * f13 * f32 * f32
           + f21 * f22 * f31 * f33 - f21 * f23 * f31 * f32
           + f22 * f22 * f32 * f33 - f22 * f23 * f32 * f32)
    num = -f33 * (f12 * f13 * f33 - f13 * f13 * f32
                  + f22 * f23 * f33 - f23 * f23 * f32)
    return num / den


@njit(cache=True)
def rybkin_focals_sq(F, pp1x, pp1y, pp2x, pp2y, out):
    # squared focal lengths of both cameras from a flat row-major fundamental
    # matrix and the principal points; out[0] = f1^2, out[1] = f2^2. First
    # conjugates F with the principal point translations (Fc = P2^T F P1, the
    # fundamental matrix of the centered coordinates), then applies Rybkin's
    # formula to Fc and Fc^T. Deliberately not fastmath: the finiteness and
    # positivity guards must be honest about the nan/inf a degenerate
    # denominator produces, because the caller feeds the result to math.sqrt
    # inside the RANSAC driver where an exception cannot propagate.
    c00 = F[0]
    c01 = F[1]
    c02 = F[0] * pp1x + F[1] * pp1y + F[2]
    c10 = F[3]
    c11 = F[4]
    c12 = F[3] * pp1x + F[4] * pp1y + F[5]
    c20 = pp2x * F[0] + pp2y * F[3] + F[6]
    c21 = pp2x * F[1] + pp2y * F[4] + F[7]
    c22 = pp2x * c02 + pp2y * c12 + F[6] * pp1x + F[7] * pp1y + F[8]

    f1_sq = _rybkin_f_sq(c00, c01, c02, c10, c11, c12, c20, c21, c22)
    f2_sq = _rybkin_f_sq(c00, c10, c20, c01, c11, c21, c02, c12, c22)
    out[0] = f1_sq
    out[1] = f2_sq
    return (f1_sq > 0.0 and f2_sq > 0.0
            and math.isfinite(f1_sq) and math.isfinite(f2_sq))


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
        if not rybkin_focals_sq(tmp, pp1x, pp1y, pp2x, pp2y, focals_sq):
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
