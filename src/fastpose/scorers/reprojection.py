"""Truncated squared reprojection error (MSAC) scorers for absolute pose.

`ReprojectionScorer` scores pose models `[R | t]` (12 flat parameters, R
row-major) against 2D-3D correspondences in calibrated image coordinates;
`FocalReprojectionScorer` scores `[R | t | f]` models (unknown focal) in
pixel coordinates relative to the principal point. Points behind the camera
are outliers in both.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def reprojection_score(model, data, max_error_sq, best_score):
    # truncated (MSAC) reprojection score + inlier count in one fused pass
    # with the same per-chunk early bail-out as the Sampson scorer; the
    # inlier test (zx - x*zz)^2 + (zy - y*zz)^2 < max_error_sq * zz^2 avoids
    # dividing for outliers
    x_x, x_y, X_x, X_y, X_z = data
    n = x_x.shape[0]
    r00 = model[0]
    r01 = model[1]
    r02 = model[2]
    r10 = model[3]
    r11 = model[4]
    r12 = model[5]
    r20 = model[6]
    r21 = model[7]
    r22 = model[8]
    t0 = model[9]
    t1 = model[10]
    t2 = model[11]

    chunk = 512
    score = 0.0
    num_inliers = 0
    start = 0
    while start < n:
        end = min(start + chunk, n)
        for i in range(start, end):
            Xx = X_x[i]
            Xy = X_y[i]
            Xz = X_z[i]
            zx = r00 * Xx + r01 * Xy + r02 * Xz + t0
            zy = r10 * Xx + r11 * Xy + r12 * Xz + t1
            zz = r20 * Xx + r21 * Xy + r22 * Xz + t2
            dx = zx - x_x[i] * zz
            dy = zy - x_y[i] * zz
            r2 = dx * dx + dy * dy
            zz_sq = zz * zz
            if zz > 0.0 and r2 < max_error_sq * zz_sq:
                score += r2 / zz_sq
                num_inliers += 1
            else:
                score += max_error_sq
        if score >= best_score:
            return score, num_inliers
        start = end
    return score, num_inliers


@njit(cache=True, fastmath=True)
def focal_reprojection_score(model, data, max_error_sq, best_score):
    # scorer for pose models [R | t | f]: the reprojection residual is
    # f * pi(R X + t) - x in (principal-point-centered) pixel coordinates
    f = model[12]
    if f <= 0.0:
        return 1e300, 0
    x_x, x_y, X_x, X_y, X_z = data
    n = x_x.shape[0]
    r00 = f * model[0]
    r01 = f * model[1]
    r02 = f * model[2]
    r10 = f * model[3]
    r11 = f * model[4]
    r12 = f * model[5]
    r20 = model[6]
    r21 = model[7]
    r22 = model[8]
    t0 = f * model[9]
    t1 = f * model[10]
    t2 = model[11]

    chunk = 512
    score = 0.0
    num_inliers = 0
    start = 0
    while start < n:
        end = min(start + chunk, n)
        for i in range(start, end):
            Xx = X_x[i]
            Xy = X_y[i]
            Xz = X_z[i]
            zx = r00 * Xx + r01 * Xy + r02 * Xz + t0
            zy = r10 * Xx + r11 * Xy + r12 * Xz + t1
            zz = r20 * Xx + r21 * Xy + r22 * Xz + t2
            dx = zx - x_x[i] * zz
            dy = zy - x_y[i] * zz
            r2 = dx * dx + dy * dy
            zz_sq = zz * zz
            if zz > 0.0 and r2 < max_error_sq * zz_sq:
                score += r2 / zz_sq
                num_inliers += 1
            else:
                score += max_error_sq
        if score >= best_score:
            return score, num_inliers
        start = end
    return score, num_inliers


class ReprojectionScorer():
    # truncated squared reprojection error (MSAC) scorer for absolute pose
    # models [R | t]; points behind the camera count as outliers
    score = staticmethod(reprojection_score)

    @staticmethod
    def score_numpy(R, t, x, X, max_error):
        # vectorized reference implementation; also used to extract the
        # final inlier mask
        Z = X @ R.T + t
        max_error_sq = max_error ** 2

        errors = np.full(len(x), np.inf)
        valid = Z[:, 2] > 0.0
        proj = Z[valid, :2] / Z[valid, 2:3]
        errors[valid] = np.sum((proj - x[valid]) ** 2, axis=1)

        inliers = errors <= max_error_sq
        np.minimum(errors, max_error_sq, out=errors)
        score = np.sum(errors)
        num_inliers = np.count_nonzero(inliers)

        return score, inliers, num_inliers


class FocalReprojectionScorer():
    # truncated squared reprojection error (MSAC) scorer for absolute pose
    # models [R | t | f] in principal-point-centered pixel coordinates
    score = staticmethod(focal_reprojection_score)

    @staticmethod
    def score_numpy(R, t, f, x, X, max_error):
        # f * pi(R X + t) - x is pi scaled: score in calibrated units
        return ReprojectionScorer.score_numpy(R, t, x / f, X, max_error / f)
