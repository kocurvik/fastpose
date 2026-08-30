"""Truncated squared reprojection error (MSAC) scorers for absolute pose.

`ReprojectionScorer` scores pose models `[R | t]` (12 flat parameters, R
row-major) against 2D-3D correspondences in calibrated image coordinates;
`FocalReprojectionScorer` scores `[R | t | f]` models (unknown focal) in
pixel coordinates relative to the principal point. Points behind the camera
are outliers in both.
"""

import numpy as np
from numba import njit

from fastpose.scorers.sampson import SCORE_CHUNK


@njit(cache=True, fastmath=True, inline='always')
def _reprojection_block(x_x, x_y, X_x, X_y, X_z, start, count,
                        r00, r01, r02, r10, r11, r12, r20, r21, r22,
                        t0, t1, t2, max_error_sq):
    # truncated (MSAC) reprojection score + inlier count over count points
    # starting at start. Called with a literal count for every full block so
    # the loop vectorizes; see SCORE_CHUNK in scorers/sampson.py
    score = 0.0
    num_inliers = 0
    for j in range(count):
        i = start + j
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
    return score, num_inliers


@njit(cache=True, fastmath=True)
def reprojection_score(model, data, max_error_sq, best_score):
    # truncated (MSAC) reprojection score + inlier count in one fused pass
    # with the same blocked early bail-out as the Sampson scorer; the
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

    score = 0.0
    num_inliers = 0
    num_full = n // SCORE_CHUNK
    for c in range(num_full):
        block_score, block_inliers = _reprojection_block(
            x_x, x_y, X_x, X_y, X_z, c * SCORE_CHUNK, SCORE_CHUNK,
            r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
        if score >= best_score:
            return score, num_inliers
    tail = n - num_full * SCORE_CHUNK
    if tail > 0:
        block_score, block_inliers = _reprojection_block(
            x_x, x_y, X_x, X_y, X_z, num_full * SCORE_CHUNK, tail,
            r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
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

    score = 0.0
    num_inliers = 0
    num_full = n // SCORE_CHUNK
    for c in range(num_full):
        block_score, block_inliers = _reprojection_block(
            x_x, x_y, X_x, X_y, X_z, c * SCORE_CHUNK, SCORE_CHUNK,
            r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
        if score >= best_score:
            return score, num_inliers
    tail = n - num_full * SCORE_CHUNK
    if tail > 0:
        block_score, block_inliers = _reprojection_block(
            x_x, x_y, X_x, X_y, X_z, num_full * SCORE_CHUNK, tail,
            r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
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


# ---------------------------------------------------------------------------
# loss-selectable cost kernels for the final refinement pass (LM accept/
# reject only; RANSAC model selection above always stays truncated MSAC via
# the *_score functions, so their early per-block bail-out stays valid).
# ---------------------------------------------------------------------------

def build_reprojection_cost(loss):
    # generalizes reprojection_score's truncated capped-quadratic into an
    # arbitrary loss.cost(r2, max_error_sq); always divides by zz_sq (no
    # need for the truncated fast path's skip-the-division trick since this
    # is only used inside the LM loop, not the O(ransac_iterations x n) scorer)
    cost_fn = loss.cost

    @njit(cache=True, fastmath=True)
    def _reprojection_cost(model, data, max_error_sq, best_score):
        x_x, x_y, X_x, X_y, X_z = data
        n = x_x.shape[0]
        r00, r01, r02 = model[0], model[1], model[2]
        r10, r11, r12 = model[3], model[4], model[5]
        r20, r21, r22 = model[6], model[7], model[8]
        t0, t1, t2 = model[9], model[10], model[11]

        score = 0.0
        num_inliers = 0
        for i in range(n):
            Xx = X_x[i]
            Xy = X_y[i]
            Xz = X_z[i]
            zx = r00 * Xx + r01 * Xy + r02 * Xz + t0
            zy = r10 * Xx + r11 * Xy + r12 * Xz + t1
            zz = r20 * Xx + r21 * Xy + r22 * Xz + t2
            if zz > 0.0:
                dx = zx - x_x[i] * zz
                dy = zy - x_y[i] * zz
                r2 = (dx * dx + dy * dy) / (zz * zz)
                score += cost_fn(r2, max_error_sq)
                if r2 < max_error_sq:
                    num_inliers += 1
            else:
                score += cost_fn(1e18, max_error_sq)
        return score, num_inliers

    return _reprojection_cost


def build_focal_reprojection_cost(loss):
    cost_fn = loss.cost

    @njit(cache=True, fastmath=True)
    def _focal_reprojection_cost(model, data, max_error_sq, best_score):
        f = model[12]
        if f <= 0.0:
            return 1e300, 0
        x_x, x_y, X_x, X_y, X_z = data
        n = x_x.shape[0]
        r00, r01, r02 = f * model[0], f * model[1], f * model[2]
        r10, r11, r12 = f * model[3], f * model[4], f * model[5]
        r20, r21, r22 = model[6], model[7], model[8]
        t0, t1, t2 = f * model[9], f * model[10], model[11]

        score = 0.0
        num_inliers = 0
        for i in range(n):
            Xx = X_x[i]
            Xy = X_y[i]
            Xz = X_z[i]
            zx = r00 * Xx + r01 * Xy + r02 * Xz + t0
            zy = r10 * Xx + r11 * Xy + r12 * Xz + t1
            zz = r20 * Xx + r21 * Xy + r22 * Xz + t2
            if zz > 0.0:
                dx = zx - x_x[i] * zz
                dy = zy - x_y[i] * zz
                r2 = (dx * dx + dy * dy) / (zz * zz)
                score += cost_fn(r2, max_error_sq)
                if r2 < max_error_sq:
                    num_inliers += 1
            else:
                score += cost_fn(1e18, max_error_sq)
        return score, num_inliers

    return _focal_reprojection_cost
