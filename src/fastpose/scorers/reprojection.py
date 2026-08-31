"""Truncated squared reprojection error (MSAC) scorers for absolute pose.

`ReprojectionScorer` scores pose models `[R | t]` (12 flat parameters, R
row-major) against 2D-3D correspondences in calibrated image coordinates;
`FocalReprojectionScorer` scores `[R | t | f]` models (unknown focal) in
pixel coordinates relative to the principal point. Points behind the camera
are outliers in both.

The focal case is not a second scorer: `focal_scale_pose` folds `f` into the
pose once per model (rows 0 and 1 of R and the first two components of t
scaled by `f`), after which the residual is the plain calibrated one. That is
the same trick `focal_reprojection_score` used inline before, hoisted into a
kernel so the CUDA scorer can reuse it.
"""

import numpy as np
from numba import float64, njit

from fastpose.jit_backend import cpu_jit
from fastpose.scorers.sampson import SCORE_CHUNK


# ---------------------------------------------------------------------------
# per-point primitives, shared by the CPU scorers below and by the CUDA
# reduction scorer in fastpose/cuda/problems/absolute*.py
#
# Same arrangement as `build_sampson_point_kernels`: these are the only place
# the reprojection residual and the focal folding are written down, so the GPU
# block reduction - which cannot reuse the blocked loops below - still agrees
# with the CPU scorer point for point. See that factory for why `real` exists.
# ---------------------------------------------------------------------------

def build_reprojection_point_kernels(jit, real=float64):
    @jit(fastmath=True, inline=True)
    def reprojection_residual(m, Xx, Xy, Xz, x, y):
        # squared reprojection numerator and the depth zz of one 2D-3D
        # correspondence under the flat pose m = [R | t]. Returned unreduced
        # so callers can apply the inlier test (r2 < max_error_sq * zz^2)
        # without dividing for outliers; zz <= 0 is a point behind the camera
        # and is the caller's cue to take the outlier branch.
        zx = m[0] * Xx + m[1] * Xy + m[2] * Xz + m[9]
        zy = m[3] * Xx + m[4] * Xy + m[5] * Xz + m[10]
        zz = m[6] * Xx + m[7] * Xy + m[8] * Xz + m[11]
        dx = zx - x * zz
        dy = zy - y * zz
        return dx * dx + dy * dy, zz

    @jit(inline=True)
    def focal_scale_pose(model, m):
        # [R | t | f] -> the 12-vector whose calibrated reprojection residual
        # is the pixel one, f * pi(R X + t) - x. False for an invalid focal.
        f = model[12]
        if f <= real(0.0):
            return False
        for j in range(3):
            m[j] = f * model[j]
            m[3 + j] = f * model[3 + j]
            m[6 + j] = model[6 + j]
        m[9] = f * model[9]
        m[10] = f * model[10]
        m[11] = model[11]
        return True

    return {
        'reprojection_residual': reprojection_residual,
        'focal_scale_pose': focal_scale_pose,
    }


_CPU_POINT = build_reprojection_point_kernels(cpu_jit)
reprojection_residual = _CPU_POINT['reprojection_residual']
focal_scale_pose = _CPU_POINT['focal_scale_pose']


@njit(cache=True, fastmath=True, inline='always')
def _reprojection_block(m, x_x, x_y, X_x, X_y, X_z, start, count,
                        max_error_sq):
    # truncated (MSAC) reprojection score + inlier count over count points
    # starting at start. Called with a literal count for every full block so
    # the loop vectorizes; see SCORE_CHUNK in scorers/sampson.py
    score = 0.0
    num_inliers = 0
    for j in range(count):
        i = start + j
        r2, zz = reprojection_residual(m, X_x[i], X_y[i], X_z[i], x_x[i],
                                       x_y[i])
        zz_sq = zz * zz
        if zz > 0.0 and r2 < max_error_sq * zz_sq:
            score += r2 / zz_sq
            num_inliers += 1
        else:
            score += max_error_sq
    return score, num_inliers


@njit(cache=True, fastmath=True, inline='always')
def _reprojection_blocked(m, data, max_error_sq, best_score):
    # blocked scan with the same early bail-out as the Sampson scorer: the
    # truncated score only grows, so once the partial score exceeds the best
    # score so far the model cannot win and scoring can stop
    x_x, x_y, X_x, X_y, X_z = data
    n = x_x.shape[0]
    score = 0.0
    num_inliers = 0
    num_full = n // SCORE_CHUNK
    for c in range(num_full):
        block_score, block_inliers = _reprojection_block(
            m, x_x, x_y, X_x, X_y, X_z, c * SCORE_CHUNK, SCORE_CHUNK,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
        if score >= best_score:
            return score, num_inliers
    tail = n - num_full * SCORE_CHUNK
    if tail > 0:
        block_score, block_inliers = _reprojection_block(
            m, x_x, x_y, X_x, X_y, X_z, num_full * SCORE_CHUNK, tail,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
    return score, num_inliers


@njit(cache=True, fastmath=True)
def reprojection_score(model, data, max_error_sq, best_score):
    # truncated (MSAC) reprojection score + inlier count in one fused pass;
    # the inlier test (zx - x*zz)^2 + (zy - y*zz)^2 < max_error_sq * zz^2
    # avoids dividing for outliers
    return _reprojection_blocked(model, data, max_error_sq, best_score)


@njit(cache=True, fastmath=True)
def focal_reprojection_score(model, data, max_error_sq, best_score):
    # scorer for pose models [R | t | f]: the reprojection residual is
    # f * pi(R X + t) - x in (principal-point-centered) pixel coordinates
    m = np.empty(12)
    if not focal_scale_pose(model, m):
        return 1e300, 0
    return _reprojection_blocked(m, data, max_error_sq, best_score)


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

    @njit(cache=True, fastmath=True, inline='always')
    def _cost_scan(m, data, max_error_sq):
        x_x, x_y, X_x, X_y, X_z = data
        n = x_x.shape[0]
        score = 0.0
        num_inliers = 0
        for i in range(n):
            r2n, zz = reprojection_residual(m, X_x[i], X_y[i], X_z[i], x_x[i],
                                            x_y[i])
            if zz > 0.0:
                r2 = r2n / (zz * zz)
                score += cost_fn(r2, max_error_sq)
                if r2 < max_error_sq:
                    num_inliers += 1
            else:
                score += cost_fn(1e18, max_error_sq)
        return score, num_inliers

    @njit(cache=True, fastmath=True)
    def _reprojection_cost(model, data, max_error_sq, best_score):
        return _cost_scan(model, data, max_error_sq)

    return _reprojection_cost


def build_focal_reprojection_cost(loss):
    cost_fn = loss.cost

    @njit(cache=True, fastmath=True)
    def _focal_reprojection_cost(model, data, max_error_sq, best_score):
        m = np.empty(12)
        if not focal_scale_pose(model, m):
            return 1e300, 0
        x_x, x_y, X_x, X_y, X_z = data
        n = x_x.shape[0]
        score = 0.0
        num_inliers = 0
        for i in range(n):
            r2n, zz = reprojection_residual(m, X_x[i], X_y[i], X_z[i], x_x[i],
                                            x_y[i])
            if zz > 0.0:
                r2 = r2n / (zz * zz)
                score += cost_fn(r2, max_error_sq)
                if r2 < max_error_sq:
                    num_inliers += 1
            else:
                score += cost_fn(1e18, max_error_sq)
        return score, num_inliers

    return _focal_reprojection_cost
