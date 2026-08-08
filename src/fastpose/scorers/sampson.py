"""Truncated Sampson error (MSAC) scorer for epipolar geometry.

The same Sampson form works for fundamental and essential matrices;
`SampsonScorer` scores flat 3x3 models directly, `PoseSampsonScorer` scores
relative pose models [R | t] by assembling E = [t]_x R on the fly.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def sampson_score(f, data, max_error_sq, best_score):
    # truncated (MSAC) Sampson score + inlier count in one fused pass;
    # the truncated score only grows with each point, so once the partial
    # score exceeds the best score so far the model cannot win and scoring
    # can stop early (exact, loss-free bail-out). Checked per chunk to keep
    # the inner loop vectorizable.
    x1_x, x1_y, x2_x, x2_y = data
    n = x1_x.shape[0]
    chunk = 512
    score = 0.0
    num_inliers = 0
    start = 0
    while start < n:
        end = min(start + chunk, n)
        for i in range(start, end):
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
            denominator = fx1_0 * fx1_0 + fx1_1 * fx1_1 + ftx2_0 * ftx2_0 + ftx2_1 * ftx2_1
            r2 = residual * residual
            # r2 / denominator < max_error_sq, without dividing for outliers.
            # Degenerate models can make the Sampson denominator exactly zero;
            # keep those points in the outlier branch instead of dividing.
            if denominator > 0.0 and r2 < max_error_sq * denominator:
                score += r2 / denominator
                num_inliers += 1
            else:
                score += max_error_sq
        if score >= best_score:
            return score, num_inliers
        start = end
    return score, num_inliers


class SampsonScorer():
    # truncated Sampson error (MSAC) scorer for epipolar geometry; the same
    # scorer works for fundamental and essential matrix models
    score = staticmethod(sampson_score)

    @staticmethod
    def score_numpy(F, x1, x2, max_error):
        # vectorized reference implementation; also used to extract the
        # final inlier mask
        x1_x = x1[:, 0]
        x1_y = x1[:, 1]
        x2_x = x2[:, 0]
        x2_y = x2[:, 1]
        max_error_sq = max_error ** 2

        fx1_0 = F[0, 0] * x1_x + F[0, 1] * x1_y + F[0, 2]
        fx1_1 = F[1, 0] * x1_x + F[1, 1] * x1_y + F[1, 2]
        fx1_2 = F[2, 0] * x1_x + F[2, 1] * x1_y + F[2, 2]
        ftx2_0 = F[0, 0] * x2_x + F[1, 0] * x2_y + F[2, 0]
        ftx2_1 = F[0, 1] * x2_x + F[1, 1] * x2_y + F[2, 1]

        residual = x2_x * fx1_0 + x2_y * fx1_1 + fx1_2
        denominator = fx1_0 * fx1_0 + fx1_1 * fx1_1 + ftx2_0 * ftx2_0 + ftx2_1 * ftx2_1
        errors = np.full(len(x1), np.inf)
        valid = denominator > 0.0
        errors[valid] = residual[valid] * residual[valid] / denominator[valid]

        inliers = errors <= max_error_sq
        np.minimum(errors, max_error_sq, out=errors)
        score = np.sum(errors)
        num_inliers = np.count_nonzero(inliers)

        return score, inliers, num_inliers


@njit(cache=True, inline='always')
def essential_from_pose(pose, e):
    # e = flat E = [t]_x R for a pose model [R (row-major 3x3) | t (3)];
    # any t scale (the Sampson error is invariant to it)
    tx = pose[9]
    ty = pose[10]
    tz = pose[11]
    for j in range(3):
        r0 = pose[j]
        r1 = pose[3 + j]
        r2 = pose[6 + j]
        e[j] = -tz * r1 + ty * r2
        e[3 + j] = tz * r0 - tx * r2
        e[6 + j] = -ty * r0 + tx * r1


@njit(cache=True, fastmath=True)
def pose_sampson_score(model, data, max_error_sq, best_score):
    # scorer for pose models: assemble E = [t]_x R, then the shared
    # truncated Sampson score
    e = np.empty(9)
    essential_from_pose(model, e)
    return sampson_score(e, data, max_error_sq, best_score)


class PoseSampsonScorer():
    # truncated Sampson error (MSAC) scorer for relative pose models
    # [R | t] (12 flat parameters, R row-major)
    score = staticmethod(pose_sampson_score)

    @staticmethod
    def score_numpy(R, t, x1, x2, max_error):
        E = np.array([[0.0, -t[2], t[1]],
                      [t[2], 0.0, -t[0]],
                      [-t[1], t[0], 0.0]]) @ R
        return SampsonScorer.score_numpy(E, x1, x2, max_error)


@njit(cache=True)
def model_to_fundamental(model, pp1x, pp1y, pp2x, pp2y, f):
    # f = flat F = K2^-T E K1^-1 for a pose model [R | t | f1 | f2] with
    # K = [[f, 0, ppx], [0, f, ppy], [0, 0, 1]]; False for invalid focals
    f1 = model[12]
    f2 = model[13]
    if f1 <= 0.0 or f2 <= 0.0:
        return False
    e = np.empty(9)
    essential_from_pose(model, e)
    inv1 = 1.0 / f1
    inv2 = 1.0 / f2
    # rows of A = K2^-T E, then columns of F = A K1^-1
    a = np.empty(9)
    for j in range(3):
        a[j] = inv2 * e[j]
        a[3 + j] = inv2 * e[3 + j]
        a[6 + j] = e[6 + j] - inv2 * (pp2x * e[j] + pp2y * e[3 + j])
    for i in range(3):
        f[3 * i] = inv1 * a[3 * i]
        f[3 * i + 1] = inv1 * a[3 * i + 1]
        f[3 * i + 2] = (a[3 * i + 2]
                        - inv1 * (pp1x * a[3 * i] + pp1y * a[3 * i + 1]))
    return True


@njit(cache=True, fastmath=True)
def varying_focal_pose_sampson_score(model, data, max_error_sq, best_score):
    x1_x, x1_y, x2_x, x2_y, pp1x, pp1y, pp2x, pp2y = data
    f = np.empty(9)
    if not model_to_fundamental(model, pp1x, pp1y, pp2x, pp2y, f):
        return 1e300, 0
    return sampson_score(f, (x1_x, x1_y, x2_x, x2_y), max_error_sq, best_score)


class VaryingFocalPoseSampsonScorer():
    # truncated Sampson error for pose models [R | t | f1 | f2], evaluated
    # in the original image coordinate system with fixed principal points.
    score = staticmethod(varying_focal_pose_sampson_score)

    @staticmethod
    def score_numpy(R, t, f1, f2, pp1, pp2, x1, x2, max_error):
        E = np.array([[0.0, -t[2], t[1]],
                      [t[2], 0.0, -t[0]],
                      [-t[1], t[0], 0.0]]) @ R
        K1i = np.array([[1.0 / f1, 0.0, -pp1[0] / f1],
                        [0.0, 1.0 / f1, -pp1[1] / f1],
                        [0.0, 0.0, 1.0]])
        K2i = np.array([[1.0 / f2, 0.0, -pp2[0] / f2],
                        [0.0, 1.0 / f2, -pp2[1] / f2],
                        [0.0, 0.0, 1.0]])
        F = K2i.T @ E @ K1i
        return SampsonScorer.score_numpy(F, x1, x2, max_error)


@njit(cache=True, fastmath=True)
def shared_focal_pose_sampson_score(model, data, max_error_sq, best_score):
    # Same model layout as varying-focal, but the shared-focal solver/refiner
    # keeps model[12] == model[13].
    return varying_focal_pose_sampson_score(model, data, max_error_sq,
                                            best_score)


@njit(cache=True, fastmath=True)
def monodepth_pose_sampson_score(model, data, max_error_sq, best_score):
    # scorer for calibrated monodepth models [R | t | scale | shift1 |
    # shift2]: the truncated Sampson error of E = [t]_x R (the depth
    # parameters do not enter the scoring, matching poselib's monodepth
    # estimators which score with the plain Sampson MSAC error)
    x1_x, x1_y, x2_x, x2_y = data[0], data[1], data[2], data[3]
    e = np.empty(9)
    essential_from_pose(model, e)
    return sampson_score(e, (x1_x, x1_y, x2_x, x2_y), max_error_sq, best_score)


class MonoDepthPoseSampsonScorer():
    # truncated Sampson error for calibrated monodepth models
    # [R | t | scale | shift1 | shift2]
    score = staticmethod(monodepth_pose_sampson_score)

    @staticmethod
    def score_numpy(R, t, x1, x2, max_error):
        return PoseSampsonScorer.score_numpy(R, t, x1, x2, max_error)


@njit(cache=True, fastmath=True)
def monodepth_focal_pose_sampson_score(model, data, max_error_sq, best_score):
    # scorer for monodepth focal models [R | t | f1 | f2 | scale] in
    # centered pixel coordinates: truncated Sampson error of the induced
    # F = K2^-T E K1^-1 (principal points already subtracted)
    x1_x, x1_y, x2_x, x2_y = data[0], data[1], data[2], data[3]
    f = np.empty(9)
    if not model_to_fundamental(model, 0.0, 0.0, 0.0, 0.0, f):
        return 1e300, 0
    return sampson_score(f, (x1_x, x1_y, x2_x, x2_y), max_error_sq, best_score)


class MonoDepthFocalPoseSampsonScorer():
    # truncated Sampson error for monodepth focal models
    # [R | t | f1 | f2 | scale] in centered pixel coordinates
    score = staticmethod(monodepth_focal_pose_sampson_score)

    @staticmethod
    def score_numpy(R, t, f1, f2, x1, x2, max_error):
        zero = np.zeros(2)
        return VaryingFocalPoseSampsonScorer.score_numpy(
            R, t, f1, f2, zero, zero, x1, x2, max_error)


class SharedFocalPoseSampsonScorer():
    # truncated Sampson error for pose models [R | t | f | f], evaluated in
    # image coordinates with fixed principal points.
    score = staticmethod(shared_focal_pose_sampson_score)

    @staticmethod
    def score_numpy(R, t, f, pp1, pp2, x1, x2, max_error):
        return VaryingFocalPoseSampsonScorer.score_numpy(
            R, t, f, f, pp1, pp2, x1, x2, max_error)


# ---------------------------------------------------------------------------
# loss-selectable cost kernels for the final refinement pass (LM accept/
# reject only; RANSAC model selection above always stays truncated MSAC via
# the *_score functions, so their early per-chunk bail-out stays valid).
# ---------------------------------------------------------------------------

def build_sampson_cost(loss):
    # generalizes sampson_score's truncated capped-quadratic into an
    # arbitrary loss.cost(r2, max_error_sq); always divides (no need for the
    # truncated fast path's skip-the-division trick since this is only used
    # inside the LM loop, not the O(ransac_iterations x n) scorer)
    cost_fn = loss.cost

    @njit(cache=True, fastmath=True)
    def _sampson_cost(f, data, max_error_sq, best_score):
        x1_x, x1_y, x2_x, x2_y = data
        n = x1_x.shape[0]
        score = 0.0
        num_inliers = 0
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
            if denominator > 0.0:
                r2 = residual * residual / denominator
                score += cost_fn(r2, max_error_sq)
                if r2 < max_error_sq:
                    num_inliers += 1
            else:
                score += cost_fn(1e18, max_error_sq)
        return score, num_inliers

    return _sampson_cost


def build_pose_sampson_cost(loss):
    sampson_cost = build_sampson_cost(loss)

    @njit(cache=True, fastmath=True)
    def _pose_sampson_cost(model, data, max_error_sq, best_score):
        e = np.empty(9)
        essential_from_pose(model, e)
        return sampson_cost(e, data, max_error_sq, best_score)

    return _pose_sampson_cost


def build_varying_focal_pose_sampson_cost(loss):
    # shared by the shared-focal refiner too (same model layout, mirroring
    # shared_focal_pose_sampson_score reusing varying_focal_pose_sampson_score)
    sampson_cost = build_sampson_cost(loss)

    @njit(cache=True, fastmath=True)
    def _varying_focal_pose_sampson_cost(model, data, max_error_sq, best_score):
        x1_x, x1_y, x2_x, x2_y, pp1x, pp1y, pp2x, pp2y = data
        f = np.empty(9)
        if not model_to_fundamental(model, pp1x, pp1y, pp2x, pp2y, f):
            return 1e300, 0
        return sampson_cost(f, (x1_x, x1_y, x2_x, x2_y), max_error_sq, best_score)

    return _varying_focal_pose_sampson_cost
