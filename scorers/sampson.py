"""Truncated Sampson error (MSAC) scorer for epipolar geometry.

The same Sampson form works for fundamental and essential matrices;
`SampsonScorer` scores flat 3x3 models directly, `PoseSampsonScorer` scores
relative pose models [R | t] by assembling E = [t]_x R on the fly.
"""

import numpy as np
from numba import njit

from solvers.varying_focal import model_to_fundamental


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
