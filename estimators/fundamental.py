"""Full fundamental matrix estimator: LO-RANSAC with the 7-point solver,
Sampson scorer and factorized LM refiner, plus Hartley-style normalization.
A pure-numpy reference implementation (`estimate_fundamental`) is kept for
benchmarking.
"""

import numpy as np

from estimators.ransac import RansacEstimator
from estimators.utils import normalize_points, point_columns
from refiners.fundamental import LMFundamentalRefiner
from scorers.sampson import SampsonScorer
from solvers.fundamental import SevenPointSolver, seven_point


def estimate_fundamental(x1, x2, iterations=1000, max_error=2.0):
    # pure-numpy reference RANSAC (no local optimization)
    x1n, x2n, T, scale = normalize_points(x1, x2)
    threshold = max_error * scale
    best_score = np.inf
    best_model = None
    for _ in range(iterations):
        idxs = np.random.choice(len(x1n), 7, replace=False)
        Fs = seven_point(x1n[idxs], x2n[idxs])
        for F in Fs:
            score, inliers, num_inliers = SampsonScorer.score_numpy(F, x1n, x2n, threshold)

            if score < best_score:
                best_score = score
                best_model = F

    if best_model is None:
        return None, 0, None

    F = T.T @ best_model @ T
    _, inliers, num_inliers = SampsonScorer.score_numpy(F, x1, x2, max_error)
    return F, num_inliers, inliers


_default_estimator = None


def _get_default_estimator():
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = RansacEstimator(SevenPointSolver(), SampsonScorer(),
                                             LMFundamentalRefiner())
    return _default_estimator


def estimate_fundamental_numba(x1, x2, iterations=1000, max_error=2.0, seed=None,
                               min_iterations=None, success_prob=0.9999,
                               lo_iterations=None):
    # params:
    # x1, x2 - (n, 2) arrays of corresponding points
    # iterations - maximum number of RANSAC iterations
    # min_iterations - minimum number of iterations before adaptive
    #                  termination may stop early; defaults to `iterations`
    #                  (fixed iteration count)
    # lo_iterations - LM step budget per local optimization; 0 disables
    #                 local optimization (plain RANSAC), None uses the
    #                 refiner default
    # returns best_model, best_num_inliers, best_inliers
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    x1n, x2n, T, scale = normalize_points(x1, x2)
    data = point_columns(x1n, x2n)

    estimator = _get_default_estimator()
    model, score, num_inliers, _ = estimator.estimate(
        data, len(x1), max_error * scale, iterations=iterations,
        min_iterations=min_iterations, success_prob=success_prob,
        lo_iterations=lo_iterations, seed=seed)

    if num_inliers == 0:
        return None, 0, None

    F = T.T @ model.reshape(3, 3) @ T
    _, inliers, num_inliers = SampsonScorer.score_numpy(F, x1, x2, max_error)
    return F, num_inliers, inliers
