"""Full absolute pose estimator: LO-RANSAC with the P3P solver (Ding et
al.), truncated reprojection scorer and direct pose LM refiner."""

import numpy as np

from fastpose.estimators.ransac import RansacEstimator
from fastpose.refiners.absolute import LMAbsolutePoseRefiner
from fastpose.scorers.reprojection import ReprojectionScorer
from fastpose.solvers.p3p import P3PSolver

_default_estimator = None


def _get_default_estimator():
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = RansacEstimator(P3PSolver(), ReprojectionScorer(),
                                             LMAbsolutePoseRefiner())
    return _default_estimator


def estimate_absolute_pose(x, X, iterations=1000, max_error=0.002, seed=4578,
                           min_iterations=None, success_prob=0.9999,
                           lo_iterations=None):
    # params:
    # x - (n, 2) array of *calibrated* (normalized) image points; for pixel
    #     points apply (x - c) / f first
    # X - (n, 3) array of corresponding 3D world points
    # iterations - maximum number of RANSAC iterations
    # max_error - truncated reprojection threshold in the same normalized
    #             units (pixel threshold / focal)
    # min_iterations - minimum number of iterations before adaptive
    #                  termination may stop early; defaults to `iterations`
    #                  (fixed iteration count)
    # lo_iterations - LM step budget per local optimization; 0 disables
    #                 local optimization (plain RANSAC), None uses the
    #                 refiner default
    # returns R, t, num_inliers, inliers with lambda * (x, y, 1) = R X + t
    x = np.ascontiguousarray(x, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    data = (np.ascontiguousarray(x[:, 0]), np.ascontiguousarray(x[:, 1]),
            np.ascontiguousarray(X[:, 0]), np.ascontiguousarray(X[:, 1]),
            np.ascontiguousarray(X[:, 2]))

    estimator = _get_default_estimator()
    model, score, num_inliers, _ = estimator.estimate(
        data, len(x), max_error, iterations=iterations,
        min_iterations=min_iterations, success_prob=success_prob,
        lo_iterations=lo_iterations, seed=seed)

    if num_inliers == 0:
        return None, None, 0, None

    R = model[:9].reshape(3, 3).copy()
    t = model[9:12].copy()
    _, inliers, num_inliers = ReprojectionScorer.score_numpy(R, t, x, X,
                                                             max_error)
    return R, t, num_inliers, inliers
