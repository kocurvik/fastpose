"""Full absolute pose estimator with unknown focal length: LO-RANSAC with
the P4Pf solver (poselib port), truncated pixel reprojection scorer and
direct pose + log-focal LM refiner."""

import numpy as np

from estimators.ransac import RansacEstimator
from refiners.absolute_focal import LMAbsolutePoseFocalRefiner
from scorers.reprojection import FocalReprojectionScorer
from solvers.p4pf import P4PFSolver

_default_estimator = None


def _get_default_estimator():
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = RansacEstimator(P4PFSolver(),
                                             FocalReprojectionScorer(),
                                             LMAbsolutePoseFocalRefiner())
    return _default_estimator


def estimate_absolute_pose_with_focal(x, X, principal_point=None,
                                      iterations=1000, max_error=2.0,
                                      seed=None, min_iterations=None,
                                      success_prob=0.9999, lo_iterations=None):
    # params:
    # x - (n, 2) array of image points in pixel coordinates (square pixels,
    #     unknown focal length)
    # X - (n, 3) array of corresponding 3D world points
    # principal_point - optional (cx, cy); if omitted, zero is assumed
    # max_error - truncated reprojection threshold in pixels
    # returns R, t, f, num_inliers, inliers with
    # lambda * (x, y, 1) = diag(f, f, 1) (R X + t)
    x = np.ascontiguousarray(x, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    if principal_point is not None:
        pp = np.asarray(principal_point, dtype=np.float64)
        if pp.shape != (2,):
            raise ValueError("principal_point must be a length-2 array")
        x = x - pp
    data = (np.ascontiguousarray(x[:, 0]), np.ascontiguousarray(x[:, 1]),
            np.ascontiguousarray(X[:, 0]), np.ascontiguousarray(X[:, 1]),
            np.ascontiguousarray(X[:, 2]))

    estimator = _get_default_estimator()
    model, score, num_inliers, _ = estimator.estimate(
        data, len(x), max_error, iterations=iterations,
        min_iterations=min_iterations, success_prob=success_prob,
        lo_iterations=lo_iterations, seed=seed)

    if num_inliers == 0:
        return None, None, None, 0, None

    R = model[:9].reshape(3, 3).copy()
    t = model[9:12].copy()
    f = float(model[12])
    _, inliers, num_inliers = FocalReprojectionScorer.score_numpy(
        R, t, f, x, X, max_error)
    return R, t, f, num_inliers, inliers
