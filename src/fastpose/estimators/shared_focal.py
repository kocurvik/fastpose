"""Relative pose with one unknown focal length shared by both cameras."""

import numpy as np

from fastpose.estimators.ransac import RansacEstimator
from fastpose.estimators.utils import normalize_points
from fastpose.refiners.losses import CauchyLoss
from fastpose.refiners.shared_focal import LMSharedFocalPoseRefiner
from fastpose.scorers.sampson import SharedFocalPoseSampsonScorer
from fastpose.solvers.shared_focal import SixPointSharedFocalSolver

_default_estimator = None
_final_refiner = None


def _get_default_estimator():
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = RansacEstimator(
            SixPointSharedFocalSolver(),
            SharedFocalPoseSampsonScorer(),
            LMSharedFocalPoseRefiner(),
        )
    return _default_estimator


def _get_final_refiner():
    # loss for the final polish pass on RANSAC inliers only; see
    # refiners/losses.py for the available Loss objects
    global _final_refiner
    if _final_refiner is None:
        _final_refiner = LMSharedFocalPoseRefiner(loss=CauchyLoss())
    return _final_refiner


def _principal_point(pp):
    if pp is None:
        return np.zeros(2, dtype=np.float64)
    arr = np.asarray(pp, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError("principal points must be length-2 arrays")
    return arr


def _data_tuple(x1, x2, pp1, pp2):
    return (
        np.ascontiguousarray(x1[:, 0]),
        np.ascontiguousarray(x1[:, 1]),
        np.ascontiguousarray(x2[:, 0]),
        np.ascontiguousarray(x2[:, 1]),
        float(pp1[0]),
        float(pp1[1]),
        float(pp2[0]),
        float(pp2[1]),
    )


def estimate_relative_pose_with_shared_focal(
        x1, x2, principal_point1=None, principal_point2=None,
        iterations=1000, max_error=2.0, seed=4578, min_iterations=None,
        success_prob=0.9999, lo_iterations=25,
        final_refinement_iterations=100):
    # x1, x2 are image coordinates for two cameras with the same unknown
    # square-pixel focal length. principal_point* are optional (cx, cy); if
    # omitted, zero is assumed. final_refinement_iterations is the LM step
    # budget for the final Cauchy-loss polish pass on the RANSAC inliers,
    # independent of lo_iterations; defaults to 100, 0 disables the pass.
    # Returns R, t, f, num_inliers, inliers.
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    pp1 = _principal_point(principal_point1)
    pp2 = _principal_point(principal_point2)

    x1n, x2n, T, scale = normalize_points(x1, x2)
    pp1n = np.array([scale * pp1[0] + T[0, 2],
                     scale * pp1[1] + T[1, 2]], dtype=np.float64)
    pp2n = np.array([scale * pp2[0] + T[0, 2],
                     scale * pp2[1] + T[1, 2]], dtype=np.float64)
    data = _data_tuple(x1n, x2n, pp1n, pp2n)

    estimator = _get_default_estimator()
    model, score, num_inliers, _ = estimator.estimate(
        data, len(x1), max_error * scale, iterations=iterations,
        min_iterations=min_iterations, success_prob=success_prob,
        lo_iterations=lo_iterations, seed=seed)

    if num_inliers == 0:
        return None, None, None, 0, None

    R = model[:9].reshape(3, 3).copy()
    t = model[9:12].copy()
    f = float(model[12] / scale)
    _, inliers, num_inliers = SharedFocalPoseSampsonScorer.score_numpy(
        R, t, f, pp1, pp2, x1, x2, max_error)

    # final polish: robust-loss refinement restricted to the RANSAC inliers,
    # done in the same normalized frame/threshold as the RANSAC pipeline
    if final_refinement_iterations != 0 and num_inliers > 0:
        final_refiner = _get_final_refiner()
        inlier_data = _data_tuple(x1n[inliers], x2n[inliers], pp1n, pp2n)
        refined = np.empty(14)
        num_final_iterations = (final_refiner.num_iterations
                                if final_refinement_iterations is None
                                else final_refinement_iterations)
        if final_refiner.refine(inlier_data, model, refined,
                                (max_error * scale) ** 2,
                                num_final_iterations):
            R_c = refined[:9].reshape(3, 3).copy()
            t_c = refined[9:12].copy()
            f_c = float(refined[12] / scale)
            _, inliers_c, num_inliers_c = SharedFocalPoseSampsonScorer.score_numpy(
                R_c, t_c, f_c, pp1, pp2, x1, x2, max_error)
            R, t, f, inliers, num_inliers = R_c, t_c, f_c, inliers_c, num_inliers_c

    return R, t, f, num_inliers, inliers
