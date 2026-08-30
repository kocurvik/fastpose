"""Relative pose with two unknown focal lengths from 7-point F hypotheses."""

import numpy as np

from fastpose.estimators.ransac import RansacEstimator
from fastpose.estimators.utils import (build_info, check_min_points, failure_info,
                                       normalize_points)
from fastpose.refiners.losses import CauchyLoss
from fastpose.refiners.utils import LO_INLIER_SCALE
from fastpose.refiners.varying_focal import LMVaryingFocalPoseRefiner
from fastpose.scorers.sampson import VaryingFocalPoseSampsonScorer
from fastpose.solvers.varying_focal import SevenPointVaryingFocalSolver

_default_estimator = None
_final_refiner = None


def _get_default_estimator():
    global _default_estimator
    if _default_estimator is None:
        # the local-optimization refit runs over the relaxed-threshold inlier
        # subset, matching poselib's
        # VaryingFocalRelativePoseEstimator::refine_model
        _default_estimator = RansacEstimator(
            SevenPointVaryingFocalSolver(),
            VaryingFocalPoseSampsonScorer(),
            LMVaryingFocalPoseRefiner(relaxed_inlier_scale=LO_INLIER_SCALE),
        )
    return _default_estimator


def _get_final_refiner():
    # loss for the final polish pass on RANSAC inliers only; see
    # refiners/losses.py for the available Loss objects
    global _final_refiner
    if _final_refiner is None:
        _final_refiner = LMVaryingFocalPoseRefiner(loss=CauchyLoss())
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


def estimate_relative_pose_with_varying_focals(
        x1, x2, principal_point1=None, principal_point2=None,
        iterations=1000, max_error=2.0, seed=4578, min_iterations=None,
        success_prob=0.9999, lo_iterations=25,
        final_refinement_iterations=100, num_threads=None,
        batch_per_thread=None):
    # x1, x2 are image coordinates in each camera's native pixel-like units.
    # principal_point* are optional (cx, cy); if omitted, zero is assumed.
    # final_refinement_iterations is the LM step budget for the final
    # Cauchy-loss polish pass on the RANSAC inliers, independent of
    # lo_iterations; defaults to 100, 0 disables the pass.
    # num_threads - >1 switches to the batched parallel RANSAC driver
    #     (see estimators/ransac.py): hypotheses are drawn in batches of
    #     num_threads * batch_per_thread and solved and scored across that
    #     many numba threads. None or 1 (default) keeps the serial driver.
    #     The parallel result is close to but not identical to the serial
    #     one, and it buys latency on a single call rather than throughput -
    #     leave it off when already running one process per core.
    # batch_per_thread - hypotheses per thread in a batch; None (default)
    #     uses ransac.DEFAULT_BATCH_PER_THREAD
    # returns (model, info) with model = {'R', 't', 'f1', 'f2'} and
    # info = {'inliers', 'num_inliers', 'model_score', 'iterations',
    # 'refinements'}; on total failure model holds the identity pose with
    # f1=f2=1.0 and info['num_inliers'] is 0.
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    check_min_points(len(x1), SevenPointVaryingFocalSolver.sample_size)
    pp1 = _principal_point(principal_point1)
    pp2 = _principal_point(principal_point2)

    x1n, x2n, T, scale = normalize_points(x1, x2)
    pp1n = np.array([scale * pp1[0] + T[0, 2],
                     scale * pp1[1] + T[1, 2]], dtype=np.float64)
    pp2n = np.array([scale * pp2[0] + T[0, 2],
                     scale * pp2[1] + T[1, 2]], dtype=np.float64)
    data = _data_tuple(x1n, x2n, pp1n, pp2n)

    estimator = _get_default_estimator()
    model, _, num_inliers, ransac_iterations = estimator.estimate(
        data, len(x1), max_error * scale, iterations=iterations,
        min_iterations=min_iterations, success_prob=success_prob,
        lo_iterations=lo_iterations, seed=seed,
        num_threads=num_threads, batch_per_thread=batch_per_thread)

    if num_inliers == 0:
        return ({'R': np.eye(3), 't': np.zeros(3), 'f1': 1.0, 'f2': 1.0},
                failure_info(len(x1), ransac_iterations))

    R = model[:9].reshape(3, 3).copy()
    t = model[9:12].copy()
    f1 = float(model[12] / scale)
    f2 = float(model[13] / scale)
    score, inliers, num_inliers = VaryingFocalPoseSampsonScorer.score_numpy(
        R, t, f1, f2, pp1, pp2, x1, x2, max_error)
    refined = False

    # final polish: robust-loss refinement restricted to the RANSAC inliers,
    # done in the same normalized frame/threshold as the RANSAC pipeline.
    # Fewer inliers than the minimal sample size cannot constrain the model,
    # so the pass is skipped there (poselib gates its bundle the same way)
    if (final_refinement_iterations != 0
            and num_inliers > SevenPointVaryingFocalSolver.sample_size):
        final_refiner = _get_final_refiner()
        inlier_data = _data_tuple(x1n[inliers], x2n[inliers], pp1n, pp2n)
        refined_model = np.empty(14)
        num_final_iterations = (final_refiner.num_iterations
                                if final_refinement_iterations is None
                                else final_refinement_iterations)
        if final_refiner.refine(inlier_data, model, refined_model,
                                (max_error * scale) ** 2,
                                num_final_iterations):
            R_c = refined_model[:9].reshape(3, 3).copy()
            t_c = refined_model[9:12].copy()
            f1_c = float(refined_model[12] / scale)
            f2_c = float(refined_model[13] / scale)
            score_c, inliers_c, num_inliers_c = VaryingFocalPoseSampsonScorer.score_numpy(
                R_c, t_c, f1_c, f2_c, pp1, pp2, x1, x2, max_error)
            R, t, f1, f2, score = R_c, t_c, f1_c, f2_c, score_c
            inliers, num_inliers = inliers_c, num_inliers_c
            refined = True

    return ({'R': R, 't': t, 'f1': f1, 'f2': f2},
            build_info(inliers, num_inliers, score, ransac_iterations, refined))
