"""Full calibrated relative pose estimator: LO-RANSAC with the 5-point
solver that outputs poses [R | t] directly (cheirality-checked on the
minimal sample), the pose Sampson scorer (E = [t]_x R assembled on the fly)
and the direct pose LM refiner. `motion_from_essential` is kept for
decomposing externally estimated E/F matrices (e.g. the fundamental matrix
benchmark).
"""

import numpy as np

from fastpose.estimators.ransac import RansacEstimator
from fastpose.estimators.utils import (build_info, check_min_points, failure_info,
                                       point_columns, unproject_pair)
from fastpose.refiners.essential import LMEssentialRefiner
from fastpose.refiners.losses import CauchyLoss
from fastpose.refiners.utils import LO_INLIER_SCALE
from fastpose.scorers.sampson import PoseSampsonScorer
from fastpose.solvers.essential import FivePointSolver

_default_estimator = None
_final_refiner = None


def _get_default_estimator():
    global _default_estimator
    if _default_estimator is None:
        # the local-optimization refit runs over the relaxed-threshold inlier
        # subset, matching poselib's RelativePoseEstimator::refine_model
        _default_estimator = RansacEstimator(
            FivePointSolver(), PoseSampsonScorer(),
            LMEssentialRefiner(relaxed_inlier_scale=LO_INLIER_SCALE))
    return _default_estimator


def _get_final_refiner():
    # loss for the final polish pass on RANSAC inliers only; see
    # refiners/losses.py for the available Loss objects
    global _final_refiner
    if _final_refiner is None:
        _final_refiner = LMEssentialRefiner(loss=CauchyLoss())
    return _final_refiner


def estimate_relative_pose(x1, x2, camera1=None, camera2=None, iterations=1000,
                           max_error=0.002, seed=4578, min_iterations=None,
                           success_prob=0.9999, lo_iterations=25,
                           final_refinement_iterations=100,
                           num_threads=None, batch_per_thread=None):
    # params:
    # x1, x2 - (n, 2) arrays of corresponding *calibrated* (normalized) image
    #          points; for pixel points apply (x - c) / f first, or pass
    #          camera1/camera2 to have this done automatically
    # camera1, camera2 - optional cameras (3x3 intrinsic matrix K, or a
    #          poselib.Camera) used to unproject x1/x2 from pixel
    #          coordinates into calibrated points; None (default) keeps x1,
    #          x2 as already-calibrated input. When supplied, max_error is
    #          also rescaled by the average focal length of the two cameras
    # max_error - Sampson threshold in the same normalized units
    #             (pixel threshold divided by focal length)
    # final_refinement_iterations - LM step budget for the final Cauchy-loss
    #                 polish pass on the RANSAC inliers; independent of
    #                 lo_iterations. Defaults to 100; 0 disables the pass
    # num_threads - >1 switches to the batched parallel RANSAC driver
    #     (see estimators/ransac.py): hypotheses are drawn in batches of
    #     num_threads * batch_per_thread and solved and scored across that
    #     many numba threads. None or 1 (default) keeps the serial driver.
    #     The parallel result is close to but not identical to the serial
    #     one, and it buys latency on a single call rather than throughput -
    #     leave it off when already running one process per core.
    # batch_per_thread - hypotheses per thread in a batch; None (default)
    #     uses ransac.DEFAULT_BATCH_PER_THREAD
    # returns (model, info) with model = {'R', 't'} (unit norm t,
    # x2 ~ R x1 + t) and info = {'inliers', 'num_inliers', 'model_score',
    # 'iterations', 'refinements'}; on total failure model holds the
    # identity pose and info['num_inliers'] is 0
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    check_min_points(len(x1), FivePointSolver.sample_size)
    x1, x2, max_error = unproject_pair(camera1, camera2, x1, x2, max_error)
    data = point_columns(x1, x2)

    estimator = _get_default_estimator()
    model, _, num_inliers, ransac_iterations = estimator.estimate(
        data, len(x1), max_error, iterations=iterations,
        min_iterations=min_iterations, success_prob=success_prob,
        lo_iterations=lo_iterations, seed=seed,
        num_threads=num_threads, batch_per_thread=batch_per_thread)

    if num_inliers == 0:
        return ({'R': np.eye(3), 't': np.zeros(3)},
                failure_info(len(x1), ransac_iterations))

    R = model[:9].reshape(3, 3).copy()
    t = model[9:12].copy()
    score, inliers, num_inliers = PoseSampsonScorer.score_numpy(R, t, x1, x2, max_error)
    refined = False

    # final polish: robust-loss refinement restricted to the RANSAC inliers.
    # Fewer inliers than the minimal sample size cannot constrain the model,
    # so the pass is skipped there (poselib gates its bundle the same way)
    if (final_refinement_iterations != 0
            and num_inliers > FivePointSolver.sample_size):
        final_refiner = _get_final_refiner()
        inlier_data = point_columns(x1[inliers], x2[inliers])
        refined_model = np.empty(12)
        num_final_iterations = (final_refiner.num_iterations
                                if final_refinement_iterations is None
                                else final_refinement_iterations)
        if final_refiner.refine(inlier_data, model, refined_model, max_error ** 2,
                                num_final_iterations):
            R_c = refined_model[:9].reshape(3, 3).copy()
            t_c = refined_model[9:12].copy()
            score_c, inliers_c, num_inliers_c = PoseSampsonScorer.score_numpy(
                R_c, t_c, x1, x2, max_error)
            R, t, inliers, num_inliers, score = (R_c, t_c, inliers_c,
                                                 num_inliers_c, score_c)
            refined = True

    return ({'R': R, 't': t},
            build_info(inliers, num_inliers, score, ransac_iterations, refined))


def motion_from_essential(E, x1, x2, max_points=100):
    # decompose E into (R, t) with x2 ~ R x1 + t, choosing among the four
    # candidates by cheirality on up to max_points calibrated correspondences
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    x1h = np.column_stack([x1[:max_points], np.ones(min(len(x1), max_points))])
    x2h = np.column_stack([x2[:max_points], np.ones(min(len(x2), max_points))])

    best_count = -1
    best_R = None
    best_t = None
    for R in (U @ W @ Vt, U @ W.T @ Vt):
        rx1 = x1h @ R.T
        for t in (U[:, 2], -U[:, 2]):
            # depth in camera 1 from x2 x (z1 * R x1 + t) = 0
            c = np.cross(x2h, rx1)
            d = np.cross(x2h, np.broadcast_to(t, x2h.shape))
            z1 = -np.sum(c * d, axis=1) / np.sum(c * c, axis=1)
            z2 = z1 * rx1[:, 2] + t[2]
            count = np.count_nonzero((z1 > 0) & (z2 > 0))
            if count > best_count:
                best_count = count
                best_R = R
                best_t = t
    return best_R, best_t
