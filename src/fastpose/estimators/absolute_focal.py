"""Full absolute pose estimator with unknown focal length: LO-RANSAC with
the P4Pf solver (poselib port), truncated pixel reprojection scorer and
direct pose + log-focal LM refiner."""

import numpy as np

from fastpose.estimators.ransac import RansacEstimator
from fastpose.estimators.utils import (build_info, check_device, check_min_points,
                                       failure_info, get_cuda_estimator)
from fastpose.refiners.absolute_focal import LMAbsolutePoseFocalRefiner
from fastpose.refiners.losses import CauchyLoss
from fastpose.scorers.reprojection import FocalReprojectionScorer
from fastpose.solvers.p4pf import P4PFSolver

_default_estimator = None
_final_refiner = None


def _get_default_estimator():
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = RansacEstimator(P4PFSolver(),
                                             FocalReprojectionScorer(),
                                             LMAbsolutePoseFocalRefiner())
    return _default_estimator


def _get_final_refiner():
    # loss for the final polish pass on RANSAC inliers only; see
    # refiners/losses.py for the available Loss objects
    global _final_refiner
    if _final_refiner is None:
        _final_refiner = LMAbsolutePoseFocalRefiner(loss=CauchyLoss())
    return _final_refiner


def estimate_absolute_pose_with_focal(x, X, principal_point=None,
                                      iterations=1000, max_error=2.0,
                                      seed=4578, min_iterations=None,
                                      success_prob=0.9999, lo_iterations=25,
                                      final_refinement_iterations=100,
                                      num_threads=None,
                                      batch_per_thread=None,
                                      device='cpu', batch=None):
    # params:
    # x - (n, 2) array of image points in pixel coordinates (square pixels,
    #     unknown focal length)
    # X - (n, 3) array of corresponding 3D world points
    # principal_point - optional (cx, cy); if omitted, zero is assumed
    # max_error - truncated reprojection threshold in pixels
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
    # device - 'cpu' (default) runs the numba CPU drivers above; 'cuda' runs
    #     the batch-parallel GPU driver (fastpose/cuda/ransac.py), which
    #     solves, scores, locally optimizes and polishes on device. Ignores
    #     num_threads. See the cuda/ransac.py docstring for what differs.
    # batch - hypotheses per GPU round; None uses cuda.ransac.DEFAULT_BATCH
    # returns (model, info) with model = {'R', 't', 'f'}
    # (lambda * (x, y, 1) = diag(f, f, 1) (R X + t)) and info = {'inliers',
    # 'num_inliers', 'model_score', 'iterations', 'refinements'}; on total
    # failure model holds the identity pose with f=1.0 and
    # info['num_inliers'] is 0
    x = np.ascontiguousarray(x, dtype=np.float64)
    X = np.ascontiguousarray(X, dtype=np.float64)
    check_min_points(len(x), P4PFSolver.sample_size)
    check_device(device)
    if principal_point is not None:
        pp = np.asarray(principal_point, dtype=np.float64)
        if pp.shape != (2,):
            raise ValueError("principal_point must be a length-2 array")
        x = x - pp
    data = (np.ascontiguousarray(x[:, 0]), np.ascontiguousarray(x[:, 1]),
            np.ascontiguousarray(X[:, 0]), np.ascontiguousarray(X[:, 1]),
            np.ascontiguousarray(X[:, 2]))

    cuda_estimator = None
    if device == 'cuda':
        cuda_estimator = get_cuda_estimator('absolute-focal', batch)
        model, _, num_inliers, ransac_iterations = cuda_estimator.estimate(
            data, len(x), max_error, iterations=iterations,
            min_iterations=min_iterations, success_prob=success_prob,
            lo_iterations=lo_iterations, seed=seed)
    else:
        estimator = _get_default_estimator()
        model, _, num_inliers, ransac_iterations = estimator.estimate(
            data, len(x), max_error, iterations=iterations,
            min_iterations=min_iterations, success_prob=success_prob,
            lo_iterations=lo_iterations, seed=seed,
            num_threads=num_threads, batch_per_thread=batch_per_thread)

    if num_inliers == 0:
        return ({'R': np.eye(3), 't': np.zeros(3), 'f': 1.0},
                failure_info(len(x), ransac_iterations))

    R = model[:9].reshape(3, 3).copy()
    t = model[9:12].copy()
    f = float(model[12])
    score, inliers, num_inliers = FocalReprojectionScorer.score_numpy(
        R, t, f, x, X, max_error)
    refined = False

    # final polish: robust-loss refinement restricted to the RANSAC inliers.
    # Fewer inliers than the minimal sample size cannot constrain the model,
    # so the pass is skipped there (poselib gates its bundle the same way)
    if final_refinement_iterations != 0 and num_inliers > P4PFSolver.sample_size:
        final_refiner = _get_final_refiner()
        num_final_iterations = (final_refiner.num_iterations
                                if final_refinement_iterations is None
                                else final_refinement_iterations)
        if cuda_estimator is not None:
            # on device, through the same LM kernel built for the Cauchy loss
            refined_model = cuda_estimator.final_refine(
                model, max_error ** 2, num_final_iterations,
                final_refiner.loss)
            ok = refined_model is not None
        else:
            inlier_data = (np.ascontiguousarray(x[inliers, 0]),
                           np.ascontiguousarray(x[inliers, 1]),
                           np.ascontiguousarray(X[inliers, 0]),
                           np.ascontiguousarray(X[inliers, 1]),
                           np.ascontiguousarray(X[inliers, 2]))
            refined_model = np.empty(13)
            ok = final_refiner.refine(inlier_data, model, refined_model,
                                      max_error ** 2, num_final_iterations)
        if ok:
            R_c = refined_model[:9].reshape(3, 3).copy()
            t_c = refined_model[9:12].copy()
            f_c = float(refined_model[12])
            score_c, inliers_c, num_inliers_c = FocalReprojectionScorer.score_numpy(
                R_c, t_c, f_c, x, X, max_error)
            R, t, f, inliers, num_inliers, score = (R_c, t_c, f_c, inliers_c,
                                                     num_inliers_c, score_c)
            refined = True

    return ({'R': R, 't': t, 'f': f},
            build_info(inliers, num_inliers, score, ransac_iterations, refined))
