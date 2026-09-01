"""Full homography estimator: LO-RANSAC with the 4-point DLT solver, the
symmetric transfer error scorer and the sphere-parametrized LM refiner, plus
Hartley-style normalization.

`max_error` is a one-way pixel distance, exactly as it is in poselib's
`estimate_homography` and OpenCV's `findHomography`: the scorer averages the
forward and backward transfer terms rather than summing them, so a threshold
ports across unchanged even though the functional being minimized is the
symmetric one. See scorers/transfer.py.
"""

import numpy as np

from fastpose.estimators.ransac import RansacEstimator
from fastpose.estimators.utils import (build_info, check_device, check_min_points,
                                       failure_info, get_cuda_estimator,
                                       normalize_points, point_columns)
from fastpose.refiners.homography import LMHomographyRefiner
from fastpose.refiners.losses import CauchyLoss
from fastpose.scorers.transfer import SymmetricTransferScorer
from fastpose.solvers.homography import FourPointSolver

_default_estimator = None
_final_refiner = None


def _get_default_estimator():
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = RansacEstimator(FourPointSolver(),
                                             SymmetricTransferScorer(),
                                             LMHomographyRefiner())
    return _default_estimator


def _get_final_refiner():
    # loss for the final polish pass on RANSAC inliers only; see
    # refiners/losses.py for the available Loss objects
    global _final_refiner
    if _final_refiner is None:
        _final_refiner = LMHomographyRefiner(loss=CauchyLoss())
    return _final_refiner


def estimate_homography(x1, x2, iterations=1000, max_error=2.0, seed=4578,
                        min_iterations=None, success_prob=0.9999,
                        lo_iterations=25, final_refinement_iterations=100,
                        num_threads=None, batch_per_thread=None,
                        device='cpu', batch=None):
    # params:
    # x1, x2 - (n, 2) arrays of corresponding points, in pixels; the estimated
    #          H maps the first image to the second (x2 ~ H x1)
    # iterations - maximum number of RANSAC iterations
    # max_error - inlier threshold in pixels, on the symmetric transfer error
    #             averaged over the two directions:
    #             sqrt((d(x2, H x1)^2 + d(x1, H^-1 x2)^2) / 2). The averaging
    #             makes this the same one-way pixel distance poselib and
    #             OpenCV threshold, so a threshold ports across unchanged
    # min_iterations - minimum number of iterations before adaptive
    #                  termination may stop early; defaults to `iterations`
    #                  (fixed iteration count)
    # lo_iterations - LM step budget per local optimization; 0 disables
    #                 local optimization (plain RANSAC)
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
    #     solves, scores, locally optimizes and polishes on device, in the
    #     same normalized frame. Ignores num_threads.
    # batch - hypotheses per GPU round; None uses cuda.ransac.DEFAULT_BATCH
    # returns (model, info) with model = {'H'} and info = {'inliers',
    # 'num_inliers', 'model_score', 'iterations', 'refinements'}; on total
    # failure model holds an all-zero placeholder H and
    # info['num_inliers'] is 0
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    check_min_points(len(x1), FourPointSolver.sample_size)
    check_device(device)
    x1n, x2n, T, scale = normalize_points(x1, x2)
    data = point_columns(x1n, x2n)
    # x1n = T x1 and x2n = T x2 with the same similarity T, so a normalized
    # H maps back as T^-1 H T
    T_inv = np.linalg.inv(T)

    cuda_estimator = None
    if device == 'cuda':
        cuda_estimator = get_cuda_estimator('homography', batch)
        model, _, num_inliers, ransac_iterations = cuda_estimator.estimate(
            data, len(x1), max_error * scale, iterations=iterations,
            min_iterations=min_iterations, success_prob=success_prob,
            lo_iterations=lo_iterations, seed=seed)
    else:
        estimator = _get_default_estimator()
        model, _, num_inliers, ransac_iterations = estimator.estimate(
            data, len(x1), max_error * scale, iterations=iterations,
            min_iterations=min_iterations, success_prob=success_prob,
            lo_iterations=lo_iterations, seed=seed,
            num_threads=num_threads, batch_per_thread=batch_per_thread)

    if num_inliers == 0:
        return {'H': np.zeros((3, 3))}, failure_info(len(x1), ransac_iterations)

    H = T_inv @ model.reshape(3, 3) @ T
    score, inliers, num_inliers = SymmetricTransferScorer.score_numpy(
        H, x1, x2, max_error)
    refined = False

    # final polish: robust-loss refinement restricted to the RANSAC inliers,
    # done in the same normalized frame/threshold as the RANSAC pipeline.
    # Fewer inliers than the minimal sample size cannot constrain the model,
    # so the pass is skipped there (poselib gates its bundle the same way)
    if (final_refinement_iterations != 0
            and num_inliers > FourPointSolver.sample_size):
        final_refiner = _get_final_refiner()
        num_final_iterations = (final_refiner.num_iterations
                                if final_refinement_iterations is None
                                else final_refinement_iterations)
        if cuda_estimator is not None:
            # on device, through the same LM kernel built for the Cauchy loss
            refined_model = cuda_estimator.final_refine(
                model, (max_error * scale) ** 2, num_final_iterations,
                final_refiner.loss)
            ok = refined_model is not None
        else:
            inlier_data = point_columns(x1n[inliers], x2n[inliers])
            refined_model = np.empty(9)
            ok = final_refiner.refine(inlier_data, model, refined_model,
                                      (max_error * scale) ** 2,
                                      num_final_iterations)
        if ok:
            H_c = T_inv @ refined_model.reshape(3, 3) @ T
            score_c, inliers_c, num_inliers_c = \
                SymmetricTransferScorer.score_numpy(H_c, x1, x2, max_error)
            H, inliers, num_inliers, score = H_c, inliers_c, num_inliers_c, score_c
            refined = True

    return {'H': H}, build_info(inliers, num_inliers, score, ransac_iterations,
                                refined)
