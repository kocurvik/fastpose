"""Full fundamental matrix estimator: LO-RANSAC with the 7-point solver,
Sampson scorer and factorized LM refiner, plus Hartley-style normalization.
A pure-numpy reference implementation (`estimate_fundamental_numpy`) is kept
for benchmarking.
"""

import numpy as np

from fastpose.estimators.ransac import RansacEstimator
from fastpose.estimators.utils import (build_info, check_device, check_min_points,
                                       failure_info, get_cuda_estimator,
                                       normalize_points, point_columns)
from fastpose.refiners.fundamental import LMFundamentalRefiner
from fastpose.refiners.losses import CauchyLoss
from fastpose.scorers.sampson import SampsonScorer
from fastpose.solvers.fundamental import SevenPointSolver, seven_point


def estimate_fundamental_numpy(x1, x2, iterations=1000, max_error=2.0, seed=4578):
    # pure-numpy reference RANSAC (no local optimization)
    check_min_points(len(x1), SevenPointSolver.sample_size)
    x1n, x2n, T, scale = normalize_points(x1, x2)
    threshold = max_error * scale
    best_score = np.inf
    best_model = None
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        idxs = rng.choice(len(x1n), 7, replace=False)
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
_final_refiner = None


def _get_default_estimator():
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = RansacEstimator(SevenPointSolver(), SampsonScorer(),
                                             LMFundamentalRefiner())
    return _default_estimator


def _get_final_refiner():
    # loss for the final polish pass on RANSAC inliers only; see
    # refiners/losses.py for the available Loss objects
    global _final_refiner
    if _final_refiner is None:
        _final_refiner = LMFundamentalRefiner(loss=CauchyLoss())
    return _final_refiner


def estimate_fundamental(x1, x2, iterations=1000, max_error=2.0, seed=4578,
                         min_iterations=None, success_prob=0.9999,
                         lo_iterations=25, final_refinement_iterations=100,
                         num_threads=None, batch_per_thread=None,
                         device='cpu', batch=None):
    # params:
    # x1, x2 - (n, 2) arrays of corresponding points
    # iterations - maximum number of RANSAC iterations
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
    # returns (model, info) with model = {'F'} and info = {'inliers',
    # 'num_inliers', 'model_score', 'iterations', 'refinements'}; on total
    # failure model holds an all-zero placeholder F and
    # info['num_inliers'] is 0
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    check_min_points(len(x1), SevenPointSolver.sample_size)
    check_device(device)
    x1n, x2n, T, scale = normalize_points(x1, x2)
    data = point_columns(x1n, x2n)

    cuda_estimator = None
    if device == 'cuda':
        cuda_estimator = get_cuda_estimator('fundamental', batch)
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
        return {'F': np.zeros((3, 3))}, failure_info(len(x1), ransac_iterations)

    F = T.T @ model.reshape(3, 3) @ T
    score, inliers, num_inliers = SampsonScorer.score_numpy(F, x1, x2, max_error)
    refined = False

    # final polish: robust-loss refinement restricted to the RANSAC inliers,
    # done in the same normalized frame/threshold as the RANSAC pipeline.
    # Fewer inliers than the minimal sample size cannot constrain the model,
    # so the pass is skipped there (poselib gates its bundle the same way)
    if (final_refinement_iterations != 0
            and num_inliers > SevenPointSolver.sample_size):
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
            F_c = T.T @ refined_model.reshape(3, 3) @ T
            score_c, inliers_c, num_inliers_c = SampsonScorer.score_numpy(
                F_c, x1, x2, max_error)
            F, inliers, num_inliers, score = F_c, inliers_c, num_inliers_c, score_c
            refined = True

    return {'F': F}, build_info(inliers, num_inliers, score, ransac_iterations, refined)
