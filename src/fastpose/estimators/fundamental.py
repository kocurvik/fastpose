"""Full fundamental matrix estimator: LO-RANSAC with the 7-point solver,
Sampson scorer and factorized LM refiner, plus Hartley-style normalization.
A pure-numpy reference implementation (`estimate_fundamental_numpy`) is kept
for benchmarking.
"""

import numpy as np

from fastpose.estimators.ransac import RansacEstimator
from fastpose.estimators.utils import (build_info, check_min_points, failure_info,
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
                         lo_iterations=25, final_refinement_iterations=100):
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
    # returns (model, info) with model = {'F'} and info = {'inliers',
    # 'num_inliers', 'model_score', 'iterations', 'refinements'}; on total
    # failure model holds an all-zero placeholder F and
    # info['num_inliers'] is 0
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    check_min_points(len(x1), SevenPointSolver.sample_size)
    x1n, x2n, T, scale = normalize_points(x1, x2)
    data = point_columns(x1n, x2n)

    estimator = _get_default_estimator()
    model, _, num_inliers, ransac_iterations = estimator.estimate(
        data, len(x1), max_error * scale, iterations=iterations,
        min_iterations=min_iterations, success_prob=success_prob,
        lo_iterations=lo_iterations, seed=seed)

    if num_inliers == 0:
        return {'F': np.zeros((3, 3))}, failure_info(len(x1), ransac_iterations)

    F = T.T @ model.reshape(3, 3) @ T
    score, inliers, num_inliers = SampsonScorer.score_numpy(F, x1, x2, max_error)
    refined = False

    # final polish: robust-loss refinement restricted to the RANSAC inliers,
    # done in the same normalized frame/threshold as the RANSAC pipeline
    if final_refinement_iterations != 0 and num_inliers > 0:
        final_refiner = _get_final_refiner()
        inlier_data = point_columns(x1n[inliers], x2n[inliers])
        refined_model = np.empty(9)
        num_final_iterations = (final_refiner.num_iterations
                                if final_refinement_iterations is None
                                else final_refinement_iterations)
        if final_refiner.refine(inlier_data, model, refined_model,
                                (max_error * scale) ** 2,
                                num_final_iterations):
            F_c = T.T @ refined_model.reshape(3, 3) @ T
            score_c, inliers_c, num_inliers_c = SampsonScorer.score_numpy(
                F_c, x1, x2, max_error)
            F, inliers, num_inliers, score = F_c, inliers_c, num_inliers_c, score_c
            refined = True

    return {'F': F}, build_info(inliers, num_inliers, score, ransac_iterations, refined)
