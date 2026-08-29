"""Generic numba-compiled LO-RANSAC engine.

The engine is problem-agnostic: a camera pose problem is defined by three
components, each a thin Python class wrapping a numba kernel with a fixed
signature. Adding a new problem (e.g. 5-point relative pose, P3P) only
requires implementing these kernels; the RANSAC loop itself is reused.

Solver class (solvers/):
    sample_size    - number of correspondences in a minimal sample
    num_params     - length of the flat model parameter vector
                     (9 for F/E as a flattened 3x3, 12 for a P3P pose [R|t])
    max_models     - maximum number of models a minimal sample can yield
    workspace_size - scratch floats the solve kernel needs (allocated once
                     by the driver, avoids per-iteration allocations)
    solve(data, sample, models, workspace) -> int
        writes up to max_models rows into `models`, returns how many

Scorer class (scorers/):
    score(model, data, max_error_sq, best_score) -> (score, num_inliers)
        truncated (MSAC) score; may bail out early once the partial score
        exceeds `best_score` since the truncated score only grows

Refiner class (refiners/, optional, enables LO-RANSAC):
    num_iterations - default number of refinement iterations
    refine(data, model, refined, max_error_sq, num_iterations) -> bool
        non-minimal refit of `model` on its inliers into `refined`

`data` is an arbitrary tuple of contiguous arrays; the engine never looks
inside it, only the problem kernels do. This keeps per-problem memory
layouts free to be SIMD-friendly (e.g. separate coordinate columns).
"""

import math

import numpy as np
from numba import njit

from fastpose.kernel_cache import stabilize


@njit(cache=True)
def _no_refine(data, model, refined, max_error_sq, num_iterations):
    # placeholder refiner used when local optimization is disabled
    return False


def build_ransac(solve_fn, score_fn, refine_fn, sample_size, num_params,
                 max_models, workspace_size):
    # compiles a LO-RANSAC driver specialized for one problem; the kernel
    # functions are closure constants so numba can bind the calls statically.
    # cache=True is what makes `fastpose-warmup` worth running: the driver is
    # the most expensive kernel in the package. Numba keys the cache on the
    # pickled closure cells, which both disambiguates the per-problem
    # specializations and - via stabilize - stays the same across processes.
    stabilize(solve_fn, score_fn, refine_fn)

    @njit(cache=True)
    def driver(data, num_points, min_iterations, max_iterations, max_error_sq,
               success_prob, lo_iterations, seed):
        if seed >= 0:
            np.random.seed(seed)

        best_score = 1e300
        best_num_inliers = 0
        best_model = np.zeros(num_params)
        refined = np.empty(num_params)
        models = np.empty((max_models, num_params))
        sample = np.empty(sample_size, dtype=np.int64)
        workspace = np.empty(workspace_size)

        # Best score and inlier count over *minimal* models only. These decide
        # when local optimization runs and are deliberately never updated by a
        # refined model, so a fresh minimal sample keeps triggering LO for the
        # whole run instead of only until the first refinement - refining from
        # an unrefined minimal sample explores a different basin than
        # re-refining the incumbent, and that is where most of the accuracy of
        # LO-RANSAC comes from. Mirrors poselib's RansacState::best_minimal_*.
        best_minimal_score = 1e300
        best_minimal_num_inliers = 0

        log_fail = math.log(1.0 - success_prob)
        dyn_max_iterations = max_iterations

        it = 0
        while it < min_iterations or (it < dyn_max_iterations and it < max_iterations):
            it += 1

            # draw sample_size distinct indices
            for k in range(sample_size):
                while True:
                    idx = np.random.randint(0, num_points)
                    duplicate = False
                    for j in range(k):
                        if sample[j] == idx:
                            duplicate = True
                            break
                    if not duplicate:
                        break
                sample[k] = idx

            num_models = solve_fn(data, sample, models, workspace)

            # the last minimal model to improve either minimal tracker; that is
            # the one handed to local optimization
            lo_candidate = -1
            lo_candidate_inliers = 0
            for m in range(num_models):
                # the bail-out bound is the best *minimal* score rather than the
                # global best, so a promising sample is still scored in full
                # long after the global best has been refined out of its reach.
                # A model that does bail out is dropped on both criteria: it
                # cannot beat best_minimal_score, and the inlier count that
                # comes back with a bailed-out score is partial, so it cannot
                # be compared either.
                score, num_inliers = score_fn(models[m], data, max_error_sq,
                                              best_minimal_score)
                more_inliers = num_inliers > best_minimal_num_inliers
                better_score = score < best_minimal_score
                if not (more_inliers or better_score):
                    continue
                if more_inliers:
                    best_minimal_num_inliers = num_inliers
                if better_score:
                    best_minimal_score = score
                lo_candidate = m
                lo_candidate_inliers = num_inliers

                if score < best_score:
                    best_score = score
                    best_num_inliers = num_inliers
                    for j in range(num_params):
                        best_model[j] = models[m, j]

            if lo_candidate >= 0:
                # local optimization: non-minimal refit seeded from the minimal
                # model, adopted only if it beats the global best
                if lo_iterations > 0 and lo_candidate_inliers > sample_size:
                    if refine_fn(data, models[lo_candidate], refined,
                                 max_error_sq, lo_iterations):
                        lo_score, lo_num_inliers = score_fn(refined, data,
                                                            max_error_sq, best_score)
                        if lo_score < best_score:
                            best_score = lo_score
                            best_num_inliers = lo_num_inliers
                            for j in range(num_params):
                                best_model[j] = refined[j]

                # adaptive iteration count (only relevant when
                # min_iterations < max_iterations)
                eps = best_num_inliers / num_points
                if eps > 0.0:
                    eps_k = eps ** sample_size
                    if eps_k >= 1.0:
                        dyn_max_iterations = 0
                    elif eps_k > 0.0:
                        log_success_fail = math.log(1.0 - eps_k)
                        if log_success_fail < 0.0:
                            dyn_max_iterations = int(log_fail / log_success_fail) + 1
                        else:
                            dyn_max_iterations = max_iterations
                    else:
                        dyn_max_iterations = max_iterations

        # final refinement of the overall best model
        if lo_iterations > 0 and best_num_inliers > sample_size:
            if refine_fn(data, best_model, refined, max_error_sq, lo_iterations):
                lo_score, lo_num_inliers = score_fn(refined, data, max_error_sq, best_score)
                if lo_score < best_score:
                    best_score = lo_score
                    best_num_inliers = lo_num_inliers
                    for j in range(num_params):
                        best_model[j] = refined[j]

        return best_model, best_score, best_num_inliers, it

    return driver


class RansacEstimator():
    # generic LO-RANSAC estimator assembled from a solver, a scorer and an
    # optional refiner (no refiner or lo_iterations=0 -> plain RANSAC)
    def __init__(self, solver, scorer, refiner=None):
        self.solver = solver
        self.scorer = scorer
        self.refiner = refiner
        refine_fn = refiner.refine if refiner is not None else _no_refine
        self._driver = build_ransac(solver.solve, scorer.score, refine_fn,
                                    solver.sample_size, solver.num_params,
                                    solver.max_models, solver.workspace_size)

    def estimate(self, data, num_points, max_error, iterations=1000,
                 min_iterations=None, success_prob=0.9999, lo_iterations=None,
                 seed=None):
        # returns (model as flat num_params vector, score, num_inliers, iterations)
        if min_iterations is None:
            min_iterations = iterations
        if lo_iterations is None:
            lo_iterations = self.refiner.num_iterations if self.refiner is not None else 0
        return self._driver(data, num_points, min_iterations, iterations,
                            max_error ** 2, success_prob, lo_iterations,
                            -1 if seed is None else seed)
