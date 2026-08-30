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

There are two drivers. `build_ransac` is the serial one and stays the
default. `build_parallel_ransac` draws hypotheses in batches and spreads
their solve+score across numba threads; `RansacEstimator.estimate` selects
it when `num_threads > 1`. It trades exact agreement with the serial driver
for wall-clock. Measured on 8 physical cores at a 1000-iteration budget:
1.7-2.8x end to end, the smaller figure at large point counts. The batched
loop itself scales about 3.5-4x; local optimization and the final
refinement stay serial and cap the rest, and they cost more as the inlier
count grows, which is why 16k matches gain less than 2k. Threading buys
latency on a single estimate and nothing on throughput - a caller already
running one process per core should leave it off.
"""

import math

import numba
import numpy as np
from numba import njit, prange

from fastpose.kernel_cache import stabilize

# hypotheses per thread in one parallel batch when num_threads > 1; the batch
# is num_threads * this. Both ends of the range cost: a batch under ~16 does
# not amortize the parallel dispatch (~14us, measured), and a very large one
# loses the early bail-out because every hypothesis in a batch is scored
# against the incumbent as it stood when the batch started - on 8 threads at
# 16k points the loop peaks near a 256 batch and is already worse at 1024.
# 32 measured fastest end to end at both 2k and 16k points on 8 threads.
DEFAULT_BATCH_PER_THREAD = 32


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


def build_parallel_ransac(solve_fn, score_fn, refine_fn, sample_size,
                          num_params, max_models, workspace_size):
    # batched variant of `build_ransac`: hypotheses are drawn in batches and
    # the solve+score half of each batch runs across numba threads. Sampling
    # stays serial and consumes the same np.random stream as the serial
    # driver, so a given (seed, batch_size) draws exactly the same samples in
    # the same order no matter how many threads run - results depend on the
    # batch size but never on the thread count.
    #
    # It is NOT bit-identical to the serial driver, and cannot be: every
    # hypothesis in a batch is scored against `best_minimal_score` as it stood
    # when the batch started rather than against the running value, so models
    # the serial driver bails out on may be scored in full here. A full score
    # carries a true inlier count where a bailed-out one carries a partial
    # count, which can flip the `more_inliers` test and pick a different LO
    # candidate. Everything downstream of scoring - the tracker updates, the
    # LO trigger, the adaptive iteration count and the final refinement -
    # is merged back in hypothesis order and mirrors the serial driver
    # exactly.
    stabilize(solve_fn, score_fn, refine_fn)

    @njit(cache=True, parallel=True)
    def driver(data, num_points, min_iterations, max_iterations, max_error_sq,
               success_prob, lo_iterations, seed, batch_size):
        if seed >= 0:
            np.random.seed(seed)

        best_score = 1e300
        best_num_inliers = 0
        best_model = np.zeros(num_params)
        refined = np.empty(num_params)
        best_minimal_score = 1e300
        best_minimal_num_inliers = 0

        # per-hypothesis scratch, allocated once and reused across batches
        samples = np.empty((batch_size, sample_size), dtype=np.int64)
        models = np.empty((batch_size, max_models, num_params))
        workspace = np.empty((batch_size, workspace_size))
        counts = np.empty(batch_size, dtype=np.int64)
        scores = np.empty((batch_size, max_models))
        inlier_counts = np.empty((batch_size, max_models), dtype=np.int64)

        log_fail = math.log(1.0 - success_prob)
        dyn_max_iterations = max_iterations

        it = 0
        while True:
            # iterations the serial driver's loop condition would still allow.
            # Computing it up front means the adaptive count can only overshoot
            # by what a batch already had in flight when it was recomputed.
            limit = min_iterations
            cap = dyn_max_iterations
            if max_iterations < cap:
                cap = max_iterations
            if cap > limit:
                limit = cap
            batch = limit - it
            if batch <= 0:
                break
            if batch > batch_size:
                batch = batch_size

            # sampling is serial: it is a handful of RNG draws per hypothesis
            # against a ~10-70us solve+score, and keeping it here is what makes
            # the run reproducible independently of the thread count
            for q in range(batch):
                for k in range(sample_size):
                    while True:
                        idx = np.random.randint(0, num_points)
                        duplicate = False
                        for j in range(k):
                            if samples[q, j] == idx:
                                duplicate = True
                                break
                        if not duplicate:
                            break
                    samples[q, k] = idx

            bound = best_minimal_score
            for q in prange(batch):
                num = solve_fn(data, samples[q], models[q], workspace[q])
                counts[q] = num
                for m in range(num):
                    s, k = score_fn(models[q, m], data, max_error_sq, bound)
                    scores[q, m] = s
                    inlier_counts[q, m] = k

            # merge in hypothesis order, applying the serial driver's logic
            for q in range(batch):
                it += 1
                lo_candidate = -1
                lo_candidate_inliers = 0
                for m in range(counts[q]):
                    score = scores[q, m]
                    num_inliers = inlier_counts[q, m]
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
                            best_model[j] = models[q, m, j]

                if lo_candidate >= 0:
                    if lo_iterations > 0 and lo_candidate_inliers > sample_size:
                        if refine_fn(data, models[q, lo_candidate], refined,
                                     max_error_sq, lo_iterations):
                            lo_score, lo_num_inliers = score_fn(
                                refined, data, max_error_sq, best_score)
                            if lo_score < best_score:
                                best_score = lo_score
                                best_num_inliers = lo_num_inliers
                                for j in range(num_params):
                                    best_model[j] = refined[j]

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
        self._refine_fn = refiner.refine if refiner is not None else _no_refine
        self._driver = build_ransac(solver.solve, scorer.score, self._refine_fn,
                                    solver.sample_size, solver.num_params,
                                    solver.max_models, solver.workspace_size)
        self._parallel_driver = None

    def _get_parallel_driver(self):
        # built on first use: compiling it is expensive and most callers stay
        # on the serial path, so `fastpose-warmup` deliberately does not touch
        # it either
        if self._parallel_driver is None:
            solver = self.solver
            self._parallel_driver = build_parallel_ransac(
                solver.solve, self.scorer.score, self._refine_fn,
                solver.sample_size, solver.num_params, solver.max_models,
                solver.workspace_size)
        return self._parallel_driver

    def estimate(self, data, num_points, max_error, iterations=1000,
                 min_iterations=None, success_prob=0.9999, lo_iterations=None,
                 seed=None, num_threads=None, batch_per_thread=None):
        # returns (model as flat num_params vector, score, num_inliers, iterations)
        #
        # num_threads - None or 1 (default) runs the serial driver. Greater
        #     than 1 switches to the batched parallel driver, which draws
        #     num_threads * batch_per_thread hypotheses at a time and solves
        #     and scores them across that many numba threads. Clamped to
        #     numba's configured thread count (NUMBA_NUM_THREADS). Threading
        #     buys latency on one estimate and nothing on throughput: if the
        #     caller already runs one process per core, leave this alone.
        # batch_per_thread - hypotheses per thread in a batch, default
        #     DEFAULT_BATCH_PER_THREAD. Larger batches amortize the parallel
        #     dispatch but score against a staler incumbent; see the note on
        #     build_parallel_ransac for why the parallel result differs from
        #     the serial one.
        if min_iterations is None:
            min_iterations = iterations
        if lo_iterations is None:
            lo_iterations = self.refiner.num_iterations if self.refiner is not None else 0
        seed_arg = -1 if seed is None else seed

        if num_threads is None or num_threads <= 1:
            return self._driver(data, num_points, min_iterations, iterations,
                                max_error ** 2, success_prob, lo_iterations,
                                seed_arg)

        if batch_per_thread is None:
            batch_per_thread = DEFAULT_BATCH_PER_THREAD
        if batch_per_thread < 1:
            raise ValueError('batch_per_thread must be >= 1, got '
                             f'{batch_per_thread}')
        # numba refuses a count above the thread pool it was configured with
        num_threads = min(int(num_threads), numba.config.NUMBA_NUM_THREADS)
        batch_size = num_threads * int(batch_per_thread)

        driver = self._get_parallel_driver()
        previous_threads = numba.get_num_threads()
        numba.set_num_threads(num_threads)
        try:
            return driver(data, num_points, min_iterations, iterations,
                          max_error ** 2, success_prob, lo_iterations,
                          seed_arg, batch_size)
        finally:
            numba.set_num_threads(previous_threads)
