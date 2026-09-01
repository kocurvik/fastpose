"""Batch-parallel LO-RANSAC driver on the GPU.

Structure of one round, all on device except the marked step:

    sample   one thread per hypothesis, xoroshiro128+ per-thread streams
    solve    one thread per hypothesis            (cuda/problems/*.py)
    score    one block per hypothesis, reduced to
             its best-by-score and best-by-inliers (cuda/scoring.py)
    -------- ~40 KB readback, host picks the LO candidates --------
    refine   one block per candidate, whole LM loop in-kernel
             (cuda/lm.py)
    score    the refined models, same kernel as above

The driver is problem-agnostic: everything it needs about a problem - sample
size, model width, the three kernel families, the data layout - comes from
the `CudaProblem` it is constructed with (see cuda/problem.py). Only the
`data` tuple width and the extra `params` vector vary between problems, and
both are passed straight through to the kernels.

How this differs from the CPU drivers
-------------------------------------
`build_ransac` is sequential: every hypothesis is scored against the running
incumbent and local optimization fires whenever a minimal model improves
either the best score or the best inlier count. That is inherently serial and
there is no way to keep a GPU busy with it.

This driver is a *batched* LO-RANSAC instead, and the differences are real
rather than rounding:

- **Local optimization sees one candidate per round**, not one per improving
  hypothesis. A round is gated on the same criterion `build_ransac` uses - a
  minimal model is worth refining only if it improves the best *minimal* score
  or inlier count - and the best-scoring hypothesis that passes the gate is
  refined. Rounds where nothing passes skip local optimization entirely.

  An earlier version refined the round's top-k in parallel, k configurable.
  Measured over a 20000-iteration budget at 16k matches, that was pure waste:
  the gate only opens wide in the first round, so k sized exactly one launch
  and nothing else, and k = 1, 4, 32 and 128 all returned the same 11210
  inliers and the same pose to 1e-4 degrees at the same wall clock. The knob
  is gone; the gate is what actually controls the local-optimization budget.
- **Scoring bails out against a round-stale bound.** The bail-out itself is
  kept (see cuda/scoring.py), but every hypothesis of a round is scored
  against the best minimal score as it stood when the round *started*, not
  against a running incumbent. Same staleness `build_parallel_ransac` accepts.
- **Adaptive termination is evaluated per round**, so it can overshoot the
  serial driver by up to one batch.
- **Sampling uses a different RNG.** Results are reproducible from
  `(seed, batch)` on a given device, but they are not the CPU driver's
  sample sequence and cannot be compared iteration by iteration.

The final polish pass also runs on device, through the same LM kernel built
for the Cauchy loss: at large `n` it is O(n) work per step and would otherwise
dominate everything the GPU just saved.
"""

import math

import numpy as np
from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_uniform_float64)

from fastpose.cuda.backend import THREADS_PER_BLOCK, quiet_low_occupancy
from fastpose.cuda.lm import RefineBuffers, gather_candidates
from fastpose.cuda.scoring import NO_MODEL, ScoreBuffers
from fastpose.refiners.losses import TruncatedLoss

# Hypotheses per round.
#
# The per-round floor is the minimal solver's *latency*, not the readback or
# the launch overhead. One 5-point solve on a single GPU thread takes ~1 ms
# on the development card (it is entirely float64, and carries ~6.9 KB of
# per-thread local memory), and the whole kernel still takes only ~5 ms with
# 4096 hypotheses in flight - measured 4.54, 4.98, 4.90, 4.65 and 5.62 ms at
# 32, 128, 512, 1024 and 4096. That is 142 us per hypothesis at 32 and 1.4 us
# at 4096, so a large round is how the latency gets amortized, and it is why
# bigger wins uniformly (~20x from 128 to 4096 over a 20000-iteration budget)
# at every match count.
#
# Expect this floor to fall sharply on a datacenter GPU: the solvers are
# float64 throughout and that card runs float64 at 1/64 of float32, where an
# A100 runs it at 1/2. A flatter curve there makes the exact value matter less.
#
# Adaptive runs are protected from the resulting overshoot by the ramp in
# `estimate` rather than by keeping this small.
DEFAULT_BATCH = 4096

# Size of the first round; subsequent rounds double until they reach `batch`.
# See the ramp note in `estimate` for why this is not simply `batch`.
FIRST_ROUND = 256

# Device-side slots for the round's gathered models: the local-optimization
# candidate, plus the round's best minimal model when the gate rejects it (the
# driver still needs that one on the host to adopt as the incumbent).
MAX_GATHERED = 2

SAMPLE_THREADS = 128


@cuda.jit(cache=True)
def _sample_kernel(rng_states, num_points, samples):
    # one thread per hypothesis; rejection-samples SAMPLE_SIZE distinct
    # indices. The state array persists across rounds, so a given seed
    # reproduces a whole run rather than just one round.
    i = cuda.grid(1)
    if i >= samples.shape[0]:
        return
    for k in range(samples.shape[1]):
        while True:
            u = xoroshiro128p_uniform_float64(rng_states, i)
            idx = int(u * num_points)
            if idx >= num_points:
                idx = num_points - 1
            duplicate = False
            for j in range(k):
                if samples[i, j] == idx:
                    duplicate = True
                    break
            if not duplicate:
                break
        samples[i, k] = idx


def score_batch(problem, data32, params, models, counts, max_error_sq,
                buffers, batch, bound=NO_MODEL, stream=0):
    # launches the batched scorer over `batch` hypotheses. `data32` is the
    # float32 coordinate columns; `models` stays float64.
    #
    # `bound` is the incumbent to bail out against - the driver passes the
    # best *minimal* score as it stood at the start of the round, which is the
    # batched analogue of the CPU driver's running bound. NO_MODEL disables
    # the bail-out, which is what the refined-model scoring pass wants: there
    # are only a handful of models and their true scores are needed.
    problem.score_kernel()[batch, THREADS_PER_BLOCK, stream](
        data32, params, models, counts, max_error_sq, bound, buffers.out)


def refine_prepared(problem, data32, params, buffers, num_candidates,
                    max_error_sq, num_iterations, loss, relaxed_scale=None,
                    stream=0):
    # locally optimizes the models already sitting in buffers.init_models;
    # results land in buffers.refined, whose last column is the per-candidate
    # success flag
    if relaxed_scale is None:
        relaxed_scale = problem.relaxed_scale
    kernel = problem.lm_kernel(loss)
    kernel[num_candidates, problem.lm_threads, stream](
        data32, params, buffers.init_models, buffers.refined, buffers.keep,
        max_error_sq, relaxed_scale * max_error_sq, num_iterations)


class CudaRansacEstimator():
    """LO-RANSAC for one problem, run in batches on the GPU.

    Holds the device-side buffers so repeated `estimate` calls on the same
    problem size reuse them instead of reallocating. Not thread-safe: one
    estimator drives one CUDA stream's worth of work at a time.
    """

    def __init__(self, problem, batch=DEFAULT_BATCH):
        self.problem = problem
        self.batch = int(batch)
        self._n = None
        self._device_data = None
        self._device_data32 = None
        self._params = None
        self._params_host = None
        self._models = None
        self._counts = None
        self._samples = None
        self._score_buf = None
        self._refine_buf = None
        self._rng = None

    # -- device buffer management -----------------------------------------

    def _ensure(self, data, params, num_points, seed):
        problem = self.problem
        batch = self.batch
        if self._models is None:
            self._models, self._counts = problem.allocate_models(batch)
            self._samples = cuda.device_array((batch, problem.sample_size),
                                              dtype=np.int64)
            self._score_buf = ScoreBuffers(batch)
        if self._n != num_points:
            self._refine_buf = RefineBuffers(MAX_GATHERED, num_points,
                                             problem.num_params)
            # persistent coordinate columns: reallocating these per call
            # churns device memory for no reason, since consecutive estimates
            # on the same problem size are the common case.
            #
            # Both precisions are kept. The minimal solvers read float64 -
            # they touch only `sample_size` points per hypothesis, so the copy
            # is free and their conditioning is the one place float32 is
            # genuinely unsafe (the 5-point action matrix exceeds cond 1e5 on
            # ~15% of samples). The scorer and the LM read float32, which is
            # where all the O(matches) traffic and arithmetic is.
            self._device_data = tuple(
                cuda.device_array(num_points, dtype=np.float64)
                for _ in range(problem.data_width))
            self._device_data32 = tuple(
                cuda.device_array(num_points, dtype=np.float32)
                for _ in range(problem.data_width))
            self._n = num_points
        for dst, dst32, src in zip(self._device_data, self._device_data32,
                                   data):
            host = np.ascontiguousarray(src, dtype=np.float64)
            dst.copy_to_device(host)
            dst32.copy_to_device(host.astype(np.float32))

        host_params = np.ascontiguousarray(
            problem.default_params if params is None else params,
            dtype=np.float64)
        if (self._params is None
                or self._params.shape[0] != host_params.shape[0]):
            self._params = cuda.device_array(host_params.shape[0],
                                             dtype=np.float64)
        if (self._params_host is None
                or not np.array_equal(self._params_host, host_params)):
            self._params.copy_to_device(host_params)
            self._params_host = host_params.copy()

        # rebuilt on every call, not cached: the states advance as rounds
        # consume them, so reusing them would make a second `estimate` with
        # the same seed continue the stream instead of repeating it
        self._rng = create_xoroshiro128p_states(batch, seed=seed)

    # -- one round ---------------------------------------------------------

    def _round(self, num_points, count, max_error_sq, bound):
        problem = self.problem
        _sample_kernel[(count + SAMPLE_THREADS - 1) // SAMPLE_THREADS,
                       SAMPLE_THREADS](self._rng, num_points, self._samples)
        problem.solve_batch(self._device_data, self._params,
                            self._samples[:count], self._models, self._counts)
        score_batch(problem, self._device_data32, self._params, self._models,
                    self._counts, max_error_sq, self._score_buf, count,
                    bound=bound)
        return self._score_buf.to_host(count)

    # -- driver ------------------------------------------------------------

    @quiet_low_occupancy()
    def estimate(self, data, num_points, max_error, iterations=1000,
                 min_iterations=None, success_prob=0.9999, lo_iterations=25,
                 seed=None, loss=None, params=None):
        # returns (best_model as a flat num_params vector, best_score,
        # num_inliers, iterations actually drawn)
        problem = self.problem
        if min_iterations is None:
            min_iterations = iterations
        if loss is None:
            loss = TruncatedLoss()
        # seed=None means "unseeded" on the CPU path, where the driver simply
        # does not reseed numpy's global stream; the device RNG has no global
        # stream to inherit, so draw a fresh seed instead of silently making
        # every unseeded call identical
        seed_arg = (int(np.random.randint(0, 2 ** 31 - 1)) if seed is None
                    else int(seed))
        max_error_sq = max_error ** 2

        self._ensure(data, params, num_points, seed_arg)

        best_score = NO_MODEL
        best_num_inliers = 0
        best_model = np.zeros(problem.num_params)
        # tracked exactly as build_ransac does: over *minimal* models only, so
        # a fresh minimal sample keeps triggering local optimization for the
        # whole run rather than only until the first refinement
        best_minimal_score = NO_MODEL
        best_minimal_num_inliers = 0

        log_fail = math.log(1.0 - success_prob)
        dyn_max_iterations = iterations

        # The two regimes want opposite round sizes, so the driver picks per
        # run rather than exposing another knob:
        #
        #   fixed iteration count (min_iterations >= iterations, the default)
        #     - every round pays the solve kernel's latency floor (see
        #     DEFAULT_BATCH), which is nearly independent of the round size, so
        #     *large* rounds win at every match count (measured 20x from 128 to
        #     4096 over a 20000-iteration budget). Run at the full batch from
        #     the start.
        #
        #   adaptive termination (min_iterations < iterations) - a round
        #     overshoots the stopping point by up to its own size, and the
        #     wasted time is `round x matches`. Ramp geometrically from
        #     FIRST_ROUND instead: a run that stops after one or two rounds
        #     barely overshoots, and a run that goes the distance reaches the
        #     full batch within a few rounds and still amortizes.
        #
        # Note this is a function of the *iteration budget*, not of the match
        # count - scaling the round size by `num_points` looked plausible but
        # measured the wrong variable; see the note in README.
        adaptive = min_iterations < iterations
        round_batch = min(FIRST_ROUND, self.batch) if adaptive else self.batch

        it = 0
        while True:
            limit = max(min_iterations, min(dyn_max_iterations, iterations))
            remaining = limit - it
            if remaining <= 0:
                break
            count = min(remaining, round_batch)
            round_batch = min(round_batch * 2, self.batch)

            # every hypothesis of the round is scored against the incumbent
            # as it stood when the round started - the same staleness the
            # parallel CPU driver accepts, and what makes the bail-out usable
            # from a block reduction
            h_score, h_idx, h_inl, h_max_inl, h_max_idx = self._round(
                num_points, count, max_error_sq, best_minimal_score)
            it += count

            valid = h_idx >= 0
            if not np.any(valid):
                continue

            # Gate local optimization on the same criterion build_ransac uses:
            # only a minimal model that improves the best *minimal* score or
            # the best minimal inlier count is worth refining. The trackers are
            # read as they stood at the start of the round rather than updated
            # hypothesis by hypothesis, which is the batched analogue of the
            # serial rule.
            improves = valid & ((h_score < best_minimal_score)
                                | (h_max_inl > best_minimal_num_inliers))

            order = np.argsort(h_score, kind='stable')
            round_best = int(order[valid[order]][0])
            qualifying = order[improves[order]]
            lo_hyp = int(qualifying[0]) if qualifying.size else -1

            if h_score[round_best] < best_minimal_score:
                best_minimal_score = float(h_score[round_best])
            round_max_inl = int(h_max_inl.max())
            if round_max_inl > best_minimal_num_inliers:
                best_minimal_num_inliers = round_max_inl

            # Everything above came from the one packed readback, so a round
            # that neither improves the incumbent nor has a candidate to refine
            # touches the device no further - no gather, no extra copies. In a
            # long run most rounds are exactly that, and each avoided
            # copy_to_host is a full host/device synchronization.
            run_lo = lo_iterations > 0 and lo_hyp >= 0
            improves_global = h_score[round_best] < best_score

            hyp = []
            if run_lo:
                hyp.append(lo_hyp)
            lo_k = len(hyp)
            best_pos = -1
            if improves_global:
                # the round's best minimal model is needed on the host to
                # become the incumbent; it may already be the LO candidate
                if round_best in hyp:
                    best_pos = hyp.index(round_best)
                else:
                    best_pos = len(hyp)
                    hyp.append(round_best)

            if hyp:
                mdl = [int(h_idx[h]) for h in hyp]
                # launch everything first, then synchronize once per result
                gather_candidates(self._models, self._refine_buf,
                                  np.asarray(hyp, dtype=np.int64),
                                  np.asarray(mdl, dtype=np.int64), len(hyp))
                if run_lo:
                    refine_prepared(problem, self._device_data32, self._params,
                                    self._refine_buf, lo_k, max_error_sq,
                                    lo_iterations, loss)
                    score_batch(problem, self._device_data32, self._params,
                                self._refine_buf.refined,
                                self._refine_buf.counts, max_error_sq,
                                self._score_buf, lo_k)

                if improves_global:
                    init_models = self._refine_buf.init_models[
                        :len(hyp)].copy_to_host()
                    best_score = float(h_score[round_best])
                    best_num_inliers = int(h_inl[round_best])
                    best_model[:] = init_models[best_pos]

                if run_lo:
                    r_score, r_idx, r_inl, _, _ = self._score_buf.to_host(lo_k)
                    # the success flag rides in the refined model's last
                    # column, so this is one copy rather than two
                    result = self._refine_buf.refined[:lo_k].copy_to_host()
                    if (result[0, 0, problem.num_params] != 0.0
                            and r_idx[0] >= 0 and r_score[0] < best_score):
                        best_score = float(r_score[0])
                        best_num_inliers = int(r_inl[0])
                        best_model[:] = result[0, 0, :problem.num_params]

            # adaptive iteration count, evaluated once per round
            if best_num_inliers > 0:
                eps = best_num_inliers / num_points
                eps_k = eps ** problem.sample_size
                if eps_k >= 1.0:
                    dyn_max_iterations = 0
                elif eps_k > 0.0:
                    log_success_fail = math.log(1.0 - eps_k)
                    if log_success_fail < 0.0:
                        dyn_max_iterations = int(log_fail / log_success_fail) + 1
                    else:
                        dyn_max_iterations = iterations
                else:
                    dyn_max_iterations = iterations

        if best_num_inliers == 0:
            return best_model, best_score, 0, it
        return best_model, best_score, best_num_inliers, it

    # -- final polish ------------------------------------------------------

    @quiet_low_occupancy()
    def final_refine(self, model, max_error_sq, num_iterations, loss):
        # robust-loss LM over the model's own inlier set, on device. Uses the
        # same kernel with relaxed_scale=1.0, which selects exactly the
        # inlier subset the CPU path passes in as inlier-only data.
        problem = self.problem
        buf = self._refine_buf
        buf.init_models[:1].copy_to_device(
            np.ascontiguousarray(model, dtype=np.float64).reshape(
                1, problem.num_params))
        refine_prepared(problem, self._device_data32, self._params, buf, 1,
                        max_error_sq, num_iterations, loss, relaxed_scale=1.0)
        result = buf.refined[:1].copy_to_host()
        if result[0, 0, problem.num_params] == 0.0:
            return None
        return result[0, 0, :problem.num_params].copy()
