"""Tests for the batched parallel RANSAC driver (estimators/ransac.py).

The serial driver is covered by the per-problem estimator tests; these pin the
two properties that are specific to `build_parallel_ransac` and are otherwise
only asserted in prose: it refines at most one local-optimization candidate
per batch, and its result depends on the batch size but never on how many
threads execute that batch.
"""

import numpy as np
import numba
import pytest
from numba import njit

from benchmarks.utils import generate_relpose_data
from fastpose.estimators.essential import estimate_relative_pose
from fastpose.estimators.ransac import RansacEstimator

FOCAL = 1000.0
IMAGE_SIZE = 2000.0
PP = np.array([IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0])


# ---------------------------------------------------------------------------
# A minimal 1-D problem, used to count local-optimization triggers. The driver
# is problem-agnostic and never looks inside `data`, so the counter can simply
# ride along in the data tuple - which is the only way a numba kernel can
# write to a buffer the test can read back afterwards (globals and closure
# cells are frozen readonly). It also compiles in a fraction of the time a
# real solver would.
#
# The model is a single scalar, a sample is two points, and the score is the
# truncated distance from the model to each point.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _line_solve(data, sample, models, workspace):
    values = data[0]
    models[0, 0] = 0.5 * (values[sample[0]] + values[sample[1]])
    return 1


@njit(cache=True)
def _line_score(model, data, max_error_sq, best_score):
    values = data[0]
    score = 0.0
    num_inliers = 0
    for i in range(values.shape[0]):
        d = values[i] - model[0]
        d2 = d * d
        if d2 < max_error_sq:
            score += d2
            num_inliers += 1
        else:
            score += max_error_sq
        if score >= best_score:
            return score, num_inliers
    return score, num_inliers


@njit(cache=True)
def _counting_refine(data, model, refined, max_error_sq, num_iterations):
    # counts local-optimization triggers and refines nothing, so what the test
    # observes is the driver's LO *rule* rather than anything a refiner does
    data[1][0] += 1
    return False


class _LineSolver:
    sample_size = 2
    num_params = 1
    max_models = 1
    workspace_size = 1
    solve = staticmethod(_line_solve)


class _LineScorer:
    score = staticmethod(_line_score)


class _CountingRefiner:
    num_iterations = 25
    refine = staticmethod(_counting_refine)


def _scene(seed, num_samples=800, outlier_ratio=0.4):
    rng = np.random.default_rng(seed)
    x1, x2, _, _ = generate_relpose_data(
        rng, num_samples, noise_sigma=1.0, outlier_ratio=outlier_ratio,
        focal=FOCAL, image_size=IMAGE_SIZE)
    return (x1 - PP) / FOCAL, (x2 - PP) / FOCAL


def _line_data(seed, n=2000, inlier_ratio=0.6):
    rng = np.random.default_rng(seed)
    values = rng.uniform(-10.0, 10.0, size=n)
    num_inliers = int(n * inlier_ratio)
    values[:num_inliers] = rng.normal(0.0, 0.05, size=num_inliers)
    rng.shuffle(values)
    return (np.ascontiguousarray(values), np.zeros(1, dtype=np.int64))


def test_parallel_driver_refines_at_most_one_candidate_per_batch():
    # LO is an O(matches) refit plus a full score and stays serial in this
    # driver however many threads the batch used, so the driver refines the
    # single best qualifying hypothesis of a batch rather than every improving
    # one - mirroring what cuda/ransac.py does per round.
    data = _line_data(0)
    estimator = RansacEstimator(_LineSolver(), _LineScorer(),
                                _CountingRefiner())

    iterations = 512
    batch_per_thread = 32
    num_threads = 2
    batch_size = num_threads * batch_per_thread
    num_batches = -(-iterations // batch_size)

    data[1][0] = 0
    estimator.estimate(data, len(data[0]), 0.2, iterations=iterations,
                       min_iterations=iterations, seed=0,
                       num_threads=num_threads,
                       batch_per_thread=batch_per_thread)
    parallel_calls = int(data[1][0])

    assert parallel_calls > 0, "local optimization never ran"
    assert parallel_calls <= num_batches, (
        f"{parallel_calls} refits over {num_batches} batches")

    # and the serial driver, which has no batches, triggers it more often on
    # the same data - otherwise this would pass on a rule that never fires
    data[1][0] = 0
    estimator.estimate(data, len(data[0]), 0.2, iterations=iterations,
                       min_iterations=iterations, seed=0)
    assert int(data[1][0]) > parallel_calls


def test_parallel_result_is_independent_of_the_thread_count():
    # the driver samples serially and merges in hypothesis order precisely so
    # that a run is reproducible from (seed, batch_size); the thread count
    # must only change how fast the batch is scored, never what comes out
    x1n, x2n = _scene(1)

    if numba.config.NUMBA_NUM_THREADS < 4:
        pytest.skip("needs a pool of at least 4 threads")

    # same batch_size (64) reached with two different thread counts
    results = []
    for num_threads, batch_per_thread in ((2, 32), (4, 16)):
        model, info = estimate_relative_pose(
            x1n, x2n, iterations=512, min_iterations=512, max_error=2.0 / FOCAL,
            seed=0, num_threads=num_threads, batch_per_thread=batch_per_thread)
        results.append((model, info))

    (m_a, i_a), (m_b, i_b) = results
    np.testing.assert_array_equal(m_a['R'], m_b['R'])
    np.testing.assert_array_equal(m_a['t'], m_b['t'])
    assert i_a['num_inliers'] == i_b['num_inliers']
    assert i_a['model_score'] == i_b['model_score']


def test_parallel_driver_agrees_with_the_serial_one_on_a_clean_scene():
    # the two drivers are not bit-identical by construction, but on a scene
    # with a decisive solution they must find the same one
    x1n, x2n = _scene(2, outlier_ratio=0.2)

    serial_model, serial_info = estimate_relative_pose(
        x1n, x2n, iterations=512, min_iterations=512, max_error=2.0 / FOCAL,
        seed=0)
    parallel_model, parallel_info = estimate_relative_pose(
        x1n, x2n, iterations=512, min_iterations=512, max_error=2.0 / FOCAL,
        seed=0, num_threads=4)

    assert abs(serial_info['num_inliers'] - parallel_info['num_inliers']) <= 2
    np.testing.assert_allclose(parallel_model['R'], serial_model['R'],
                               atol=1e-6)
    np.testing.assert_allclose(parallel_model['t'], serial_model['t'],
                               atol=1e-6)
