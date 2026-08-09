import numpy as np

from benchmarks.utils import generate_data
from fastpose.estimators.fundamental import estimate_fundamental
from fastpose.scorers.sampson import SampsonScorer


def test_fundamental_estimator_recovers_exact_model():
    rng = np.random.default_rng(0)
    x1, x2 = generate_data(rng, 100, noise_sigma=0.0, outlier_ratio=0.0,
                           image_size=1000.0)

    model, info = estimate_fundamental(
        x1, x2, iterations=30, min_iterations=30, max_error=1e-4, seed=0,
        lo_iterations=0)

    assert info['num_inliers'] == len(x1)
    assert np.all(info['inliers'])
    _, inliers, num_inliers = SampsonScorer.score_numpy(model['F'], x1, x2, 1e-3)
    assert num_inliers == len(x1)


def test_fundamental_estimator_handles_outliers_with_lo():
    rng = np.random.default_rng(1)
    x1, x2 = generate_data(rng, 500, noise_sigma=1.0, outlier_ratio=0.3,
                           image_size=1000.0)

    model, info = estimate_fundamental(
        x1, x2, iterations=200, min_iterations=200, max_error=2.0, seed=0)

    assert info['num_inliers'] > 300


def test_fundamental_estimator_info_fields():
    rng = np.random.default_rng(2)
    x1, x2 = generate_data(rng, 100, noise_sigma=0.0, outlier_ratio=0.0,
                           image_size=1000.0)

    model, info = estimate_fundamental(
        x1, x2, iterations=30, min_iterations=30, max_error=1e-4, seed=0)

    assert set(model) == {'F'}
    assert model['F'].shape == (3, 3)
    assert set(info) == {'inliers', 'num_inliers', 'model_score',
                         'iterations', 'refinements'}
    assert info['inliers'].dtype == np.bool_
    assert np.count_nonzero(info['inliers']) == info['num_inliers']
    assert info['iterations'] > 0
    assert info['refinements'] in (0, 1)


def test_fundamental_estimator_failure_returns_generic_placeholder():
    rng = np.random.default_rng(3)
    # max_error=0.0 means no correspondence can ever satisfy the inlier
    # threshold (not even a minimal sample's own points, whose residual is
    # only zero up to floating-point rounding), so RANSAC always fails
    x1 = rng.uniform(0.0, 1000.0, size=(20, 2))
    x2 = rng.uniform(0.0, 1000.0, size=(20, 2))

    model, info = estimate_fundamental(
        x1, x2, iterations=20, min_iterations=20, max_error=0.0, seed=0)

    assert info['num_inliers'] == 0
    assert info['refinements'] == 0
    assert not np.any(info['inliers'])
    np.testing.assert_array_equal(model['F'], np.zeros((3, 3)))
