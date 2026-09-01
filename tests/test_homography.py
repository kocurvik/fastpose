"""Homography estimator, scorer and solver on synthetic planar scenes.

`max_error` gates the symmetric transfer error *averaged* over the two
directions, so it is a one-way pixel distance and the thresholds here mean
what they would in poselib or OpenCV; see scorers/transfer.py.
"""

import numpy as np
import pytest

from benchmarks.utils import generate_homography_data
from fastpose.estimators.homography import estimate_homography
from fastpose.refiners.homography import LMHomographyRefiner, sphere_tangent_basis
from fastpose.scorers.transfer import (DERIVED_SIZE, SymmetricTransferScorer,
                                       homography_derived,
                                       symmetric_transfer_score)
from fastpose.solvers.homography import FourPointSolver, _solve_h4p, four_point

IMAGE_SIZE = 2000.0


def _columns(x1, x2):
    return (np.ascontiguousarray(x1[:, 0]), np.ascontiguousarray(x1[:, 1]),
            np.ascontiguousarray(x2[:, 0]), np.ascontiguousarray(x2[:, 1]))


def _model_error(H_est, H_gt):
    # both unit-normalized, minimized over the sign ambiguity
    H = np.asarray(H_est).ravel()
    H = H / np.linalg.norm(H)
    G = np.asarray(H_gt).ravel()
    G = G / np.linalg.norm(G)
    return float(min(np.linalg.norm(H - G), np.linalg.norm(H + G)))


# ---------------------------------------------------------------------------
# minimal solver
# ---------------------------------------------------------------------------

def test_four_point_solver_recovers_the_exact_homography():
    rng = np.random.default_rng(0)
    x1, x2, H_gt, _ = generate_homography_data(
        rng, 4, noise_sigma=0.0, outlier_ratio=0.0, image_size=IMAGE_SIZE,
        return_gt=True)

    models = np.zeros((FourPointSolver.max_models, FourPointSolver.num_params))
    workspace = np.empty(FourPointSolver.workspace_size)
    count = _solve_h4p(_columns(x1, x2), np.arange(4, dtype=np.int64), models,
                       workspace)

    assert count == 1
    assert _model_error(models[0], H_gt) < 1e-9
    # the solver hands back a unit-norm model, which is what keeps the
    # scorer's inverse well scaled
    np.testing.assert_allclose(np.linalg.norm(models[0]), 1.0, rtol=1e-12)


def test_four_point_solver_matches_the_numpy_reference():
    rng = np.random.default_rng(5)
    x1, x2 = generate_homography_data(rng, 40, noise_sigma=1.0,
                                      outlier_ratio=0.0,
                                      image_size=IMAGE_SIZE)
    data = _columns(x1, x2)
    models = np.zeros((1, 9))
    workspace = np.empty(FourPointSolver.workspace_size)

    for start in range(0, 36, 4):
        sample = np.arange(start, start + 4, dtype=np.int64)
        assert _solve_h4p(data, sample, models, workspace) == 1
        reference = four_point(x1[sample], x2[sample])
        assert reference is not None
        assert _model_error(models[0], reference) < 1e-7


def test_four_point_solver_rejects_a_degenerate_sample():
    # four collinear points constrain nothing beyond a line, so the DLT matrix
    # is rank deficient and the elimination must bail rather than return noise
    t = np.array([0.0, 0.3, 0.6, 0.9])
    x1 = np.column_stack([100.0 + 500.0 * t, 200.0 + 250.0 * t])
    x2 = np.column_stack([300.0 + 400.0 * t, 150.0 + 600.0 * t])

    models = np.zeros((1, 9))
    workspace = np.empty(FourPointSolver.workspace_size)
    assert _solve_h4p(_columns(x1, x2), np.arange(4, dtype=np.int64), models,
                      workspace) == 0


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------

def test_derived_form_is_the_inverse():
    rng = np.random.default_rng(11)
    H = rng.normal(size=(3, 3))
    H /= np.linalg.norm(H)
    d = np.empty(DERIVED_SIZE)

    assert homography_derived(H.ravel().copy(), d)
    np.testing.assert_allclose(d[:9], H.ravel())
    np.testing.assert_allclose(d[9:].reshape(3, 3) @ H, np.eye(3), atol=1e-12)


def test_derived_form_rejects_a_singular_homography():
    d = np.empty(DERIVED_SIZE)
    # rank 2 (two identical rows) and rank 1 (an outer product)
    rank2 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    rank1 = np.outer([1.0, 2.0, 3.0], [0.5, -1.0, 2.0]).ravel()
    assert not homography_derived(rank2, d)
    assert not homography_derived(rank1, d)
    assert not homography_derived(np.zeros(9), d)
    assert not homography_derived(np.full(9, np.nan), d)


def test_symmetric_transfer_score_matches_the_numpy_reference():
    rng = np.random.default_rng(2)
    x1, x2, H_gt, _ = generate_homography_data(
        rng, 800, noise_sigma=1.0, outlier_ratio=0.3, image_size=IMAGE_SIZE,
        return_gt=True)
    max_error = 3.0

    kernel_score, kernel_inliers = symmetric_transfer_score(
        H_gt.ravel().copy(), _columns(x1, x2), max_error ** 2, 1e300)
    numpy_score, mask, numpy_inliers = SymmetricTransferScorer.score_numpy(
        H_gt, x1, x2, max_error)

    assert kernel_inliers == numpy_inliers
    assert np.count_nonzero(mask) == numpy_inliers
    np.testing.assert_allclose(kernel_score, numpy_score, rtol=1e-12)


def test_symmetric_transfer_error_is_the_mean_of_both_directions():
    # the definition the threshold gates, spelled out against a hand-rolled
    # forward and backward transfer. The *mean*, not the sum: that is what
    # keeps max_error a one-way pixel distance, as it is in poselib
    rng = np.random.default_rng(6)
    x1, x2, H_gt, _ = generate_homography_data(
        rng, 50, noise_sigma=2.0, outlier_ratio=0.0, image_size=IMAGE_SIZE,
        return_gt=True)

    x1h = np.column_stack([x1, np.ones(len(x1))])
    x2h = np.column_stack([x2, np.ones(len(x2))])
    p = x1h @ H_gt.T
    q = x2h @ np.linalg.inv(H_gt).T
    expected = 0.5 * (np.sum((p[:, :2] / p[:, 2:3] - x2) ** 2, axis=1)
                      + np.sum((q[:, :2] / q[:, 2:3] - x1) ** 2, axis=1))

    from fastpose.scorers.transfer import symmetric_transfer_errors_numpy
    np.testing.assert_allclose(
        symmetric_transfer_errors_numpy(H_gt, x1, x2), expected, rtol=1e-10)


def test_scorer_rejects_a_singular_model():
    rng = np.random.default_rng(9)
    x1, x2 = generate_homography_data(rng, 100, noise_sigma=0.0,
                                      outlier_ratio=0.0,
                                      image_size=IMAGE_SIZE)
    singular = np.outer([1.0, 2.0, 3.0], [0.5, -1.0, 2.0]).ravel()

    score, num_inliers = symmetric_transfer_score(singular, _columns(x1, x2),
                                                  4.0, 1e300)
    assert score == 1e300
    assert num_inliers == 0
    _, _, numpy_inliers = SymmetricTransferScorer.score_numpy(
        singular.reshape(3, 3), x1, x2, 2.0)
    assert numpy_inliers == 0


# ---------------------------------------------------------------------------
# refiner
# ---------------------------------------------------------------------------

def test_tangent_basis_is_orthonormal_and_orthogonal_to_the_state():
    rng = np.random.default_rng(13)
    B = np.empty((8, 9))
    for _ in range(10):
        h = rng.normal(size=9)
        h /= np.linalg.norm(h)
        sphere_tangent_basis(h, B)
        np.testing.assert_allclose(B @ B.T, np.eye(8), atol=1e-12)
        np.testing.assert_allclose(B @ h, np.zeros(8), atol=1e-12)


def test_refiner_improves_a_perturbed_model_and_stays_on_the_sphere():
    rng = np.random.default_rng(4)
    x1, x2, H_gt, _ = generate_homography_data(
        rng, 400, noise_sigma=1.0, outlier_ratio=0.0, image_size=IMAGE_SIZE,
        return_gt=True)
    data = _columns(x1, x2)
    max_error = 4.0

    # relative perturbation: the entries of a pixel-coordinate homography span
    # several orders of magnitude, so an additive one would not be a small step
    start = (H_gt * (1.0 + 1e-3 * rng.normal(size=(3, 3)))).ravel()
    start /= np.linalg.norm(start)

    refined = np.empty(9)
    assert LMHomographyRefiner.refine(data, start, refined, max_error ** 2, 50)

    # the retraction keeps the model on the unit sphere
    np.testing.assert_allclose(np.linalg.norm(refined), 1.0, rtol=1e-12)

    score_start, _, _ = SymmetricTransferScorer.score_numpy(
        start.reshape(3, 3), x1, x2, max_error)
    score_refined, _, _ = SymmetricTransferScorer.score_numpy(
        refined.reshape(3, 3), x1, x2, max_error)
    score_gt, _, _ = SymmetricTransferScorer.score_numpy(H_gt, x1, x2,
                                                         max_error)
    assert score_refined < score_start
    assert score_refined <= score_gt * 1.05   # at least ground-truth quality


def test_refiner_does_not_degrade_a_ground_truth_start():
    rng = np.random.default_rng(8)
    x1, x2, H_gt, _ = generate_homography_data(
        rng, 400, noise_sigma=1.0, outlier_ratio=0.0, image_size=IMAGE_SIZE,
        return_gt=True)
    max_error = 4.0

    score_gt, _, _ = SymmetricTransferScorer.score_numpy(H_gt, x1, x2,
                                                         max_error)
    refined = np.empty(9)
    assert LMHomographyRefiner.refine(_columns(x1, x2), H_gt.ravel().copy(),
                                      refined, max_error ** 2, 50)
    score_refined, _, _ = SymmetricTransferScorer.score_numpy(
        refined.reshape(3, 3), x1, x2, max_error)
    assert score_refined <= score_gt * (1.0 + 1e-9)


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------

def test_homography_estimator_recovers_exact_model():
    rng = np.random.default_rng(0)
    x1, x2, H_gt, _ = generate_homography_data(
        rng, 100, noise_sigma=0.0, outlier_ratio=0.0, image_size=IMAGE_SIZE,
        return_gt=True)

    model, info = estimate_homography(
        x1, x2, iterations=30, min_iterations=30, max_error=1e-4, seed=0,
        lo_iterations=0)

    assert info['num_inliers'] == len(x1)
    assert np.all(info['inliers'])
    assert _model_error(model['H'], H_gt) < 1e-9


def test_homography_estimator_handles_outliers_with_lo():
    rng = np.random.default_rng(1)
    x1, x2, H_gt, mask = generate_homography_data(
        rng, 500, noise_sigma=1.0, outlier_ratio=0.3, image_size=IMAGE_SIZE,
        return_gt=True)

    model, info = estimate_homography(
        x1, x2, iterations=200, min_iterations=200, max_error=4.0, seed=0)

    assert info['num_inliers'] > 300
    # the recovered inlier set is essentially the true one
    assert np.mean(info['inliers'] == mask) > 0.95
    assert _model_error(model['H'], H_gt) < 0.05


def test_homography_estimator_info_fields():
    rng = np.random.default_rng(2)
    x1, x2 = generate_homography_data(rng, 100, noise_sigma=0.0,
                                      outlier_ratio=0.0,
                                      image_size=IMAGE_SIZE)

    model, info = estimate_homography(
        x1, x2, iterations=30, min_iterations=30, max_error=1e-4, seed=0)

    assert set(model) == {'H'}
    assert model['H'].shape == (3, 3)
    assert set(info) == {'inliers', 'num_inliers', 'model_score',
                         'iterations', 'refinements'}
    assert info['inliers'].dtype == np.bool_
    assert np.count_nonzero(info['inliers']) == info['num_inliers']
    assert info['iterations'] > 0
    assert info['refinements'] in (0, 1)


def test_homography_estimator_failure_returns_generic_placeholder():
    rng = np.random.default_rng(3)
    # max_error=0.0 means no correspondence can ever satisfy the inlier
    # threshold (not even a minimal sample's own points, whose residual is
    # only zero up to floating-point rounding), so RANSAC always fails
    x1 = rng.uniform(0.0, 1000.0, size=(20, 2))
    x2 = rng.uniform(0.0, 1000.0, size=(20, 2))

    model, info = estimate_homography(
        x1, x2, iterations=20, min_iterations=20, max_error=0.0, seed=0)

    assert info['num_inliers'] == 0
    assert info['refinements'] == 0
    assert not np.any(info['inliers'])
    np.testing.assert_array_equal(model['H'], np.zeros((3, 3)))


def test_homography_estimator_requires_four_points():
    rng = np.random.default_rng(4)
    x1 = rng.uniform(0.0, 1000.0, size=(3, 2))
    x2 = rng.uniform(0.0, 1000.0, size=(3, 2))
    with pytest.raises(ValueError, match="at least 4"):
        estimate_homography(x1, x2)


def test_homography_estimator_rejects_an_unknown_device():
    rng = np.random.default_rng(5)
    x1, x2 = generate_homography_data(rng, 50, image_size=IMAGE_SIZE)
    with pytest.raises(ValueError, match="device must be"):
        estimate_homography(x1, x2, device='tpu')


def test_parallel_driver_agrees_with_the_serial_one():
    # the parallel driver is not bit-identical (see build_parallel_ransac),
    # but on the same scene it must land on an equally good model
    rng = np.random.default_rng(7)
    x1, x2, H_gt, _ = generate_homography_data(
        rng, 1000, noise_sigma=1.0, outlier_ratio=0.3, image_size=IMAGE_SIZE,
        return_gt=True)
    kwargs = dict(iterations=300, min_iterations=300, max_error=4.0, seed=0)

    _, serial = estimate_homography(x1, x2, **kwargs)
    model, parallel = estimate_homography(x1, x2, num_threads=2, **kwargs)

    assert parallel['num_inliers'] >= serial['num_inliers'] * 0.95
    assert _model_error(model['H'], H_gt) < 0.05
