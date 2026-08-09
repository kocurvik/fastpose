import numpy as np
import pytest

from benchmarks.utils import (generate_relpose_data, rotation_error_deg,
                              translation_error_deg)
from fastpose.estimators.essential import estimate_relative_pose

FOCAL = 1000.0
IMAGE_SIZE = 2000.0
PP = np.array([IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0])


def make_relpose_scene(seed, num_samples=100, noise_sigma=0.0,
                       outlier_ratio=0.0):
    rng = np.random.default_rng(seed)
    x1, x2, R, t = generate_relpose_data(
        rng, num_samples, noise_sigma=noise_sigma, outlier_ratio=outlier_ratio,
        focal=FOCAL, image_size=IMAGE_SIZE)
    return x1, x2, R, t


def test_essential_estimator_recovers_exact_pose():
    x1, x2, R_gt, t_gt = make_relpose_scene(0)
    x1n = (x1 - PP) / FOCAL
    x2n = (x2 - PP) / FOCAL

    model, info = estimate_relative_pose(
        x1n, x2n, iterations=30, min_iterations=30, max_error=1e-3,
        seed=0, lo_iterations=0)

    assert info['num_inliers'] == len(x1n)
    assert np.all(info['inliers'])
    assert rotation_error_deg(model['R'], R_gt) < 1e-4
    assert translation_error_deg(model['t'], t_gt) < 1e-4


def test_essential_estimator_camera_matrix_recovers_exact_pose():
    # same scene, but x1/x2 passed in pixel coordinates with camera1/camera2
    # doing the unprojection that the baseline test does manually
    x1, x2, R_gt, t_gt = make_relpose_scene(0)
    K = np.array([[FOCAL, 0.0, PP[0]], [0.0, FOCAL, PP[1]], [0.0, 0.0, 1.0]])

    model, info = estimate_relative_pose(
        x1, x2, camera1=K, camera2=K, iterations=30, min_iterations=30,
        max_error=1e-3 * FOCAL, seed=0, lo_iterations=0)

    assert info['num_inliers'] == len(x1)
    assert np.all(info['inliers'])
    assert rotation_error_deg(model['R'], R_gt) < 1e-4
    assert translation_error_deg(model['t'], t_gt) < 1e-4


def test_essential_estimator_poselib_camera_recovers_exact_pose():
    poselib = pytest.importorskip('poselib')
    x1, x2, R_gt, t_gt = make_relpose_scene(0)
    camera = poselib.Camera('PINHOLE', [FOCAL, FOCAL, PP[0], PP[1]],
                            int(IMAGE_SIZE), int(IMAGE_SIZE))

    model, info = estimate_relative_pose(
        x1, x2, camera1=camera, camera2=camera, iterations=30,
        min_iterations=30, max_error=1e-3 * FOCAL, seed=0, lo_iterations=0)

    assert info['num_inliers'] == len(x1)
    assert np.all(info['inliers'])
    assert rotation_error_deg(model['R'], R_gt) < 1e-4
    assert translation_error_deg(model['t'], t_gt) < 1e-4


def test_essential_estimator_camera_matrix_handles_outliers_with_lo():
    x1, x2, R_gt, t_gt = make_relpose_scene(
        1, num_samples=500, noise_sigma=0.5, outlier_ratio=0.3)
    K = np.array([[FOCAL, 0.0, PP[0]], [0.0, FOCAL, PP[1]], [0.0, 0.0, 1.0]])

    model, info = estimate_relative_pose(
        x1, x2, camera1=K, camera2=K, iterations=200, min_iterations=200,
        max_error=2.0, seed=0)

    assert info['num_inliers'] > 300
    assert rotation_error_deg(model['R'], R_gt) < 1.0
    assert translation_error_deg(model['t'], t_gt) < 1.0


def test_essential_estimator_info_fields():
    x1, x2, R_gt, t_gt = make_relpose_scene(2, num_samples=200)
    x1n = (x1 - PP) / FOCAL
    x2n = (x2 - PP) / FOCAL

    model, info = estimate_relative_pose(
        x1n, x2n, iterations=50, min_iterations=50, max_error=1e-3, seed=0)

    assert set(info) == {'inliers', 'num_inliers', 'model_score',
                         'iterations', 'refinements'}
    assert info['inliers'].dtype == np.bool_
    assert info['inliers'].shape == (len(x1n),)
    assert np.count_nonzero(info['inliers']) == info['num_inliers']
    assert info['model_score'] >= 0.0
    assert info['iterations'] > 0
    assert info['refinements'] in (0, 1)


def test_essential_estimator_failure_returns_generic_pose():
    rng = np.random.default_rng(3)
    # max_error=0.0 means no correspondence can ever satisfy the inlier
    # threshold (not even a minimal sample's own points, whose residual is
    # only zero up to floating-point rounding), so RANSAC always fails
    x1n = rng.uniform(-0.5, 0.5, size=(20, 2))
    x2n = rng.uniform(-0.5, 0.5, size=(20, 2))

    model, info = estimate_relative_pose(
        x1n, x2n, iterations=20, min_iterations=20, max_error=0.0, seed=0)

    assert info['num_inliers'] == 0
    assert info['refinements'] == 0
    assert not np.any(info['inliers'])
    np.testing.assert_array_equal(model['R'], np.eye(3))
    np.testing.assert_array_equal(model['t'], np.zeros(3))
