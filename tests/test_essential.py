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

    R, t, num_inliers, inliers = estimate_relative_pose(
        x1n, x2n, iterations=30, min_iterations=30, max_error=1e-3,
        seed=0, lo_iterations=0)

    assert num_inliers == len(x1n)
    assert np.all(inliers)
    assert rotation_error_deg(R, R_gt) < 1e-4
    assert translation_error_deg(t, t_gt) < 1e-4


def test_essential_estimator_camera_matrix_recovers_exact_pose():
    # same scene, but x1/x2 passed in pixel coordinates with camera1/camera2
    # doing the unprojection that the baseline test does manually
    x1, x2, R_gt, t_gt = make_relpose_scene(0)
    K = np.array([[FOCAL, 0.0, PP[0]], [0.0, FOCAL, PP[1]], [0.0, 0.0, 1.0]])

    R, t, num_inliers, inliers = estimate_relative_pose(
        x1, x2, camera1=K, camera2=K, iterations=30, min_iterations=30,
        max_error=1e-3 * FOCAL, seed=0, lo_iterations=0)

    assert num_inliers == len(x1)
    assert np.all(inliers)
    assert rotation_error_deg(R, R_gt) < 1e-4
    assert translation_error_deg(t, t_gt) < 1e-4


def test_essential_estimator_poselib_camera_recovers_exact_pose():
    poselib = pytest.importorskip('poselib')
    x1, x2, R_gt, t_gt = make_relpose_scene(0)
    camera = poselib.Camera('PINHOLE', [FOCAL, FOCAL, PP[0], PP[1]],
                            int(IMAGE_SIZE), int(IMAGE_SIZE))

    R, t, num_inliers, inliers = estimate_relative_pose(
        x1, x2, camera1=camera, camera2=camera, iterations=30,
        min_iterations=30, max_error=1e-3 * FOCAL, seed=0, lo_iterations=0)

    assert num_inliers == len(x1)
    assert np.all(inliers)
    assert rotation_error_deg(R, R_gt) < 1e-4
    assert translation_error_deg(t, t_gt) < 1e-4


def test_essential_estimator_camera_matrix_handles_outliers_with_lo():
    x1, x2, R_gt, t_gt = make_relpose_scene(
        1, num_samples=500, noise_sigma=0.5, outlier_ratio=0.3)
    K = np.array([[FOCAL, 0.0, PP[0]], [0.0, FOCAL, PP[1]], [0.0, 0.0, 1.0]])

    R, t, num_inliers, inliers = estimate_relative_pose(
        x1, x2, camera1=K, camera2=K, iterations=200, min_iterations=200,
        max_error=2.0, seed=0)

    assert R is not None
    assert num_inliers > 300
    assert rotation_error_deg(R, R_gt) < 1.0
    assert translation_error_deg(t, t_gt) < 1.0
