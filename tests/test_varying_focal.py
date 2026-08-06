import numpy as np

from benchmarks.utils import (generate_relpose_data, rotation_error_deg, skew,
                              translation_error_deg)
from fastpose.estimators.varying_focal import estimate_relative_pose_with_varying_focals
from fastpose.solvers.varying_focal import bougnoux_focals_sq


def make_varying_focal_scene(seed, num_samples=100, noise_sigma=0.0,
                             outlier_ratio=0.0):
    rng = np.random.default_rng(seed)
    f1 = 800.0
    f2 = 1300.0
    pp1 = np.array([500.0, 480.0])
    pp2 = np.array([620.0, 510.0])
    y1, y2, R, t = generate_relpose_data(
        rng, num_samples, noise_sigma=0.0, outlier_ratio=0.0,
        focal=1.0, image_size=2.0)
    x1 = f1 * (y1 - 1.0) + pp1
    x2 = f2 * (y2 - 1.0) + pp2
    x2 += rng.normal(scale=noise_sigma, size=x2.shape)
    num_outliers = int(num_samples * outlier_ratio)
    if num_outliers:
        idxs = rng.choice(num_samples, num_outliers, replace=False)
        x2[idxs, 0] = rng.uniform(0.0, 2.0 * pp2[0], size=num_outliers)
        x2[idxs, 1] = rng.uniform(0.0, 2.0 * pp2[1], size=num_outliers)
    return x1, x2, R, t, f1, f2, pp1, pp2


def test_bougnoux_recovers_focals_from_exact_fundamental():
    x1, x2, R, t, f1, f2, pp1, pp2 = make_varying_focal_scene(0)
    E = skew(t) @ R
    K1 = np.array([[f1, 0.0, pp1[0]], [0.0, f1, pp1[1]], [0.0, 0.0, 1.0]])
    K2 = np.array([[f2, 0.0, pp2[0]], [0.0, f2, pp2[1]], [0.0, 0.0, 1.0]])
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    focals_sq = np.empty(2)
    assert bougnoux_focals_sq(F.ravel(), pp1[0], pp1[1], pp2[0], pp2[1],
                              focals_sq)
    np.testing.assert_allclose(np.sqrt(focals_sq), [f1, f2], rtol=1e-8)


def test_varying_focal_estimator_recovers_exact_pose_and_focals():
    x1, x2, R_gt, t_gt, f1_gt, f2_gt, pp1, pp2 = make_varying_focal_scene(1)

    R, t, f1, f2, num_inliers, inliers = estimate_relative_pose_with_varying_focals(
        x1, x2, pp1, pp2, iterations=30, min_iterations=30,
        max_error=1.0, seed=0, lo_iterations=0)

    assert num_inliers == len(x1)
    assert np.all(inliers)
    assert abs(f1 - f1_gt) < 1e-5
    assert abs(f2 - f2_gt) < 1e-5
    assert rotation_error_deg(R, R_gt) < 1e-5
    assert translation_error_deg(t, t_gt) < 1e-5


def test_varying_focal_estimator_handles_default_zero_principal_points():
    x1, x2, R_gt, t_gt, f1_gt, f2_gt, pp1, pp2 = make_varying_focal_scene(2)
    x1 = x1 - pp1
    x2 = x2 - pp2

    R, t, f1, f2, num_inliers, inliers = estimate_relative_pose_with_varying_focals(
        x1, x2, iterations=30, min_iterations=30, max_error=1.0,
        seed=0, lo_iterations=0)

    assert num_inliers == len(x1)
    assert abs(f1 - f1_gt) < 1e-5
    assert abs(f2 - f2_gt) < 1e-5
    assert rotation_error_deg(R, R_gt) < 1e-5
    assert translation_error_deg(t, t_gt) < 1e-5


def test_varying_focal_estimator_runs_local_optimization():
    x1, x2, R_gt, t_gt, f1_gt, f2_gt, pp1, pp2 = make_varying_focal_scene(3)

    R, t, f1, f2, num_inliers, inliers = estimate_relative_pose_with_varying_focals(
        x1, x2, pp1, pp2, iterations=30, min_iterations=30,
        max_error=1.0, seed=2, lo_iterations=2)

    assert R is not None
    assert num_inliers > 50
    assert f1 > 0.0
    assert f2 > 0.0
    assert abs(f1 - f1_gt) < 1e-5
    assert abs(f2 - f2_gt) < 1e-5
    assert rotation_error_deg(R, R_gt) < 1e-5
    assert translation_error_deg(t, t_gt) < 1e-5
