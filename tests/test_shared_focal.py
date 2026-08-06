import numpy as np

from benchmarks.utils import (generate_relpose_data, rotation_error_deg,
                              translation_error_deg)
from fastpose.estimators.shared_focal import estimate_relative_pose_with_shared_focal
from fastpose.estimators.utils import normalize_points
from fastpose.solvers.shared_focal import SixPointSharedFocalSolver, _solve_shared_focal_6pt

FOCAL = 1000.0
PP1 = np.array([500.0, 480.0])
PP2 = np.array([620.0, 510.0])


def make_shared_focal_scene(seed, num_samples=100, noise_sigma=0.0,
                            outlier_ratio=0.0):
    rng = np.random.default_rng(seed)
    y1, y2, R, t = generate_relpose_data(
        rng, num_samples, noise_sigma=0.0, outlier_ratio=0.0,
        focal=1.0, image_size=2.0)
    x1 = FOCAL * (y1 - 1.0) + PP1
    x2 = FOCAL * (y2 - 1.0) + PP2
    x2 += rng.normal(scale=noise_sigma, size=x2.shape)
    num_outliers = int(num_samples * outlier_ratio)
    if num_outliers:
        idxs = rng.choice(num_samples, num_outliers, replace=False)
        x2[idxs, 0] = rng.uniform(0.0, 2.0 * PP2[0], size=num_outliers)
        x2[idxs, 1] = rng.uniform(0.0, 2.0 * PP2[1], size=num_outliers)
    return x1, x2, R, t


def test_shared_focal_solver_recovers_exact_pose_and_focal():
    x1, x2, R_gt, t_gt = make_shared_focal_scene(0, num_samples=20)
    x1c = x1 - PP1
    x2c = x2 - PP2
    x1n, x2n, T, scale = normalize_points(x1c, x2c)
    ppn = np.array([T[0, 2], T[1, 2]])
    data = (np.ascontiguousarray(x1n[:, 0]), np.ascontiguousarray(x1n[:, 1]),
            np.ascontiguousarray(x2n[:, 0]), np.ascontiguousarray(x2n[:, 1]),
            ppn[0], ppn[1], ppn[0], ppn[1])
    sample = np.arange(6, dtype=np.int64)
    models = np.empty((SixPointSharedFocalSolver.max_models,
                       SixPointSharedFocalSolver.num_params))
    workspace = np.empty(SixPointSharedFocalSolver.workspace_size)

    num_models = _solve_shared_focal_6pt(data, sample, models, workspace)
    assert num_models > 0

    errors = []
    for i in range(num_models):
        R = models[i, :9].reshape(3, 3)
        t = models[i, 9:12]
        f = models[i, 12] / scale
        errors.append((rotation_error_deg(R, R_gt),
                       translation_error_deg(t, t_gt),
                       abs(f - FOCAL)))
    assert min(max(r, t, f) for r, t, f in errors) < 1e-5


def test_shared_focal_estimator_recovers_exact_pose_and_focal():
    x1, x2, R_gt, t_gt = make_shared_focal_scene(1)

    R, t, f, num_inliers, inliers = estimate_relative_pose_with_shared_focal(
        x1, x2, PP1, PP2, iterations=20, min_iterations=20,
        max_error=1.0, seed=0, lo_iterations=0)

    assert num_inliers == len(x1)
    assert np.all(inliers)
    assert abs(f - FOCAL) < 1e-5
    assert rotation_error_deg(R, R_gt) < 1e-5
    assert translation_error_deg(t, t_gt) < 1e-5


def test_shared_focal_estimator_handles_default_zero_principal_points():
    x1, x2, R_gt, t_gt = make_shared_focal_scene(2)

    R, t, f, num_inliers, inliers = estimate_relative_pose_with_shared_focal(
        x1 - PP1, x2 - PP2, iterations=20, min_iterations=20,
        max_error=1.0, seed=0, lo_iterations=0)

    assert num_inliers == len(x1)
    assert abs(f - FOCAL) < 1e-5
    assert rotation_error_deg(R, R_gt) < 1e-5
    assert translation_error_deg(t, t_gt) < 1e-5


def test_shared_focal_estimator_handles_outliers_with_lo():
    x1, x2, R_gt, t_gt = make_shared_focal_scene(
        5, num_samples=500, noise_sigma=0.5, outlier_ratio=0.2)

    R, t, f, num_inliers, inliers = estimate_relative_pose_with_shared_focal(
        x1, x2, PP1, PP2, iterations=100, min_iterations=100,
        max_error=2.0, seed=0)

    assert R is not None
    assert num_inliers > 350
    assert abs(f - FOCAL) / FOCAL < 0.02
    assert rotation_error_deg(R, R_gt) < 0.5
    assert translation_error_deg(t, t_gt) < 1.0
