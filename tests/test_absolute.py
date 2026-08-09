import numpy as np
import pytest

from benchmarks.utils import generate_abspose_data, rotation_error_deg
from fastpose.estimators.absolute import estimate_absolute_pose
from fastpose.solvers.p3p import P3PSolver, _solve_p3p

FOCAL = 1000.0
IMAGE_SIZE = 2000.0
PP = np.array([IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0])


def make_abspose_scene(seed, num_samples=100, noise_sigma=0.0,
                       outlier_ratio=0.0):
    rng = np.random.default_rng(seed)
    x, X, R, t = generate_abspose_data(
        rng, num_samples, noise_sigma=noise_sigma,
        outlier_ratio=outlier_ratio, focal=FOCAL, image_size=IMAGE_SIZE)
    xn = (x - PP) / FOCAL
    return xn, X, R, t


def make_abspose_scene_pixels(seed, num_samples=100, noise_sigma=0.0,
                              outlier_ratio=0.0):
    rng = np.random.default_rng(seed)
    x, X, R, t = generate_abspose_data(
        rng, num_samples, noise_sigma=noise_sigma,
        outlier_ratio=outlier_ratio, focal=FOCAL, image_size=IMAGE_SIZE)
    return x, X, R, t


def test_p3p_solver_recovers_exact_pose():
    xn, X, R_gt, t_gt = make_abspose_scene(0, num_samples=3)
    data = (np.ascontiguousarray(xn[:, 0]), np.ascontiguousarray(xn[:, 1]),
            np.ascontiguousarray(X[:, 0]), np.ascontiguousarray(X[:, 1]),
            np.ascontiguousarray(X[:, 2]))
    sample = np.arange(3, dtype=np.int64)
    models = np.empty((P3PSolver.max_models, P3PSolver.num_params))
    workspace = np.empty(P3PSolver.workspace_size)

    num_models = _solve_p3p(data, sample, models, workspace)
    assert num_models > 0

    best = min(rotation_error_deg(models[m, :9].reshape(3, 3), R_gt)
               + np.linalg.norm(models[m, 9:12] - t_gt)
               for m in range(num_models))
    assert best < 1e-6


def test_absolute_estimator_recovers_exact_pose():
    xn, X, R_gt, t_gt = make_abspose_scene(1)

    R, t, num_inliers, inliers = estimate_absolute_pose(
        xn, X, iterations=30, min_iterations=30, max_error=1e-3,
        seed=0, lo_iterations=0)

    assert num_inliers == len(xn)
    assert np.all(inliers)
    assert rotation_error_deg(R, R_gt) < 1e-4
    assert np.linalg.norm(t - t_gt) < 1e-4


def test_absolute_estimator_camera_matrix_recovers_exact_pose():
    x, X, R_gt, t_gt = make_abspose_scene_pixels(1)
    K = np.array([[FOCAL, 0.0, PP[0]], [0.0, FOCAL, PP[1]], [0.0, 0.0, 1.0]])

    R, t, num_inliers, inliers = estimate_absolute_pose(
        x, X, camera=K, iterations=30, min_iterations=30,
        max_error=1e-3 * FOCAL, seed=0, lo_iterations=0)

    assert num_inliers == len(x)
    assert np.all(inliers)
    assert rotation_error_deg(R, R_gt) < 1e-4
    assert np.linalg.norm(t - t_gt) < 1e-4


def test_absolute_estimator_poselib_camera_recovers_exact_pose():
    poselib = pytest.importorskip('poselib')
    x, X, R_gt, t_gt = make_abspose_scene_pixels(1)
    camera = poselib.Camera('PINHOLE', [FOCAL, FOCAL, PP[0], PP[1]],
                            int(IMAGE_SIZE), int(IMAGE_SIZE))

    R, t, num_inliers, inliers = estimate_absolute_pose(
        x, X, camera=camera, iterations=30, min_iterations=30,
        max_error=1e-3 * FOCAL, seed=0, lo_iterations=0)

    assert num_inliers == len(x)
    assert np.all(inliers)
    assert rotation_error_deg(R, R_gt) < 1e-4
    assert np.linalg.norm(t - t_gt) < 1e-4


def test_absolute_estimator_handles_outliers_with_lo():
    xn, X, R_gt, t_gt = make_abspose_scene(2, num_samples=500,
                                           noise_sigma=1.0, outlier_ratio=0.3)

    R, t, num_inliers, inliers = estimate_absolute_pose(
        xn, X, iterations=100, min_iterations=100, max_error=2.0 / 1000.0,
        seed=0)

    assert R is not None
    assert num_inliers > 300
    assert rotation_error_deg(R, R_gt) < 0.2
    assert np.linalg.norm(t - t_gt) < 0.05


def test_absolute_estimator_lo_improves_over_plain_ransac():
    xn, X, R_gt, t_gt = make_abspose_scene(3, num_samples=500,
                                           noise_sigma=2.0, outlier_ratio=0.4)

    errs = {}
    for label, lo in (('plain', 0), ('lo', None)):
        R, t, num_inliers, inliers = estimate_absolute_pose(
            xn, X, iterations=50, min_iterations=50, max_error=2.0 / 1000.0,
            seed=3, lo_iterations=lo)
        assert R is not None
        errs[label] = rotation_error_deg(R, R_gt)

    assert errs['lo'] <= errs['plain'] + 1e-9
