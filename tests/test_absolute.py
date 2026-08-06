import numpy as np

from benchmarks.utils import generate_abspose_data, rotation_error_deg
from fastpose.estimators.absolute import estimate_absolute_pose
from fastpose.solvers.p3p import P3PSolver, _solve_p3p


def make_abspose_scene(seed, num_samples=100, noise_sigma=0.0,
                       outlier_ratio=0.0):
    rng = np.random.default_rng(seed)
    focal = 1000.0
    image_size = 2000.0
    x, X, R, t = generate_abspose_data(
        rng, num_samples, noise_sigma=noise_sigma,
        outlier_ratio=outlier_ratio, focal=focal, image_size=image_size)
    xn = (x - image_size / 2.0) / focal
    return xn, X, R, t


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
