import numpy as np

from benchmarks.utils import generate_abspose_data, rotation_error_deg
from estimators.absolute_focal import estimate_absolute_pose_with_focal
from solvers.p4pf import P4PFSolver, _solve_p4pf

FOCAL = 1200.0
IMAGE_SIZE = 2000.0


def make_abspose_focal_scene(seed, num_samples=100, noise_sigma=0.0,
                             outlier_ratio=0.0):
    rng = np.random.default_rng(seed)
    x, X, R, t = generate_abspose_data(
        rng, num_samples, noise_sigma=noise_sigma,
        outlier_ratio=outlier_ratio, focal=FOCAL, image_size=IMAGE_SIZE)
    return x - IMAGE_SIZE / 2.0, X, R, t


def test_p4pf_solver_recovers_exact_pose_and_focal():
    xc, X, R_gt, t_gt = make_abspose_focal_scene(0, num_samples=4)
    data = (np.ascontiguousarray(xc[:, 0]), np.ascontiguousarray(xc[:, 1]),
            np.ascontiguousarray(X[:, 0]), np.ascontiguousarray(X[:, 1]),
            np.ascontiguousarray(X[:, 2]))
    sample = np.arange(4, dtype=np.int64)
    models = np.empty((P4PFSolver.max_models, P4PFSolver.num_params))
    workspace = np.empty(P4PFSolver.workspace_size)

    num_models = _solve_p4pf(data, sample, models, workspace)
    assert num_models == 1

    R = models[0, :9].reshape(3, 3)
    t = models[0, 9:12]
    f = models[0, 12]
    assert rotation_error_deg(R, R_gt) < 1e-4
    assert np.linalg.norm(t - t_gt) < 1e-6
    assert abs(f - FOCAL) < 0.01
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_absolute_focal_estimator_recovers_exact_pose():
    xc, X, R_gt, t_gt = make_abspose_focal_scene(1)

    R, t, f, num_inliers, inliers = estimate_absolute_pose_with_focal(
        xc, X, iterations=30, min_iterations=30, max_error=1.0,
        seed=0, lo_iterations=0)

    assert num_inliers == len(xc)
    assert np.all(inliers)
    assert abs(f - FOCAL) < 1e-3
    assert rotation_error_deg(R, R_gt) < 1e-4
    assert np.linalg.norm(t - t_gt) < 1e-4


def test_absolute_focal_estimator_handles_principal_point():
    xc, X, R_gt, t_gt = make_abspose_focal_scene(2)
    pp = np.array([640.0, 520.0])

    R, t, f, num_inliers, inliers = estimate_absolute_pose_with_focal(
        xc + pp, X, principal_point=pp, iterations=30, min_iterations=30,
        max_error=1.0, seed=0, lo_iterations=0)

    assert num_inliers == len(xc)
    assert abs(f - FOCAL) < 1e-3
    assert rotation_error_deg(R, R_gt) < 1e-4


def test_absolute_focal_estimator_handles_outliers_with_lo():
    xc, X, R_gt, t_gt = make_abspose_focal_scene(3, num_samples=500,
                                                 noise_sigma=1.0,
                                                 outlier_ratio=0.3)

    R, t, f, num_inliers, inliers = estimate_absolute_pose_with_focal(
        xc, X, iterations=100, min_iterations=100, max_error=2.0, seed=0)

    assert R is not None
    assert num_inliers > 280
    assert abs(f - FOCAL) / FOCAL < 0.02
    assert rotation_error_deg(R, R_gt) < 0.5
    assert np.linalg.norm(t - t_gt) < 0.1
