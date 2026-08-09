import numpy as np
import pytest

from benchmarks.utils import (generate_monodepth_relpose_data,
                              rotation_error_deg)
from fastpose.estimators.monodepth import (
    estimate_relative_pose_with_monodepth,
    estimate_shared_focal_relative_pose_with_monodepth,
    estimate_varying_focal_relative_pose_with_monodepth)
from fastpose.refiners import monodepth as md_refiners
from fastpose.solvers import monodepth as md_solvers

FOCAL = 1000.0
FOCAL1 = 800.0
FOCAL2 = 1300.0
ALPHA1 = 0.7
ALPHA2 = 1.4
SCALE_GT = ALPHA1 / ALPHA2


def make_scene(seed, num_samples=100, noise_sigma=0.0, outlier_ratio=0.0,
               depth_noise=0.0, calibrated=True, shared_focal=True,
               with_shift=False):
    rng = np.random.default_rng(seed)
    beta1 = 0.3 if with_shift else 0.0
    beta2 = -0.2 if with_shift else 0.0
    if calibrated:
        f1 = f2 = 1.0
    elif shared_focal:
        f1 = f2 = FOCAL
    else:
        f1, f2 = FOCAL1, FOCAL2
    x1, x2, d1, d2, R, t = generate_monodepth_relpose_data(
        rng, num_samples, noise_sigma=noise_sigma, outlier_ratio=outlier_ratio,
        depth_noise=depth_noise, focal1=f1, focal2=f2,
        alpha1=ALPHA1, beta1=beta1 * ALPHA1, alpha2=ALPHA2, beta2=beta2 * ALPHA2)
    # generator shifts satisfy d + shift = alpha * z with shift = -beta/alpha
    return x1, x2, d1, d2, R, t, -beta1, -beta2


def _data_tuple(x1, x2, d1, d2):
    return (np.ascontiguousarray(x1[:, 0]), np.ascontiguousarray(x1[:, 1]),
            np.ascontiguousarray(x2[:, 0]), np.ascontiguousarray(x2[:, 1]),
            np.ascontiguousarray(d1), np.ascontiguousarray(d2),
            1.0 / 64.0, 1.0)


def _best_model_errors(models, count, R_gt, t_gt, scale_gt):
    # translation is metric here: the model t equals alpha1 * t_gt
    best = (180.0, 1e300, 1e300)
    for m in range(count):
        R = models[m, :9].reshape(3, 3)
        rot = rotation_error_deg(R, R_gt)
        terr = np.abs(models[m, 9:12] - ALPHA1 * t_gt).max()
        best = min(best, (rot, terr, m))
    return best


# ---------------------------------------------------------------------------
# minimal solvers on exact data
# ---------------------------------------------------------------------------

def test_monodepth_shift_solver_recovers_exact_model():
    x1, x2, d1, d2, R_gt, t_gt, u_gt, v_gt = make_scene(0, 10, with_shift=True)
    data = _data_tuple(x1, x2, d1, d2)
    sample = np.arange(3, dtype=np.int64)
    models = np.empty((4, md_solvers.MODEL_SIZE))
    ws = np.empty(md_solvers.MonoDepthShiftSolver.workspace_size)
    count = md_solvers._solve_monodepth_shift_3pt(data, sample, models, ws)
    assert count > 0
    errs = []
    for m in range(count):
        errs.append(max(
            rotation_error_deg(models[m, :9].reshape(3, 3), R_gt),
            np.abs(models[m, 9:12] - ALPHA1 * t_gt).max(),
            abs(models[m, 12] - SCALE_GT),
            abs(models[m, 13] - u_gt * ALPHA1),
            abs(models[m, 14] - v_gt * ALPHA2)))
    assert min(errs) < 1e-6


def test_monodepth_p3p_solver_recovers_exact_model():
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(1, 10)
    data = _data_tuple(x1, x2, d1, d2)
    sample = np.arange(3, dtype=np.int64)
    models = np.empty((4, md_solvers.MODEL_SIZE))
    ws = np.empty(md_solvers.MonoDepthP3PSolver.workspace_size)
    count = md_solvers._solve_monodepth_p3p(data, sample, models, ws)
    assert count > 0
    errs = []
    for m in range(count):
        errs.append(max(
            rotation_error_deg(models[m, :9].reshape(3, 3), R_gt),
            np.abs(models[m, 9:12] - ALPHA1 * t_gt).max(),
            abs(models[m, 12] - SCALE_GT)))
    assert min(errs) < 1e-6


def test_monodepth_shared_focal_solver_recovers_exact_model():
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(2, 10, calibrated=False)
    data = _data_tuple(x1, x2, d1, d2)
    sample = np.arange(3, dtype=np.int64)
    models = np.empty((4, md_solvers.MODEL_SIZE))
    ws = np.empty(md_solvers.MonoDepthSharedFocalSolver.workspace_size)
    count = md_solvers._solve_monodepth_shared_focal_3pt(data, sample, models, ws)
    assert count > 0
    errs = []
    for m in range(count):
        errs.append(max(
            rotation_error_deg(models[m, :9].reshape(3, 3), R_gt),
            np.abs(models[m, 9:12] - ALPHA1 * t_gt).max(),
            abs(models[m, 12] - FOCAL) / FOCAL,
            abs(models[m, 14] - SCALE_GT)))
    assert min(errs) < 1e-6


def test_monodepth_varying_focal_solver_recovers_exact_model():
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(
        3, 10, calibrated=False, shared_focal=False)
    data = _data_tuple(x1, x2, d1, d2)
    sample = np.arange(3, dtype=np.int64)
    models = np.empty((4, md_solvers.MODEL_SIZE))
    ws = np.empty(md_solvers.MonoDepthVaryingFocalSolver.workspace_size)
    count = md_solvers._solve_monodepth_varying_focal_3pt(data, sample, models, ws)
    assert count == 1
    assert rotation_error_deg(models[0, :9].reshape(3, 3), R_gt) < 1e-6
    assert np.abs(models[0, 9:12] - ALPHA1 * t_gt).max() < 1e-6
    assert abs(models[0, 12] - FOCAL1) / FOCAL1 < 1e-6
    assert abs(models[0, 13] - FOCAL2) / FOCAL2 < 1e-6
    assert abs(models[0, 14] - SCALE_GT) < 1e-6


# ---------------------------------------------------------------------------
# estimators on exact data
# ---------------------------------------------------------------------------

def test_monodepth_estimator_recovers_exact_pose():
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(4)
    R, t, scale, shift1, shift2, num_inliers, inliers = (
        estimate_relative_pose_with_monodepth(
            x1, x2, d1, d2, iterations=20, min_iterations=20,
            max_error=0.001, seed=0, lo_iterations=0))
    assert num_inliers == len(x1)
    assert np.all(inliers)
    assert rotation_error_deg(R, R_gt) < 1e-5
    assert abs(scale - SCALE_GT) < 1e-6
    assert shift1 == 0.0 and shift2 == 0.0
    assert np.abs(t - ALPHA1 * t_gt).max() < 1e-6


def test_monodepth_estimator_recovers_exact_shifts():
    x1, x2, d1, d2, R_gt, t_gt, u_gt, v_gt = make_scene(5, with_shift=True)
    R, t, scale, shift1, shift2, num_inliers, inliers = (
        estimate_relative_pose_with_monodepth(
            x1, x2, d1, d2, estimate_shift=True, iterations=20,
            min_iterations=20, max_error=0.001, seed=0, lo_iterations=0))
    assert num_inliers == len(x1)
    assert rotation_error_deg(R, R_gt) < 1e-5
    assert abs(scale - SCALE_GT) < 1e-6
    assert abs(shift1 - u_gt * ALPHA1) < 1e-5
    assert abs(shift2 - v_gt * ALPHA2) < 1e-5


def test_monodepth_shared_focal_estimator_recovers_exact_model():
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(6, calibrated=False)
    R, t, f, scale, num_inliers, inliers = (
        estimate_shared_focal_relative_pose_with_monodepth(
            x1, x2, d1, d2, iterations=20, min_iterations=20,
            max_error=1.0, seed=0, lo_iterations=0))
    assert num_inliers == len(x1)
    assert rotation_error_deg(R, R_gt) < 1e-5
    assert abs(f - FOCAL) / FOCAL < 1e-6
    assert abs(scale - SCALE_GT) < 1e-6


def test_monodepth_varying_focal_estimator_recovers_exact_model():
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(
        7, calibrated=False, shared_focal=False)
    pp1 = np.array([500.0, 480.0])
    pp2 = np.array([620.0, 510.0])
    R, t, f1, f2, scale, num_inliers, inliers = (
        estimate_varying_focal_relative_pose_with_monodepth(
            x1 + pp1, x2 + pp2, d1, d2, pp1, pp2, iterations=20,
            min_iterations=20, max_error=1.0, seed=0, lo_iterations=0))
    assert num_inliers == len(x1)
    assert rotation_error_deg(R, R_gt) < 1e-5
    assert abs(f1 - FOCAL1) / FOCAL1 < 1e-6
    assert abs(f2 - FOCAL2) / FOCAL2 < 1e-6
    assert abs(scale - SCALE_GT) < 1e-6


# ---------------------------------------------------------------------------
# estimators with noise + outliers + local optimization
# ---------------------------------------------------------------------------

def test_monodepth_estimator_handles_outliers_with_lo():
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(
        8, num_samples=500, noise_sigma=0.5 / FOCAL, outlier_ratio=0.3,
        depth_noise=0.01)
    R, t, scale, shift1, shift2, num_inliers, inliers = (
        estimate_relative_pose_with_monodepth(
            x1, x2, d1, d2, iterations=100, min_iterations=100,
            max_error=2.0 / FOCAL, max_reproj_error=16.0 / FOCAL, seed=0))
    assert R is not None
    assert num_inliers > 300
    assert rotation_error_deg(R, R_gt) < 0.5
    assert abs(scale - SCALE_GT) / SCALE_GT < 0.05


def test_monodepth_estimator_camera_matrix_recovers_exact_pose():
    # calibrated=False, shared_focal=True with zero principal points gives a
    # scene in pixel coordinates for a single shared square-pixel focal
    # length, so the manual x/FOCAL calibration and the camera1/camera2 path
    # should agree exactly
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(
        30, calibrated=False, shared_focal=True)
    K = np.array([[FOCAL, 0.0, 0.0], [0.0, FOCAL, 0.0], [0.0, 0.0, 1.0]])

    R, t, scale, shift1, shift2, num_inliers, inliers = (
        estimate_relative_pose_with_monodepth(
            x1, x2, d1, d2, camera1=K, camera2=K, iterations=20,
            min_iterations=20, max_error=0.001 * FOCAL, seed=0,
            lo_iterations=0))

    assert num_inliers == len(x1)
    assert np.all(inliers)
    assert rotation_error_deg(R, R_gt) < 1e-5
    assert abs(scale - SCALE_GT) < 1e-6
    assert np.abs(t - ALPHA1 * t_gt).max() < 1e-6


def test_monodepth_estimator_poselib_camera_recovers_exact_pose():
    poselib = pytest.importorskip('poselib')
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(
        31, calibrated=False, shared_focal=True)
    camera = poselib.Camera('PINHOLE', [FOCAL, FOCAL, 0.0, 0.0], 2000, 2000)

    R, t, scale, shift1, shift2, num_inliers, inliers = (
        estimate_relative_pose_with_monodepth(
            x1, x2, d1, d2, camera1=camera, camera2=camera, iterations=20,
            min_iterations=20, max_error=0.001 * FOCAL, seed=0,
            lo_iterations=0))

    assert num_inliers == len(x1)
    assert np.all(inliers)
    assert rotation_error_deg(R, R_gt) < 1e-5
    assert abs(scale - SCALE_GT) < 1e-6
    assert np.abs(t - ALPHA1 * t_gt).max() < 1e-6


def test_monodepth_shift_estimator_handles_outliers_with_lo():
    x1, x2, d1, d2, R_gt, t_gt, u_gt, v_gt = make_scene(
        9, num_samples=500, noise_sigma=0.5 / FOCAL, outlier_ratio=0.3,
        depth_noise=0.01, with_shift=True)
    R, t, scale, shift1, shift2, num_inliers, inliers = (
        estimate_relative_pose_with_monodepth(
            x1, x2, d1, d2, estimate_shift=True, iterations=100,
            min_iterations=100, max_error=2.0 / FOCAL,
            max_reproj_error=16.0 / FOCAL, seed=0))
    assert R is not None
    assert num_inliers > 300
    assert rotation_error_deg(R, R_gt) < 0.5
    assert abs(scale - SCALE_GT) / SCALE_GT < 0.05


def test_monodepth_shared_focal_estimator_handles_outliers_with_lo():
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(
        10, num_samples=500, noise_sigma=0.5, outlier_ratio=0.3,
        depth_noise=0.01, calibrated=False)
    R, t, f, scale, num_inliers, inliers = (
        estimate_shared_focal_relative_pose_with_monodepth(
            x1, x2, d1, d2, iterations=100, min_iterations=100,
            max_error=2.0, seed=0))
    assert R is not None
    assert num_inliers > 300
    assert rotation_error_deg(R, R_gt) < 0.5
    assert abs(f - FOCAL) / FOCAL < 0.05
    assert abs(scale - SCALE_GT) / SCALE_GT < 0.05


def test_monodepth_varying_focal_estimator_handles_outliers_with_lo():
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(
        11, num_samples=500, noise_sigma=0.5, outlier_ratio=0.3,
        depth_noise=0.01, calibrated=False, shared_focal=False)
    R, t, f1, f2, scale, num_inliers, inliers = (
        estimate_varying_focal_relative_pose_with_monodepth(
            x1, x2, d1, d2, iterations=100, min_iterations=100,
            max_error=2.0, seed=0))
    assert R is not None
    assert num_inliers > 300
    assert rotation_error_deg(R, R_gt) < 0.5
    assert abs(f1 - FOCAL1) / FOCAL1 < 0.05
    assert abs(f2 - FOCAL2) / FOCAL2 < 0.05
    assert abs(scale - SCALE_GT) / SCALE_GT < 0.05


# ---------------------------------------------------------------------------
# finite-difference check of the hybrid refiner normal equations
# ---------------------------------------------------------------------------

def _hybrid_residuals(model, x1, x2, d1, d2, scale_reproj, weight_sampson,
                      focal_mode):
    # numpy reference of the full (untruncated) hybrid residual vector with
    # the sqrt weights folded in, in accumulation order per point:
    # [sqrt(sr) * forward(2), sqrt(sr) * backward(2), sqrt(ws) * sampson(1)]
    R = model[:9].reshape(3, 3)
    t = model[9:12]
    if focal_mode:
        f1, f2, s = model[12], model[13], model[14]
        u = v = 0.0
    else:
        f1 = f2 = 1.0
        s, u, v = model[12], model[13], model[14]
    E = np.array([[0.0, -t[2], t[1]],
                  [t[2], 0.0, -t[0]],
                  [-t[1], t[0], 0.0]]) @ R
    F = np.diag([1.0, 1.0, f2]) @ E @ np.diag([1.0, 1.0, f1])

    sr = np.sqrt(scale_reproj)
    ws = np.sqrt(weight_sampson)
    res = []
    for i in range(len(x1)):
        b1 = np.array([x1[i, 0] / f1, x1[i, 1] / f1, 1.0])
        b2 = np.array([x2[i, 0] / f2, x2[i, 1] / f2, 1.0])
        Z1 = R @ ((d1[i] + u) * b1) + t
        assert Z1[2] > 0.0
        res.extend(sr * (f2 * Z1[:2] / Z1[2] - x2[i]))
        Z2 = R.T @ (s * (d2[i] + v) * b2 - t)
        assert Z2[2] > 0.0
        res.extend(sr * (f1 * Z2[:2] / Z2[2] - x1[i]))

        x1h = np.array([x1[i, 0], x1[i, 1], 1.0])
        x2h = np.array([x2[i, 0], x2[i, 1], 1.0])
        C = x2h @ F @ x1h
        Fx1 = F @ x1h
        Ftx2 = F.T @ x2h
        den = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2
        res.append(ws * C / np.sqrt(den))
    return np.array(res)


MD_CASES = {
    'monodepth': dict(
        accumulate=md_refiners._accumulate_calibrated,
        apply_step=md_refiners._apply_step_calibrated,
        num_tangent=7, focal_mode=False),
    'monodepth-shift': dict(
        accumulate=md_refiners._accumulate_calibrated_shift,
        apply_step=md_refiners._apply_step_calibrated,
        num_tangent=9, focal_mode=False),
    'monodepth-shared-focal': dict(
        accumulate=md_refiners._accumulate_shared_focal,
        apply_step=md_refiners._apply_step_shared_focal,
        num_tangent=8, focal_mode=True),
    'monodepth-varying-focal': dict(
        accumulate=md_refiners._accumulate_varying_focal,
        apply_step=md_refiners._apply_step_varying_focal,
        num_tangent=9, focal_mode=True),
}


@pytest.mark.parametrize('name', sorted(MD_CASES))
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_monodepth_normal_equations_match_finite_differences(name, seed):
    case = MD_CASES[name]
    focal_mode = case['focal_mode']
    num_tangent = case['num_tangent']

    # a mildly perturbed exact scene so all residuals are finite and small
    # but nonzero, all cheirality checks pass, and nothing is truncated
    shared = name == 'monodepth-shared-focal'
    x1, x2, d1, d2, R_gt, t_gt, _, _ = make_scene(
        20 + seed, num_samples=40,
        noise_sigma=0.5 if focal_mode else 0.5 / FOCAL,
        depth_noise=0.01, calibrated=not focal_mode, shared_focal=shared,
        with_shift=(name == 'monodepth-shift'))
    scale_reproj = 1.0 / 64.0
    weight_sampson = 0.7
    data = (np.ascontiguousarray(x1[:, 0]), np.ascontiguousarray(x1[:, 1]),
            np.ascontiguousarray(x2[:, 0]), np.ascontiguousarray(x2[:, 1]),
            np.ascontiguousarray(d1), np.ascontiguousarray(d2),
            scale_reproj, weight_sampson)

    model = np.empty(md_refiners.MODEL_SIZE)
    model[:9] = R_gt.ravel()
    model[9:12] = ALPHA1 * t_gt
    if focal_mode:
        model[12] = FOCAL if shared else FOCAL1
        model[13] = FOCAL if shared else FOCAL2
        model[14] = SCALE_GT
    else:
        model[12] = SCALE_GT
        model[13] = 0.0
        model[14] = 0.0
    state = model.copy()

    JtJ = np.empty((num_tangent, num_tangent))
    Jtr = np.empty(num_tangent)
    num_residuals = case['accumulate'](data, model, state, JtJ, Jtr, 1e30)
    assert num_residuals == 5 * len(x1)

    def residuals_at(delta):
        state_new = np.empty(md_refiners.MODEL_SIZE)
        case['apply_step'](state, delta, state_new)
        model_new = np.empty(md_refiners.MODEL_SIZE)
        md_refiners._state_to_model(state_new, model_new)
        return _hybrid_residuals(model_new, x1, x2, d1, d2, scale_reproj,
                                 weight_sampson, focal_mode)

    h = 1e-6
    r0 = residuals_at(np.zeros(num_tangent))
    J_fd = np.empty((len(r0), num_tangent))
    delta = np.zeros(num_tangent)
    for p in range(num_tangent):
        step = h * max(1.0, abs(state[12 + min(p, 2)])) if focal_mode and p >= 7 else h
        delta[p] = step
        r_plus = residuals_at(delta)
        delta[p] = -step
        r_minus = residuals_at(delta)
        delta[p] = 0.0
        J_fd[:, p] = (r_plus - r_minus) / (2.0 * step)

    scale = max(1.0, np.abs(JtJ).max())
    np.testing.assert_allclose(Jtr, J_fd.T @ r0, rtol=2e-4,
                               atol=1e-6 * np.sqrt(scale))
    np.testing.assert_allclose(JtJ, J_fd.T @ J_fd, rtol=2e-4,
                               atol=1e-6 * scale)
