"""Finite-difference checks of the refiner jacobians.

The refiners feed the shared LM engine with the normal equations
(JtJ = J^T J, Jtr = J^T r) accumulated by their `_accumulate` kernels. Here
the full residual jacobian J is built by central finite differences through
the actual retraction (`apply_step`) and model reassembly (`state_to_model`)
kernels, and the accumulated normal equations are compared against
J_fd^T J_fd and J_fd^T r.

The fundamental refiner works on flat 3x3 models with the factorized
U/Vt/sigma state; the essential refiner works on pose models [R | t] whose
residuals go through E = [t]_x R; the two focal refiners work on
[R | t | f1 | f2] whose residuals go through F = K2^-T E K1^-1. The focal
cases use non-zero principal points on purpose - the analytic focal jacobian
rows carry pp-dependent terms that a pp = 0 test would leave unchecked.
"""

import numpy as np
import pytest

from fastpose.refiners import essential, fundamental, shared_focal, varying_focal
from fastpose.refiners import utils as refiner_utils

PP1 = np.array([0.3, -0.2])
PP2 = np.array([-0.1, 0.25])


def _skew(t):
    return np.array([[0.0, -t[2], t[1]],
                     [t[2], 0.0, -t[0]],
                     [-t[1], t[0], 0.0]])


def _random_f_model(rng):
    model = rng.normal(size=9)
    return model / np.linalg.norm(model)


def _random_pose_model(rng):
    u, _, vt = np.linalg.svd(rng.normal(size=(3, 3)))
    R = u @ vt
    if np.linalg.det(R) < 0:
        R = -R
    t = rng.normal(size=3)
    t /= np.linalg.norm(t)
    return np.concatenate([R.ravel(), t])


def _pose_to_e(model):
    R = model[:9].reshape(3, 3)
    t = model[9:12]
    return (_skew(t) @ R).ravel()


def _random_focal(rng):
    return float(np.exp(rng.normal(scale=0.3)))


def _random_shared_focal_model(rng):
    f = _random_focal(rng)
    return np.concatenate([_random_pose_model(rng), [f, f]])


def _random_varying_focal_model(rng):
    return np.concatenate([_random_pose_model(rng),
                           [_random_focal(rng), _random_focal(rng)]])


def _focal_model_to_f(model):
    # F = K2^-T E K1^-1, the matrix the focal refiners' residuals go through
    E = _skew(model[9:12]) @ model[:9].reshape(3, 3)
    f1, f2 = model[12], model[13]
    K1i = np.array([[1.0 / f1, 0.0, -PP1[0] / f1],
                    [0.0, 1.0 / f1, -PP1[1] / f1],
                    [0.0, 0.0, 1.0]])
    K2i = np.array([[1.0 / f2, 0.0, -PP2[0] / f2],
                    [0.0, 1.0 / f2, -PP2[1] / f2],
                    [0.0, 0.0, 1.0]])
    return (K2i.T @ E @ K1i).ravel()


_PP_DATA = (PP1[0], PP1[1], PP2[0], PP2[1])

CASES = {
    'fundamental': dict(
        init_state=fundamental._init_state,
        accumulate=fundamental._accumulate,
        apply_step=fundamental._apply_step,
        state_to_model=refiner_utils.state_to_model,
        state_size=refiner_utils.STATE_SIZE,
        num_tangent=fundamental.NUM_TANGENT,
        model_size=9,
        make_model=_random_f_model,
        model_to_e=lambda model: model,
    ),
    'essential': dict(
        init_state=essential._init_state,
        accumulate=essential._accumulate,
        apply_step=essential._apply_step,
        state_to_model=essential._state_to_model,
        state_size=essential.STATE_SIZE,
        num_tangent=essential.NUM_TANGENT,
        model_size=essential.MODEL_SIZE,
        make_model=_random_pose_model,
        model_to_e=_pose_to_e,
    ),
    'shared_focal': dict(
        init_state=shared_focal._init_state,
        accumulate=shared_focal._accumulate,
        apply_step=shared_focal._apply_step,
        state_to_model=shared_focal._state_to_model,
        state_size=shared_focal.STATE_SIZE,
        num_tangent=shared_focal.NUM_TANGENT,
        model_size=14,
        make_model=_random_shared_focal_model,
        model_to_e=_focal_model_to_f,
        extra_data=_PP_DATA,
    ),
    'varying_focal': dict(
        init_state=varying_focal._init_state,
        accumulate=varying_focal._accumulate,
        apply_step=varying_focal._apply_step,
        state_to_model=varying_focal._state_to_model,
        state_size=varying_focal.STATE_SIZE,
        num_tangent=varying_focal.NUM_TANGENT,
        model_size=14,
        make_model=_random_varying_focal_model,
        model_to_e=_focal_model_to_f,
        extra_data=_PP_DATA,
    ),
}


def sampson_residuals(e, x1, x2):
    # numpy reference of the (untruncated) Sampson residuals s_i for the
    # flat 3x3 epipolar matrix e
    F = e.reshape(3, 3)
    ones = np.ones(len(x1))
    x1h = np.column_stack([x1, ones])
    x2h = np.column_stack([x2, ones])
    Fx1 = x1h @ F.T
    Ftx2 = x2h @ F
    residual = np.sum(x2h * Fx1, axis=1)
    denominator = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return residual / np.sqrt(denominator)


def residuals_at(state, delta, case, x1, x2):
    state_new = np.empty(case['state_size'])
    case['apply_step'](state, delta, state_new)
    model = np.empty(case['model_size'])
    case['state_to_model'](state_new, model)
    return sampson_residuals(case['model_to_e'](model), x1, x2)


def fd_jacobian(state, case, x1, x2, h=1e-6):
    num_tangent = case['num_tangent']
    J = np.empty((len(x1), num_tangent))
    delta = np.zeros(num_tangent)
    for p in range(num_tangent):
        delta[p] = h
        s_plus = residuals_at(state, delta, case, x1, x2)
        delta[p] = -h
        s_minus = residuals_at(state, delta, case, x1, x2)
        delta[p] = 0.0
        J[:, p] = (s_plus - s_minus) / (2.0 * h)
    return J


def make_points(rng, n, case):
    x1 = rng.uniform(-1.0, 1.0, size=(n, 2))
    x2 = rng.uniform(-1.0, 1.0, size=(n, 2))
    data = (np.ascontiguousarray(x1[:, 0]), np.ascontiguousarray(x1[:, 1]),
            np.ascontiguousarray(x2[:, 0]), np.ascontiguousarray(x2[:, 1]))
    return x1, x2, data + case.get('extra_data', ())


def make_state(case, rng):
    model = case['make_model'](rng)
    state = np.empty(case['state_size'])
    assert case['init_state'](model, state)
    f = np.empty(case['model_size'])
    case['state_to_model'](state, f)
    return state, f


@pytest.mark.parametrize('name', sorted(CASES))
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_normal_equations_match_finite_differences(name, seed):
    case = CASES[name]
    rng = np.random.default_rng(seed)
    n = 60
    x1, x2, data = make_points(rng, n, case)
    state, f = make_state(case, rng)
    num_tangent = case['num_tangent']

    JtJ = np.empty((num_tangent, num_tangent))
    Jtr = np.empty(num_tangent)
    # huge threshold: every point contributes, no truncation kinks
    num_residuals = case['accumulate'](data, f, state, JtJ, Jtr, 1e30)
    assert num_residuals == n

    J_fd = fd_jacobian(state, case, x1, x2)
    s = sampson_residuals(case['model_to_e'](f), x1, x2)

    np.testing.assert_allclose(Jtr, J_fd.T @ s, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(JtJ, J_fd.T @ J_fd, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize('name', sorted(CASES))
def test_truncation_only_counts_inliers(name):
    # with a finite threshold the accumulated residual count must match the
    # number of points whose Sampson error is below it
    case = CASES[name]
    rng = np.random.default_rng(7)
    n = 200
    x1, x2, data = make_points(rng, n, case)
    state, f = make_state(case, rng)
    num_tangent = case['num_tangent']

    s = sampson_residuals(case['model_to_e'](f), x1, x2)
    max_error_sq = float(np.median(s ** 2))

    JtJ = np.empty((num_tangent, num_tangent))
    Jtr = np.empty(num_tangent)
    num_residuals = case['accumulate'](data, f, state, JtJ, Jtr, max_error_sq)
    assert num_residuals == int(np.count_nonzero(s ** 2 < max_error_sq))
