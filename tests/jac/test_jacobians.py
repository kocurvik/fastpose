"""Finite-difference checks of the refiner jacobians.

The refiners feed the shared LM engine with the normal equations
(JtJ = J^T J, Jtr = J^T r) accumulated by their `_accumulate` kernels. Here
the full residual jacobian J is built by central finite differences through
the actual retraction (`apply_step`) and model reassembly (`state_to_model`)
kernels, and the accumulated normal equations are compared against
J_fd^T J_fd and J_fd^T r.

The fundamental refiner works on flat 3x3 models with the factorized
U/Vt/sigma state; the essential refiner works on pose models [R | t] whose
residuals go through E = [t]_x R.
"""

import numpy as np
import pytest

from fastpose.refiners import essential, fundamental
from fastpose.refiners import utils as refiner_utils


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


def make_points(rng, n):
    x1 = rng.uniform(-1.0, 1.0, size=(n, 2))
    x2 = rng.uniform(-1.0, 1.0, size=(n, 2))
    data = (np.ascontiguousarray(x1[:, 0]), np.ascontiguousarray(x1[:, 1]),
            np.ascontiguousarray(x2[:, 0]), np.ascontiguousarray(x2[:, 1]))
    return x1, x2, data


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
    x1, x2, data = make_points(rng, n)
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
    x1, x2, data = make_points(rng, n)
    state, f = make_state(case, rng)
    num_tangent = case['num_tangent']

    s = sampson_residuals(case['model_to_e'](f), x1, x2)
    max_error_sq = float(np.median(s ** 2))

    JtJ = np.empty((num_tangent, num_tangent))
    Jtr = np.empty(num_tangent)
    num_residuals = case['accumulate'](data, f, state, JtJ, Jtr, max_error_sq)
    assert num_residuals == int(np.count_nonzero(s ** 2 < max_error_sq))
