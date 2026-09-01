"""Finite-difference check of the homography refiner's jacobian.

The other refiners all reduce a correspondence to one Sampson residual, which
is what `test_jacobians.py` is written around. The homography refiner has four
residuals per point - two forward transfer components and two backward ones -
and a state that lives on the unit sphere rather than on a rotation manifold,
so it gets its own harness here.

The residual jacobian is built by central finite differences through the
actual retraction (`_apply_step`) and model reassembly (`_state_to_model`)
kernels, and the accumulated normal equations are compared against
J_fd^T J_fd and J_fd^T r. The backward pair is what this really exercises:
its analytic form leans on d(H^-1) = -H^-1 dH H^-1, which nothing else in the
package uses.
"""

import numpy as np
import pytest

from fastpose.refiners import homography
from fastpose.scorers.transfer import DERIVED_SIZE, homography_derived

NUM_TANGENT = homography.NUM_TANGENT
NUM_RESIDUALS = homography.NUM_RESIDUALS


def transfer_residuals(model, x1, x2):
    # numpy reference of the four (untruncated) symmetric transfer residuals
    # per correspondence, as an (n, 4) array in the kernel's own order. The
    # 1/sqrt(2) is the kernel's too: the scorer averages the two transfer
    # terms, so the four squares must sum to that mean rather than to twice it
    d = np.empty(DERIVED_SIZE)
    assert homography_derived(np.ascontiguousarray(model), d)
    H = d[:9].reshape(3, 3)
    G = d[9:].reshape(3, 3)
    ones = np.ones(len(x1))
    p = np.column_stack([x1, ones]) @ H.T
    q = np.column_stack([x2, ones]) @ G.T
    return np.sqrt(0.5) * np.column_stack([p[:, :2] / p[:, 2:3] - x2,
                                           q[:, :2] / q[:, 2:3] - x1])


def make_points(rng, n):
    x1 = rng.uniform(-1.0, 1.0, size=(n, 2))
    x2 = rng.uniform(-1.0, 1.0, size=(n, 2))
    data = (np.ascontiguousarray(x1[:, 0]), np.ascontiguousarray(x1[:, 1]),
            np.ascontiguousarray(x2[:, 0]), np.ascontiguousarray(x2[:, 1]))
    return x1, x2, data


def make_state(rng):
    # A *plausible* homography rather than a uniformly random 3x3. A random
    # one sends some point of [-1, 1]^2 arbitrarily close to the line at
    # infinity of H or of H^-1, where the residuals reach 1e7 and the central
    # differences below stop resolving them - the kernel is fine there, the
    # reference is not. A perturbed identity keeps both third rows near 1.
    model = (np.eye(3) + 0.15 * rng.normal(size=(3, 3))).ravel()
    model /= np.linalg.norm(model)
    state = np.empty(homography.STATE_SIZE)
    assert homography._init_state(model, state)
    f = np.empty(homography.MODEL_SIZE)
    homography._state_to_model(state, f)
    return state, f


def fd_jacobian(state, x1, x2, h=1e-6):
    J = np.empty((len(x1) * NUM_RESIDUALS, NUM_TANGENT))
    delta = np.zeros(NUM_TANGENT)
    state_new = np.empty(homography.STATE_SIZE)
    model = np.empty(homography.MODEL_SIZE)

    def residuals_at():
        homography._apply_step(state, delta, state_new)
        homography._state_to_model(state_new, model)
        return transfer_residuals(model, x1, x2).ravel()

    for p in range(NUM_TANGENT):
        delta[p] = h
        r_plus = residuals_at()
        delta[p] = -h
        r_minus = residuals_at()
        delta[p] = 0.0
        J[:, p] = (r_plus - r_minus) / (2.0 * h)
    return J


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_normal_equations_match_finite_differences(seed):
    rng = np.random.default_rng(seed)
    n = 60
    x1, x2, data = make_points(rng, n)
    state, f = make_state(rng)

    JtJ = np.empty((NUM_TANGENT, NUM_TANGENT))
    Jtr = np.empty(NUM_TANGENT)
    # huge threshold: every point contributes, no truncation kinks
    num_residuals = homography._accumulate(data, f, state, JtJ, Jtr, 1e30)
    assert num_residuals == n * NUM_RESIDUALS

    J_fd = fd_jacobian(state, x1, x2)
    r = transfer_residuals(f, x1, x2).ravel()

    np.testing.assert_allclose(Jtr, J_fd.T @ r, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(JtJ, J_fd.T @ J_fd, rtol=1e-4, atol=1e-6)


def test_truncation_only_counts_inliers():
    # with a finite threshold the accumulated residual count must match four
    # times the number of points whose symmetric transfer error is below it
    rng = np.random.default_rng(7)
    n = 200
    x1, x2, data = make_points(rng, n)
    state, f = make_state(rng)

    e2 = np.sum(transfer_residuals(f, x1, x2) ** 2, axis=1)
    max_error_sq = float(np.median(e2))

    JtJ = np.empty((NUM_TANGENT, NUM_TANGENT))
    Jtr = np.empty(NUM_TANGENT)
    num_residuals = homography._accumulate(data, f, state, JtJ, Jtr,
                                           max_error_sq)
    assert num_residuals == NUM_RESIDUALS * int(
        np.count_nonzero(e2 < max_error_sq))


def test_retraction_stays_on_the_sphere_and_moves_along_the_basis():
    rng = np.random.default_rng(21)
    state, _ = make_state(rng)
    B = np.empty((NUM_TANGENT, 9))
    homography.sphere_tangent_basis(state, B)

    delta = 1e-3 * rng.normal(size=NUM_TANGENT)
    state_new = np.empty(homography.STATE_SIZE)
    homography._apply_step(state, delta, state_new)

    np.testing.assert_allclose(np.linalg.norm(state_new), 1.0, rtol=1e-12)
    # to first order the step is exactly B^T delta
    expected = state + delta @ B
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(state_new, expected, atol=1e-12)
