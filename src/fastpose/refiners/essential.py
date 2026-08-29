"""Relative pose refiner: LM on the truncated Sampson error of
E = [t]_x R with the pose optimized directly.

The state/model layout is the flat pose [R (row-major 3x3) | t (3)] with t
kept at unit norm. The 5 tangent parameters are a minimal parametrization of
the essential manifold (no gauge freedom): a rotation update applied as
R exp([w]_x) (3) and a translation-direction update in an orthonormal basis
of the plane orthogonal to t, retracted back to the unit sphere (2). Only
the jacobian accumulation and the retraction step are defined here — the LM
loop itself lives in refiners/lm.py.
"""

import math

import numpy as np
from numba import njit

from fastpose.refiners.lm import build_lm_refine
from fastpose.refiners.losses import TruncatedLoss
from fastpose.refiners.utils import (accumulate_sampson_normal_eqs, build_sampson_accumulate,
                            essential_tangent_rows, mat3_mul, rodrigues,
                            tangent_basis as _tangent_basis)
from fastpose.scorers.sampson import build_pose_sampson_cost, essential_from_pose, pose_sampson_score

STATE_SIZE = 12
MODEL_SIZE = 12
NUM_TANGENT = 5  # 3 rotation + 2 translation direction


@njit(cache=True)
def _init_state(model, state):
    # the model already is the pose; normalize t so the translation
    # retraction stays on the unit sphere
    norm_t = math.sqrt(model[9] * model[9] + model[10] * model[10]
                       + model[11] * model[11])
    if norm_t < 1e-12:
        return False
    inv = 1.0 / norm_t
    for j in range(9):
        state[j] = model[j]
    for j in range(9, 12):
        state[j] = model[j] * inv
    return True


@njit(cache=True)
def _state_to_model(state, f):
    for j in range(12):
        f[j] = state[j]


def _make_accumulate(sampson_accumulate):
    @njit(cache=True)
    def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
        e = np.empty(9)
        essential_from_pose(state, e)

        B = np.empty((NUM_TANGENT, 9))
        essential_tangent_rows(state, e, B)

        return sampson_accumulate(data, e, B, JtJ, Jtr, max_error_sq)

    return _accumulate


_accumulate_truncated = _make_accumulate(accumulate_sampson_normal_eqs)
_accumulate = _accumulate_truncated  # back-compat alias for the default kernel


@njit(cache=True)
def _apply_step(state, delta, state_new):
    R = state[0:9].reshape(3, 3)
    R_new = state_new[0:9].reshape(3, 3)
    Rod = np.empty((3, 3))
    rodrigues(delta[0], delta[1], delta[2], Rod)
    mat3_mul(R, Rod, R_new)

    t = state[9:12]
    b1 = np.empty(3)
    b2 = np.empty(3)
    _tangent_basis(t, b1, b2)
    t0 = t[0] + delta[3] * b1[0] + delta[4] * b2[0]
    t1 = t[1] + delta[3] * b1[1] + delta[4] * b2[1]
    t2 = t[2] + delta[3] * b1[2] + delta[4] * b2[2]
    inv = 1.0 / math.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
    state_new[9] = t0 * inv
    state_new[10] = t1 * inv
    state_new[11] = t2 * inv


_refine_essential_lm = build_lm_refine(_init_state, _state_to_model,
                                       _accumulate_truncated, _apply_step,
                                       pose_sampson_score, STATE_SIZE,
                                       NUM_TANGENT, MODEL_SIZE)

_final_kernels = {}


def _get_final_refine(loss):
    # lazily compiles (and caches) the LM kernel for a non-default loss,
    # e.g. CauchyLoss() or TruncatedCauchyLoss(), so any Loss object with
    # njit weight/cost kernels (see refiners/losses.py) can be selected
    # without adding a named kernel for it here
    key = type(loss)
    if key not in _final_kernels:
        accumulate_final = _make_accumulate(build_sampson_accumulate(loss))
        cost_final = build_pose_sampson_cost(loss)
        _final_kernels[key] = build_lm_refine(
            _init_state, _state_to_model, accumulate_final, _apply_step,
            cost_final, STATE_SIZE, NUM_TANGENT, MODEL_SIZE)
    return _final_kernels[key]


class LMEssentialRefiner():
    # local optimization: LM on the Sampson error with the pose (R, t)
    # optimized directly (5 tangent parameters, t on the unit sphere).
    # `loss` selects the robust cost/weighting (TruncatedLoss by default,
    # matching every RANSAC-internal use; pass e.g. CauchyLoss() or
    # TruncatedCauchyLoss() for a final polish pass on an inlier-only
    # subset).
    def __init__(self, num_iterations=25, loss=TruncatedLoss()):
        self.num_iterations = num_iterations
        self.loss = loss
        if not isinstance(loss, TruncatedLoss):
            self.refine = _get_final_refine(loss)

    refine = staticmethod(_refine_essential_lm)
