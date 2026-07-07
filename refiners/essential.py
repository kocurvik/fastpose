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

from refiners.lm import build_lm_refine
from refiners.utils import accumulate_sampson_normal_eqs, mat3_mul, rodrigues
from scorers.sampson import essential_from_pose, pose_sampson_score

STATE_SIZE = 12
MODEL_SIZE = 12
NUM_TANGENT = 5  # 3 rotation + 2 translation direction


@njit(cache=True)
def _tangent_basis(t, b1, b2):
    # orthonormal basis (b1, b2) of the plane orthogonal to the unit vector
    # t; deterministic in t so the jacobian (in _accumulate) and the
    # retraction (in _apply_step) always agree
    a0 = abs(t[0])
    a1 = abs(t[1])
    a2 = abs(t[2])
    # b1 = t x e_k for the axis e_k of the smallest |component|
    if a0 <= a1 and a0 <= a2:
        b1[0] = 0.0
        b1[1] = t[2]
        b1[2] = -t[1]
    elif a1 <= a2:
        b1[0] = -t[2]
        b1[1] = 0.0
        b1[2] = t[0]
    else:
        b1[0] = t[1]
        b1[1] = -t[0]
        b1[2] = 0.0
    inv = 1.0 / math.sqrt(b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2])
    b1[0] *= inv
    b1[1] *= inv
    b1[2] *= inv
    b2[0] = t[1] * b1[2] - t[2] * b1[1]
    b2[1] = t[2] * b1[0] - t[0] * b1[2]
    b2[2] = t[0] * b1[1] - t[1] * b1[0]


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


@njit(cache=True)
def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
    e = np.empty(9)
    essential_from_pose(state, e)

    B = np.empty((NUM_TANGENT, 9))
    # rotation rows: dE/dw_k = E skew(e_k) for the retraction R exp([w]_x) —
    # plain column shuffles of E
    for i in range(3):
        e0 = e[3 * i]
        e1 = e[3 * i + 1]
        e2 = e[3 * i + 2]
        B[0, 3 * i] = 0.0
        B[0, 3 * i + 1] = e2
        B[0, 3 * i + 2] = -e1
        B[1, 3 * i] = -e2
        B[1, 3 * i + 1] = 0.0
        B[1, 3 * i + 2] = e0
        B[2, 3 * i] = e1
        B[2, 3 * i + 1] = -e0
        B[2, 3 * i + 2] = 0.0

    # translation rows: dE/dalpha_i = [b_i]_x R (dt/dalpha_i = b_i on the
    # unit sphere since b_i is orthogonal to t)
    b1 = np.empty(3)
    b2 = np.empty(3)
    _tangent_basis(state[9:12], b1, b2)
    for j in range(3):
        r0 = state[j]
        r1 = state[3 + j]
        r2 = state[6 + j]
        B[3, j] = -b1[2] * r1 + b1[1] * r2
        B[3, 3 + j] = b1[2] * r0 - b1[0] * r2
        B[3, 6 + j] = -b1[1] * r0 + b1[0] * r1
        B[4, j] = -b2[2] * r1 + b2[1] * r2
        B[4, 3 + j] = b2[2] * r0 - b2[0] * r2
        B[4, 6 + j] = -b2[1] * r0 + b2[0] * r1

    return accumulate_sampson_normal_eqs(data, e, B, JtJ, Jtr, max_error_sq)


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
                                       _accumulate, _apply_step,
                                       pose_sampson_score, STATE_SIZE,
                                       NUM_TANGENT, MODEL_SIZE)


class LMEssentialRefiner():
    # local optimization: LM on the truncated Sampson error with the pose
    # (R, t) optimized directly (5 tangent parameters, t on the unit sphere)
    def __init__(self, num_iterations=25):
        self.num_iterations = num_iterations

    refine = staticmethod(_refine_essential_lm)
