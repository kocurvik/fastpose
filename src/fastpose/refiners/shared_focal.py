"""Relative pose refiner with one focal length shared by both cameras."""

import math

import numpy as np
from numba import njit

from fastpose.refiners.essential import _tangent_basis
from fastpose.refiners.lm import build_lm_refine
from fastpose.refiners.utils import accumulate_sampson_normal_eqs, mat3_mul, rodrigues
from fastpose.scorers.sampson import model_to_fundamental, shared_focal_pose_sampson_score
from fastpose.solvers.shared_focal import MODEL_SIZE

STATE_SIZE = 13
NUM_TANGENT = 6  # 3 rotation + 2 translation-direction + 1 log-focal


@njit(cache=True)
def _init_state(model, state):
    norm_t = math.sqrt(model[9] * model[9] + model[10] * model[10]
                       + model[11] * model[11])
    if norm_t < 1e-12 or model[12] <= 0.0:
        return False
    inv = 1.0 / norm_t
    for j in range(9):
        state[j] = model[j]
    state[9] = model[9] * inv
    state[10] = model[10] * inv
    state[11] = model[11] * inv
    state[12] = math.log(model[12])
    return True


@njit(cache=True)
def _state_to_model(state, model):
    for j in range(12):
        model[j] = state[j]
    f = math.exp(state[12])
    model[12] = f
    model[13] = f


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
    state_new[12] = state[12] + delta[5]


@njit(cache=True)
def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
    x1_x, x1_y, x2_x, x2_y, pp1x, pp1y, pp2x, pp2y = data
    base_f = np.empty(9)
    model = np.empty(MODEL_SIZE)
    _state_to_model(state, model)
    if not model_to_fundamental(model, pp1x, pp1y, pp2x, pp2y, base_f):
        return 0

    B = np.empty((NUM_TANGENT, 9))
    state_plus = np.empty(STATE_SIZE)
    state_minus = np.empty(STATE_SIZE)
    model_plus = np.empty(MODEL_SIZE)
    model_minus = np.empty(MODEL_SIZE)
    f_plus = np.empty(9)
    f_minus = np.empty(9)
    delta = np.zeros(NUM_TANGENT)
    h = 1e-6
    inv_2h = 0.5 / h
    for p in range(NUM_TANGENT):
        delta[p] = h
        _apply_step(state, delta, state_plus)
        delta[p] = -h
        _apply_step(state, delta, state_minus)
        delta[p] = 0.0
        _state_to_model(state_plus, model_plus)
        _state_to_model(state_minus, model_minus)
        if not model_to_fundamental(model_plus, pp1x, pp1y, pp2x, pp2y, f_plus):
            return 0
        if not model_to_fundamental(model_minus, pp1x, pp1y, pp2x, pp2y, f_minus):
            return 0
        for j in range(9):
            B[p, j] = (f_plus[j] - f_minus[j]) * inv_2h

    return accumulate_sampson_normal_eqs(
        (x1_x, x1_y, x2_x, x2_y), base_f, B, JtJ, Jtr, max_error_sq)


_refine_shared_focal_lm = build_lm_refine(
    _init_state, _state_to_model, _accumulate, _apply_step,
    shared_focal_pose_sampson_score, STATE_SIZE, NUM_TANGENT, MODEL_SIZE)


class LMSharedFocalPoseRefiner():
    def __init__(self, num_iterations=15):
        self.num_iterations = num_iterations

    refine = staticmethod(_refine_shared_focal_lm)
