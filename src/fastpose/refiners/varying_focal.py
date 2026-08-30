"""Relative pose refiner with two unknown focal lengths.

The model is `[R | t | f1 | f2]`. LM optimizes a 7-dimensional tangent:
3 rotation parameters, 2 translation-direction parameters and 2 log-focal
parameters. The Sampson residual Jacobian is built from an analytic tangent
basis of the induced fundamental matrix.
"""

import math

import numpy as np
from numba import njit

from fastpose.refiners.lm import build_lm_refine
from fastpose.refiners.losses import TruncatedLoss
from fastpose.refiners.utils import (accumulate_sampson_normal_eqs, build_focal_lo_refine,
                            build_sampson_accumulate, essential_tangent_rows,
                            log_focal_tangent_rows, mat3_mul, rodrigues,
                            tangent_basis as _tangent_basis)
from fastpose.scorers.sampson import (build_varying_focal_pose_sampson_cost,
                             calibrate_epipolar, essential_from_pose,
                             varying_focal_pose_sampson_score)
from fastpose.solvers.varying_focal import MODEL_SIZE

STATE_SIZE = 14
NUM_TANGENT = 7
# poselib's `if (num_inl <= 7) return;` gate in
# VaryingFocalRelativePoseEstimator::refine_model
MIN_INLIERS = 7


@njit(cache=True)
def _init_state(model, state):
    norm_t = math.sqrt(model[9] * model[9] + model[10] * model[10]
                       + model[11] * model[11])
    if norm_t < 1e-12 or model[12] <= 0.0 or model[13] <= 0.0:
        return False
    inv = 1.0 / norm_t
    for j in range(9):
        state[j] = model[j]
    state[9] = model[9] * inv
    state[10] = model[10] * inv
    state[11] = model[11] * inv
    state[12] = math.log(model[12])
    state[13] = math.log(model[13])
    return True


@njit(cache=True)
def _state_to_model(state, model):
    for j in range(12):
        model[j] = state[j]
    model[12] = math.exp(state[12])
    model[13] = math.exp(state[13])


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
    state_new[13] = state[13] + delta[6]


def _make_accumulate(sampson_accumulate):
    @njit(cache=True)
    def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
        x1_x, x1_y, x2_x, x2_y, pp1x, pp1y, pp2x, pp2y = data
        f1 = f[12]
        f2 = f[13]
        if f1 <= 0.0 or f2 <= 0.0:
            return 0
        inv1 = 1.0 / f1
        inv2 = 1.0 / f2

        e = np.empty(9)
        essential_from_pose(f, e)
        base_f = np.empty(9)
        calibrate_epipolar(e, pp1x, pp1y, pp2x, pp2y, inv1, inv2, base_f)

        B = np.empty((NUM_TANGENT, 9))
        # pose rows: E -> F is linear, so dF/dtheta is the tangent direction
        # dE/dtheta pushed through the very same map
        dE = np.empty((5, 9))
        essential_tangent_rows(f, e, dE)
        for p in range(5):
            calibrate_epipolar(dE[p], pp1x, pp1y, pp2x, pp2y, inv1, inv2, B[p])

        # focal rows: one per focal, both read straight off F
        log_focal_tangent_rows(base_f, pp1x, pp1y, pp2x, pp2y, B[5], B[6])

        return sampson_accumulate(
            (x1_x, x1_y, x2_x, x2_y), base_f, B, JtJ, Jtr, max_error_sq)

    return _accumulate


_accumulate_truncated = _make_accumulate(accumulate_sampson_normal_eqs)
_accumulate = _accumulate_truncated  # back-compat alias for the default kernel

_refine_varying_focal_lm = build_lm_refine(
    _init_state, _state_to_model, _accumulate_truncated, _apply_step,
    varying_focal_pose_sampson_score, STATE_SIZE, NUM_TANGENT, MODEL_SIZE)

_final_kernels = {}
_lo_kernels = {}


def _get_final_refine(loss):
    # lazily compiles (and caches) the LM kernel for a non-default loss; see
    # essential.py's _get_final_refine for the general rationale
    key = type(loss)
    if key not in _final_kernels:
        accumulate_final = _make_accumulate(build_sampson_accumulate(loss))
        cost_final = build_varying_focal_pose_sampson_cost(loss)
        _final_kernels[key] = build_lm_refine(
            _init_state, _state_to_model, accumulate_final, _apply_step,
            cost_final, STATE_SIZE, NUM_TANGENT, MODEL_SIZE)
    return _final_kernels[key]


def _get_lo_refine(refine_fn, relaxed_scale):
    key = (refine_fn, relaxed_scale)
    if key not in _lo_kernels:
        _lo_kernels[key] = build_focal_lo_refine(refine_fn, MIN_INLIERS,
                                                 relaxed_scale)
    return _lo_kernels[key]


class LMVaryingFocalPoseRefiner():
    # `loss` selects the robust cost/weighting (TruncatedLoss by default,
    # matching every RANSAC-internal use; pass e.g. CauchyLoss() or
    # TruncatedCauchyLoss() for a final polish pass on an inlier-only
    # subset). `relaxed_inlier_scale` restricts the refit to the relaxed
    # inlier subset, as poselib's refine_model does - see
    # refiners/essential.py's LMEssentialRefiner.
    def __init__(self, num_iterations=15, loss=TruncatedLoss(),
                 relaxed_inlier_scale=None):
        self.num_iterations = num_iterations
        self.loss = loss
        self.relaxed_inlier_scale = relaxed_inlier_scale
        refine = (_refine_varying_focal_lm if isinstance(loss, TruncatedLoss)
                  else _get_final_refine(loss))
        if relaxed_inlier_scale is not None:
            refine = _get_lo_refine(refine, float(relaxed_inlier_scale))
        if refine is not _refine_varying_focal_lm:
            self.refine = refine

    refine = staticmethod(_refine_varying_focal_lm)
