"""Absolute pose refiner with unknown focal length.

The model is `[R | t | f]`. LM optimizes a 7-dimensional tangent: 3 rotation
parameters (retraction R exp([w]_x)), 3 translation parameters and one
log-focal parameter. The reprojection jacobian is analytic like in
refiners/absolute.py, with the extra column dr/dlog(f) = f * pi(R X + t).
"""

import math

import numpy as np
from numba import njit

from fastpose.jit_backend import cpu_jit
from fastpose.refiners.absolute import build_reprojection_primitives
from fastpose.refiners.lm import build_lm_refine
from fastpose.refiners.losses import TruncatedLoss
from fastpose.refiners.utils import mat3_mul, rodrigues
from fastpose.scorers.reprojection import build_focal_reprojection_cost, focal_reprojection_score

STATE_SIZE = 13
MODEL_SIZE = 13
NUM_TANGENT = 7  # 3 rotation + 3 translation + log-focal

# same jacobian source as the calibrated refiner, built with the focal column
_CPU_PRIM = build_reprojection_primitives(cpu_jit, focal=True)
focal_reprojection_point_jacobian = _CPU_PRIM['reprojection_point_jacobian']


@njit(cache=True)
def _init_state(model, state):
    if model[12] <= 0.0:
        return False
    for j in range(12):
        state[j] = model[j]
    state[12] = math.log(model[12])
    return True


@njit(cache=True)
def _state_to_model(state, f):
    for j in range(12):
        f[j] = state[j]
    f[12] = math.exp(state[12])


@njit(cache=True)
def _apply_step(state, delta, state_new):
    R = state[0:9].reshape(3, 3)
    R_new = state_new[0:9].reshape(3, 3)
    Rod = np.empty((3, 3))
    rodrigues(delta[0], delta[1], delta[2], Rod)
    mat3_mul(R, Rod, R_new)
    state_new[9] = state[9] + delta[3]
    state_new[10] = state[10] + delta[4]
    state_new[11] = state[11] + delta[5]
    state_new[12] = state[12] + delta[6]


def build_accumulate(loss):
    # normal equations of the pixel reprojection residuals
    # r = fl * pi(R X + t) - x, weighted by loss.weight(r2, max_error_sq).
    # Returns the number of contributing scalar residuals (2 per point).
    weight_fn = loss.weight

    @njit(cache=True, fastmath=True)
    def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
        x_x, x_y, X_x, X_y, X_z = data
        n = x_x.shape[0]
        if f[12] <= 0.0:
            return 0

        for p in range(NUM_TANGENT):
            Jtr[p] = 0.0
            for q in range(NUM_TANGENT):
                JtJ[p, q] = 0.0

        J0 = np.empty(NUM_TANGENT)
        J1 = np.empty(NUM_TANGENT)

        num_residuals = 0
        for i in range(n):
            rx, ry, ok = focal_reprojection_point_jacobian(
                f, X_x[i], X_y[i], X_z[i], x_x[i], x_y[i], J0, J1)
            if not ok:
                continue
            w = weight_fn(rx * rx + ry * ry, max_error_sq)
            if w <= 0.0:
                continue
            num_residuals += 2

            for p in range(NUM_TANGENT):
                Jtr[p] += w * (J0[p] * rx + J1[p] * ry)
                for q in range(p, NUM_TANGENT):
                    JtJ[p, q] += w * (J0[p] * J0[q] + J1[p] * J1[q])

        for p in range(NUM_TANGENT):
            for q in range(p):
                JtJ[p, q] = JtJ[q, p]
        return num_residuals

    return _accumulate


_accumulate_truncated = build_accumulate(TruncatedLoss())
_accumulate = _accumulate_truncated  # back-compat alias for the default kernel

_refine_absolute_focal_lm = build_lm_refine(_init_state, _state_to_model,
                                            _accumulate_truncated, _apply_step,
                                            focal_reprojection_score,
                                            STATE_SIZE, NUM_TANGENT,
                                            MODEL_SIZE)

_final_kernels = {}


def _get_final_refine(loss):
    # lazily compiles (and caches) the LM kernel for a non-default loss; see
    # essential.py's _get_final_refine for the general rationale
    key = type(loss)
    if key not in _final_kernels:
        accumulate_final = build_accumulate(loss)
        cost_final = build_focal_reprojection_cost(loss)
        _final_kernels[key] = build_lm_refine(
            _init_state, _state_to_model, accumulate_final, _apply_step,
            cost_final, STATE_SIZE, NUM_TANGENT, MODEL_SIZE)
    return _final_kernels[key]


class LMAbsolutePoseFocalRefiner():
    # local optimization: LM on the pixel reprojection error with the pose
    # and log-focal optimized directly (7 tangent parameters). `loss`
    # selects the robust cost/weighting (TruncatedLoss by default, matching
    # every RANSAC-internal use; pass e.g. CauchyLoss() or
    # TruncatedCauchyLoss() for a final polish pass on an inlier-only
    # subset).
    def __init__(self, num_iterations=25, loss=TruncatedLoss()):
        self.num_iterations = num_iterations
        self.loss = loss
        if not isinstance(loss, TruncatedLoss):
            self.refine = _get_final_refine(loss)

    refine = staticmethod(_refine_absolute_focal_lm)
