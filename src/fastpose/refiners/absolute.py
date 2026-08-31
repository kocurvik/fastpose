"""Absolute pose refiner: LM on the truncated reprojection error with the
pose optimized directly.

The state/model layout is the flat pose [R (row-major 3x3) | t (3)]. The 6
tangent parameters are a rotation update applied as R exp([w]_x) (3) and a
plain translation update (3). The reprojection jacobian is analytic:
r = pi(R X + t) - x with pi the pinhole projection, dZ/dw = -R [X]_x and
dZ/dt = I. Only the jacobian accumulation and the retraction are defined
here - the LM loop itself lives in refiners/lm.py.

`build_reprojection_primitives` is the shared-with-CUDA factory: the same
source is compiled with `njit` for the accumulate below and with
`cuda.jit(device=True)` for the block-reduction accumulate in
`cuda/problems/absolute.py`, so there is only one copy of the jacobian. Its
`focal` flag adds the log-focal column and the pixel scaling, which is what
`refiners/absolute_focal.py` builds it with.
"""

import numpy as np
from numba import float64, njit

from fastpose.jit_backend import cpu_jit
from fastpose.refiners.lm import build_lm_refine
from fastpose.refiners.losses import TruncatedLoss
from fastpose.refiners.utils import mat3_mul, rodrigues
from fastpose.scorers.reprojection import build_reprojection_cost, reprojection_score

STATE_SIZE = 12
MODEL_SIZE = 12
NUM_TANGENT = 6  # 3 rotation + 3 translation


def build_reprojection_primitives(jit, real=float64, focal=False):
    # `real` types the float literals, so the CUDA refiner can build this in
    # float32 without a bare `1.0` silently promoting the chain back to
    # float64; see build_sampson_point_kernels for the full rationale.
    # `focal` is a compile-time flag: it switches the residual to pixels
    # (r = f pi(Z) - x) and appends the dr/dlog(f) column, giving the 7
    # tangent parameters refiners/absolute_focal.py wants.
    @jit(fastmath=True, inline=True)
    def reprojection_point_jacobian(f, Xx, Xy, Xz, x, y, J0, J1):
        # residuals (rx, ry) of one 2D-3D correspondence and the two rows of
        # dr/dtheta, written into J0 and J1. Returns (rx, ry, ok); ok is
        # False for a point behind the camera, where J0/J1 are left untouched
        # and the caller drops the point.
        zx = f[0] * Xx + f[1] * Xy + f[2] * Xz + f[9]
        zy = f[3] * Xx + f[4] * Xy + f[5] * Xz + f[10]
        zz = f[6] * Xx + f[7] * Xy + f[8] * Xz + f[11]
        if zz <= real(0.0):
            return real(0.0), real(0.0), False
        inv = real(1.0) / zz
        px = zx * inv
        py = zy * inv
        if focal:
            fl = f[12]
            rx = fl * px - x
            ry = fl * py - y
            # dr/dZ carries the focal in the pixel case
            g = fl * inv
        else:
            rx = px - x
            ry = py - y
            g = inv

        # A = dr/dZ @ R with dr/dZ = g * [[1, 0, -px], [0, 1, -py]]
        a00 = g * (f[0] - px * f[6])
        a01 = g * (f[1] - px * f[7])
        a02 = g * (f[2] - px * f[8])
        a10 = g * (f[3] - py * f[6])
        a11 = g * (f[4] - py * f[7])
        a12 = g * (f[5] - py * f[8])

        # rotation columns dZ/dw_k = R (e_k x X); translation dZ/dt = I
        J0[0] = -a01 * Xz + a02 * Xy
        J0[1] = a00 * Xz - a02 * Xx
        J0[2] = -a00 * Xy + a01 * Xx
        J1[0] = -a11 * Xz + a12 * Xy
        J1[1] = a10 * Xz - a12 * Xx
        J1[2] = -a10 * Xy + a11 * Xx
        J0[3] = g
        J0[4] = real(0.0)
        J0[5] = -px * g
        J1[3] = real(0.0)
        J1[4] = g
        J1[5] = -py * g
        if focal:
            # focal column dr/dlog(fl) = fl * pi(Z)
            J0[6] = fl * px
            J1[6] = fl * py
        return rx, ry, True

    return {'reprojection_point_jacobian': reprojection_point_jacobian}


_CPU_PRIM = build_reprojection_primitives(cpu_jit)
reprojection_point_jacobian = _CPU_PRIM['reprojection_point_jacobian']


@njit(cache=True)
def _init_state(model, state):
    for j in range(12):
        state[j] = model[j]
    return True


@njit(cache=True)
def _state_to_model(state, f):
    for j in range(12):
        f[j] = state[j]


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


def build_accumulate(loss):
    # normal equations of the reprojection residuals, weighted by
    # loss.weight(r2, max_error_sq); points behind the camera or with zero
    # weight are dropped. Returns the number of contributing scalar
    # residuals (2 per point).
    weight_fn = loss.weight

    @njit(cache=True, fastmath=True)
    def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
        x_x, x_y, X_x, X_y, X_z = data
        n = x_x.shape[0]

        for p in range(NUM_TANGENT):
            Jtr[p] = 0.0
            for q in range(NUM_TANGENT):
                JtJ[p, q] = 0.0

        J0 = np.empty(NUM_TANGENT)
        J1 = np.empty(NUM_TANGENT)

        num_residuals = 0
        for i in range(n):
            rx, ry, ok = reprojection_point_jacobian(
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

_refine_absolute_lm = build_lm_refine(_init_state, _state_to_model,
                                      _accumulate_truncated, _apply_step,
                                      reprojection_score, STATE_SIZE,
                                      NUM_TANGENT, MODEL_SIZE)

_final_kernels = {}


def _get_final_refine(loss):
    # lazily compiles (and caches) the LM kernel for a non-default loss; see
    # essential.py's _get_final_refine for the general rationale
    key = type(loss)
    if key not in _final_kernels:
        accumulate_final = build_accumulate(loss)
        cost_final = build_reprojection_cost(loss)
        _final_kernels[key] = build_lm_refine(
            _init_state, _state_to_model, accumulate_final, _apply_step,
            cost_final, STATE_SIZE, NUM_TANGENT, MODEL_SIZE)
    return _final_kernels[key]


class LMAbsolutePoseRefiner():
    # local optimization: LM on the reprojection error with the pose (R, t)
    # optimized directly (6 tangent parameters). `loss` selects the robust
    # cost/weighting (TruncatedLoss by default, matching every RANSAC-
    # internal use; pass e.g. CauchyLoss() or TruncatedCauchyLoss() for a
    # final polish pass on an inlier-only subset).
    def __init__(self, num_iterations=25, loss=TruncatedLoss()):
        self.num_iterations = num_iterations
        self.loss = loss
        if not isinstance(loss, TruncatedLoss):
            self.refine = _get_final_refine(loss)

    refine = staticmethod(_refine_absolute_lm)
