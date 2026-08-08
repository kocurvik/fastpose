"""Fundamental matrix refiner: LM on the truncated Sampson error with F
parametrized by its SVD factorization F = U diag(1, sigma, 0) Vt (like
poselib's FactorizedFundamentalMatrix). The 7 tangent parameters are two
rotation updates and the singular value ratio sigma; rank 2 holds by
construction. Only the jacobian accumulation and the retraction step are
defined here — the LM loop itself lives in refiners/lm.py.
"""

import numpy as np
from numba import njit

from fastpose.refiners.lm import build_lm_refine
from fastpose.refiners.losses import TruncatedLoss
from fastpose.refiners.utils import (STATE_SIZE, accumulate_sampson_normal_eqs,
                            apply_rotation_step, build_sampson_accumulate,
                            epipolar_jacobian_basis, state_to_model, svd_init_state)
from fastpose.scorers.sampson import build_sampson_cost, sampson_score

NUM_TANGENT = 7  # 3 left rotation + 3 right rotation + sigma

_init_state = svd_init_state


def _make_accumulate(sampson_accumulate):
    @njit(cache=True)
    def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
        U = state[0:9].reshape(3, 3)
        Vt = state[9:18].reshape(3, 3)
        B = np.empty((NUM_TANGENT, 9))
        epipolar_jacobian_basis(U, Vt, state[18], B)
        return sampson_accumulate(data, f, B, JtJ, Jtr, max_error_sq)

    return _accumulate


_accumulate_truncated = _make_accumulate(accumulate_sampson_normal_eqs)
_accumulate = _accumulate_truncated  # back-compat alias for the default kernel


@njit(cache=True)
def _apply_step(state, delta, state_new):
    apply_rotation_step(state, delta, state_new)
    state_new[18] = state[18] + delta[6]


_refine_fundamental_lm = build_lm_refine(_init_state, state_to_model,
                                         _accumulate_truncated, _apply_step,
                                         sampson_score, STATE_SIZE,
                                         NUM_TANGENT, 9)

_final_kernels = {}


def _get_final_refine(loss):
    # lazily compiles (and caches) the LM kernel for a non-default loss; see
    # essential.py's _get_final_refine for the general rationale
    key = type(loss)
    if key not in _final_kernels:
        accumulate_final = _make_accumulate(build_sampson_accumulate(loss))
        cost_final = build_sampson_cost(loss)
        _final_kernels[key] = build_lm_refine(
            _init_state, state_to_model, accumulate_final, _apply_step,
            cost_final, STATE_SIZE, NUM_TANGENT, 9)
    return _final_kernels[key]


class LMFundamentalRefiner():
    # local optimization: Levenberg-Marquardt on the Sampson error with the
    # SVD-factorized parametrization F = U diag(1, sigma, 0) Vt (7 tangent
    # parameters), analogous to poselib. num_iterations is the LM step
    # budget; converged runs stop earlier. `loss` selects the robust cost/
    # weighting (TruncatedLoss by default, matching every RANSAC-internal
    # use; pass e.g. CauchyLoss() or TruncatedCauchyLoss() for a final
    # polish pass on an inlier-only subset).
    def __init__(self, num_iterations=25, loss=TruncatedLoss()):
        self.num_iterations = num_iterations
        self.loss = loss
        if not isinstance(loss, TruncatedLoss):
            self.refine = _get_final_refine(loss)

    refine = staticmethod(_refine_fundamental_lm)
