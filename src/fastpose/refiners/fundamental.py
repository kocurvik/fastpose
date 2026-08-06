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
from fastpose.refiners.utils import (STATE_SIZE, accumulate_sampson_normal_eqs,
                            apply_rotation_step, epipolar_jacobian_basis,
                            state_to_model, svd_init_state)
from fastpose.scorers.sampson import sampson_score

NUM_TANGENT = 7  # 3 left rotation + 3 right rotation + sigma

_init_state = svd_init_state


@njit(cache=True)
def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
    U = state[0:9].reshape(3, 3)
    Vt = state[9:18].reshape(3, 3)
    B = np.empty((NUM_TANGENT, 9))
    epipolar_jacobian_basis(U, Vt, state[18], B)
    return accumulate_sampson_normal_eqs(data, f, B, JtJ, Jtr, max_error_sq)


@njit(cache=True)
def _apply_step(state, delta, state_new):
    apply_rotation_step(state, delta, state_new)
    state_new[18] = state[18] + delta[6]


_refine_fundamental_lm = build_lm_refine(_init_state, state_to_model,
                                         _accumulate, _apply_step,
                                         sampson_score, STATE_SIZE,
                                         NUM_TANGENT, 9)


class LMFundamentalRefiner():
    # local optimization: Levenberg-Marquardt on the truncated Sampson error
    # with the SVD-factorized parametrization F = U diag(1, sigma, 0) Vt
    # (7 tangent parameters), analogous to poselib. num_iterations is the LM
    # step budget; converged runs stop earlier.
    def __init__(self, num_iterations=25):
        self.num_iterations = num_iterations

    refine = staticmethod(_refine_fundamental_lm)
