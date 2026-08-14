"""Robust loss functions selectable per refinement pass.

A Loss exposes two njit kernels of a squared residual `r2` and the squared
threshold `max_error_sq`:

    weight(r2, max_error_sq) -> IRLS weight in [0, 1]
        multiplies a residual's contribution to the normal equations
        (JtJ, Jtr) built by the refiners/ accumulate kernels; 0 drops the
        point entirely (mirrors classical iteratively-reweighted least
        squares for an M-estimator with cost `cost`)
    cost(r2, max_error_sq) -> robustified per-residual cost
        summed by the LM loop's accept/reject test (refiners/lm.py) and by
        the cost kernels built in scorers/; monotonically increasing in r2
        so the scorers' per-chunk early bail-out stays valid

Both are plain njit functions wrapped in `staticmethod` so refiner/scorer
factories can close over them as compile-time constants, the same pattern
already used for the kernels handed to `build_lm_refine`.

`TruncatedLoss` reproduces the hard MSAC cutoff every refiner used before
this module existed, and stays the default for every RANSAC-internal use
(scoring and local optimization). `CauchyLoss` and `TruncatedCauchyLoss`
down-weight residuals continuously instead of hard-rejecting them at the
threshold; either is a reasonable choice to pass to a refiner for a final
polish pass on RANSAC inliers only, where letting remaining noisy-but-
plausible points keep pulling the fit avoids a handful of threshold-
adjacent points flipping in and out of the (sub)gradient. Any object
exposing the same `weight`/`cost` njit kernels works as a Loss - refiners
compile and cache a kernel per loss type on first use (see
`_get_final_refine` in each refiners/*.py module), so adding a new loss
here is enough to make it selectable everywhere.

`LOSSES` maps the string names the estimator entry points accept
('truncated', 'cauchy', 'truncated_cauchy') to these classes; `get_loss`
resolves a name to an instance and passes non-string Loss objects through
unchanged, so both spellings work wherever a loss is selected.
"""

import math

from numba import njit

_LOG2 = math.log(2.0)


@njit(cache=True, inline='always')
def _truncated_weight(r2, max_error_sq):
    return 1.0 if r2 < max_error_sq else 0.0


@njit(cache=True, inline='always')
def _truncated_cost(r2, max_error_sq):
    return r2 if r2 < max_error_sq else max_error_sq


@njit(cache=True, inline='always')
def _cauchy_weight(r2, max_error_sq):
    return 1.0 / (1.0 + r2 / max_error_sq)


@njit(cache=True, inline='always')
def _cauchy_cost(r2, max_error_sq):
    return max_error_sq * math.log1p(r2 / max_error_sq)


@njit(cache=True, inline='always')
def _truncated_cauchy_weight(r2, max_error_sq):
    return 1.0 / (1.0 + r2 / max_error_sq) if r2 < max_error_sq else 0.0


@njit(cache=True, inline='always')
def _truncated_cauchy_cost(r2, max_error_sq):
    if r2 < max_error_sq:
        return max_error_sq * math.log1p(r2 / max_error_sq)
    return max_error_sq * _LOG2


class TruncatedLoss():
    # hard threshold (MSAC): unit weight inside max_error_sq, zero outside
    weight = staticmethod(_truncated_weight)
    cost = staticmethod(_truncated_cost)


class CauchyLoss():
    # smooth robust loss of scale max_error_sq: down-weights residuals
    # continuously instead of hard-rejecting them at the threshold
    weight = staticmethod(_cauchy_weight)
    cost = staticmethod(_cauchy_cost)


class TruncatedCauchyLoss():
    # Cauchy-shaped down-weighting below max_error_sq, hard-rejected (zero
    # weight, cost capped at its value at r2 = max_error_sq) above it: a
    # smooth in-threshold taper combined with TruncatedLoss's hard cutoff
    weight = staticmethod(_truncated_cauchy_weight)
    cost = staticmethod(_truncated_cauchy_cost)


LOSSES = {
    'truncated': TruncatedLoss,
    'cauchy': CauchyLoss,
    'truncated_cauchy': TruncatedCauchyLoss,
}


def get_loss(loss):
    # resolve a loss name from LOSSES to a fresh instance; anything that is
    # not a string (a Loss object, or any duck-typed weight/cost pair) is
    # returned unchanged. Cheap - instantiating a Loss compiles nothing, the
    # kernels are only built when a refiner closes over it
    if not isinstance(loss, str):
        return loss
    try:
        cls = LOSSES[loss]
    except KeyError:
        raise ValueError(
            "unknown loss %r; expected one of %s"
            % (loss, ", ".join(LOSSES))) from None
    return cls()
