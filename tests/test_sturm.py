"""The Sturm root finder must never write past the caller's `roots` buffer.

Every solver sizes `roots` at the polynomial degree and then turns each
returned root into one `models` row of a `max_models`-tall array, so a single
extra root is an out-of-bounds write into the heap - which surfaces much
later as glibc "free(): invalid size" / "corrupted double-linked list".

The chain built by `_real_roots_sturm` is truncated on a nontrivial gcd
(multiple roots) and its sign counts come from floating-point evaluations, so
a cluster of near-multiple roots can be emitted more than once; the polynomial
below is a degree-10 case that returned 11 roots before the capacity check.
"""

import numpy as np
import pytest

from fastpose.solvers.essential import _real_roots_sturm

# (x - r_i)(x - r_i - 1e-12) ... : five near-double roots, ascending coeffs
NEAR_DOUBLE_ROOTS_DEG10 = np.array([
    -6.33088172501595491948e-02,
    7.80965717968628680978e-01,
    -4.02591680899388748571e+00,
    1.09911006390529664145e+01,
    -1.59125294647558099825e+01,
    7.62614329075548091907e+00,
    1.23958971264642485721e+01,
    -2.47474790771409374202e+01,
    1.89235888290003941847e+01,
    -6.96849969642849931262e+00,
    1.00000000000000000000e+00,
])


def _roots(coef, degree, capacity):
    chain = np.zeros((degree + 1, degree + 1))
    degs = np.zeros(degree + 1, dtype=np.int64)
    roots = np.zeros(capacity)
    lo_stack = np.zeros(64)
    hi_stack = np.zeros(64)
    slo_stack = np.zeros(64, dtype=np.int64)
    shi_stack = np.zeros(64, dtype=np.int64)
    num = _real_roots_sturm(coef, degree, chain, degs, roots, lo_stack,
                            hi_stack, slo_stack, shi_stack)
    return num, roots


def test_near_multiple_roots_stay_within_the_buffer():
    # the exact-size buffer every solver passes
    num, _ = _roots(NEAR_DOUBLE_ROOTS_DEG10, 10, 10)
    assert num <= 10


def test_capacity_is_taken_from_the_buffer_not_the_degree():
    # a caller with a shorter buffer must also be respected
    num, roots = _roots(NEAR_DOUBLE_ROOTS_DEG10, 10, 3)
    assert num <= 3
    assert np.all(np.isfinite(roots))


def test_simple_roots_are_still_all_found():
    # the capacity check must not cost roots on well-separated polynomials
    gt = np.array([-3.0, -1.5, -0.5, 0.25, 1.0, 2.0, 3.5, 5.0, 7.0, 11.0])
    coef = np.ascontiguousarray(np.polynomial.polynomial.polyfromroots(gt))
    num, roots = _roots(coef, 10, 10)
    assert num == len(gt)
    assert np.sort(roots[:num]) == pytest.approx(gt, abs=1e-6)


def test_random_polynomials_never_exceed_capacity():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        k = rng.integers(1, 6)
        r = rng.normal(size=int(k))
        eps = 10.0 ** rng.integers(-15, -6)
        gt = np.concatenate([r, r + eps])[:10]
        if len(gt) < 10:
            gt = np.concatenate([gt, rng.normal(size=10 - len(gt))])
        coef = np.ascontiguousarray(np.polynomial.polynomial.polyfromroots(gt))
        num, _ = _roots(coef, 10, 10)
        assert num <= 10
