"""The numba cache keys of the closure-specialized kernels must be both
reproducible (same specialization -> same key, so a warmed cache actually
hits in a later process) and injective (different specializations -> different
keys, so a hit never returns another specialization's code).

Numba derives the key from the pickled closure cells, so these tests check the
uuids that fastpose.kernel_cache pins onto the cells; see kernel_cache.py for
why the default per-process uuids made every key miss.
"""

import hashlib

import numpy as np
import pytest
from numba.core.serialize import dumps

from fastpose.estimators.ransac import RansacEstimator
from fastpose.kernel_cache import _UUID_ATTR, stabilize
from fastpose.refiners.essential import LMEssentialRefiner
from fastpose.refiners.fundamental import LMFundamentalRefiner
from fastpose.refiners.losses import CauchyLoss, TruncatedCauchyLoss, TruncatedLoss
from fastpose.scorers.sampson import (PoseSampsonScorer, SampsonScorer,
                                      build_sampson_cost)
from fastpose.solvers.essential import FivePointSolver
from fastpose.solvers.fundamental import SevenPointSolver


def _uuid(kernel):
    return getattr(kernel, _UUID_ATTR, None)


def _cvar_hash(kernel):
    # the part of numba's cache key that distinguishes two closures compiled
    # from the same source (Cache._index_key in numba/core/caching.py)
    py_func = kernel.py_func
    cvars = tuple(c.cell_contents for c in py_func.__closure__)
    return hashlib.sha256(dumps(cvars)).hexdigest()


def test_same_specialization_gets_the_same_uuid():
    # two independently built kernels with identical contents are the same
    # code and may (must, for the cache to hit at all) share a key
    a = build_sampson_cost(TruncatedLoss())
    b = build_sampson_cost(TruncatedLoss())
    stabilize(a, b)
    assert _uuid(a) is not None
    assert _uuid(a) == _uuid(b)


def test_different_losses_get_different_uuids():
    kernels = [build_sampson_cost(loss) for loss in
               (TruncatedLoss(), CauchyLoss(), TruncatedCauchyLoss())]
    stabilize(*kernels)
    uuids = [_uuid(k) for k in kernels]
    assert all(u is not None for u in uuids)
    assert len(set(uuids)) == len(uuids)


def test_uuid_is_derived_from_identity_not_construction_order():
    # rebuilding in the opposite order must not shuffle the uuids, otherwise a
    # cache written by one process is useless to the next
    first = [build_sampson_cost(loss) for loss in (TruncatedLoss(), CauchyLoss())]
    stabilize(*first)
    second = [build_sampson_cost(loss) for loss in (CauchyLoss(), TruncatedLoss())]
    stabilize(*second)
    assert _uuid(first[0]) == _uuid(second[1])
    assert _uuid(first[1]) == _uuid(second[0])


def test_drivers_of_different_problems_do_not_share_a_cache_key():
    # every build_ransac driver is compiled from the same source and so shares
    # one cache index file; only the closure cells keep them apart
    essential = RansacEstimator(FivePointSolver(), PoseSampsonScorer(),
                                LMEssentialRefiner())
    fundamental = RansacEstimator(SevenPointSolver(), SampsonScorer(),
                                  LMFundamentalRefiner())
    assert _cvar_hash(essential._driver) != _cvar_hash(fundamental._driver)


def test_refiner_losses_do_not_share_a_cache_key():
    truncated = LMEssentialRefiner()
    cauchy = LMEssentialRefiner(loss=CauchyLoss())
    assert _cvar_hash(truncated.refine) != _cvar_hash(cauchy.refine)


def test_stabilize_ignores_non_kernels_without_raising():
    stabilize(None, 3, np.zeros(3), object())


def test_stabilized_estimator_still_produces_the_same_result():
    # a guard that pinning uuids is inert with respect to the actual numerics
    from fastpose.estimators.essential import estimate_relative_pose

    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(-0.6, 0.6, 40),
                              rng.uniform(-0.4, 0.4, 40),
                              rng.uniform(3.0, 6.0, 40)])
    angle = np.deg2rad(5.0)
    R = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                  [0.0, 1.0, 0.0],
                  [-np.sin(angle), 0.0, np.cos(angle)]])
    t = np.array([0.3, 0.02, 0.01])
    points2 = points @ R.T + t
    x1 = points[:, :2] / points[:, 2:3]
    x2 = points2[:, :2] / points2[:, 2:3]

    model, info = estimate_relative_pose(x1, x2, iterations=30,
                                         min_iterations=30, max_error=1e-3,
                                         seed=0)
    assert info['num_inliers'] == len(x1)
    assert model['R'] == pytest.approx(R, abs=1e-6)
