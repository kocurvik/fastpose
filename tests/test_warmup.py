"""`fastpose-warmup` has to cover every kernel a caller can reach, not just
the default ones.

Numba's on-disk cache is not safe against concurrent writers, so anything the
warmup misses gets compiled by every worker of a `multiprocessing.Pool` at
once - which is how an index entry ends up pointing at another key's compiled
code. The monodepth entry points compile one final-refinement kernel per
`final_loss`, so each selectable loss needs its own warmup run.
"""

import importlib

import pytest

from fastpose.estimators import monodepth as md_est
from fastpose.estimators.warmup import MONODEPTH_FINAL_LOSSES, warmup
from fastpose.refiners.losses import LOSSES, TruncatedLoss
from fastpose.refiners.monodepth import (LMMonoDepthPoseRefiner,
                                         _refine_monodepth_lm)

# problem name -> the estimator module that owns its final-polish refiner
POSE_PROBLEMS = {
    'fundamental': 'fastpose.estimators.fundamental',
    'essential': 'fastpose.estimators.essential',
    'absolute': 'fastpose.estimators.absolute',
    'absolute-focal': 'fastpose.estimators.absolute_focal',
    'varying-focal': 'fastpose.estimators.varying_focal',
    'shared-focal': 'fastpose.estimators.shared_focal',
}


@pytest.mark.parametrize('problem', sorted(POSE_PROBLEMS))
def test_warmup_scene_reaches_the_final_polish_refiner(problem):
    # The polish pass compiles a second LM kernel, and it only runs when the
    # estimate found inliers - so a warmup scene that is degenerate for a
    # solver leaves that kernel cold *silently*. It did: the absolute-pose
    # scene used to put the world origin at the camera, giving the pose
    # (I, 0), which P4Pf cannot solve, so absolute-focal found no inliers at
    # any iteration count and warmed neither of its LM kernels.
    module = importlib.import_module(POSE_PROBLEMS[problem])
    module._final_refiner = None
    warmup(problem=problem, iterations=3, lo_iterations=1,
           final_refinement_iterations=1)
    assert module._final_refiner is not None, (
        f"{problem}: the warmup scene never reached the final polish pass")


def test_warmup_covers_every_selectable_monodepth_final_loss():
    md_est._final_refiners.clear()
    warmup(problem="monodepth", iterations=1, lo_iterations=1)

    expected = {(kind, LOSSES[name])
                for kind in md_est._REFINER_CLS
                for name in MONODEPTH_FINAL_LOSSES}
    missing = expected - set(md_est._final_refiners)
    assert not missing, "warmup left these final refiners cold: %s" % (missing,)


def test_truncated_cauchy_is_one_of_them():
    # the loss this warmup was extended for; guards against the derivation in
    # warmup.py silently dropping to an empty tuple
    assert 'truncated_cauchy' in MONODEPTH_FINAL_LOSSES


def test_truncated_needs_no_warmup_of_its_own():
    # why MONODEPTH_FINAL_LOSSES omits it: a truncated-loss refiner reuses the
    # local-optimization kernel that the RANSAC loop already compiles
    assert LOSSES['truncated'] is TruncatedLoss
    assert 'truncated' not in MONODEPTH_FINAL_LOSSES
    assert LMMonoDepthPoseRefiner(loss='truncated').refine is _refine_monodepth_lm
