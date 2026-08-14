"""Relative pose estimation with monocular depth estimates (MDE priors).

Three pipelines on the shared LO-RANSAC engine, following poselib's
monodepth estimators:

- `estimate_relative_pose_with_monodepth`: calibrated cameras. With
  `estimate_shift=False` (scale-invariant depths) the minimal solver is
  P3P on the depth-induced 3D points; with `estimate_shift=True`
  (affine-invariant depths) the 3-point solver additionally recovers one
  depth shift per image.
- `estimate_shared_focal_relative_pose_with_monodepth`: unknown focal
  length shared by both cameras, pixel coordinates with known principal
  points.
- `estimate_varying_focal_relative_pose_with_monodepth`: two unknown focal
  lengths.

Scoring is the truncated Sampson error (threshold `max_error`, in pixels
for the focal problems and calibrated units for the calibrated one);
local optimization minimizes the hybrid cost of the Sampson error
(weighted by `weight_sampson`) and the symmetric monodepth reprojection
error, whose threshold `max_reproj_error` sets the relative weighting
`scale_reproj = max_error^2 / max_reproj_error^2` like poselib's
`max_errors = [reproj, epipolar]` pair.

Each entry point ends with a final refinement of the same hybrid cost on
the RANSAC inliers only, under the robust loss named by `final_loss`
('truncated', 'cauchy' - the default - or 'truncated_cauchy'; see
refiners/losses.py).
"""

import numpy as np

from fastpose.estimators.ransac import RansacEstimator
from fastpose.estimators.utils import build_info, failure_info, unproject_pair
from fastpose.refiners.losses import get_loss
from fastpose.refiners.monodepth import (LMMonoDepthPoseRefiner,
                                LMMonoDepthSharedFocalPoseRefiner,
                                LMMonoDepthShiftPoseRefiner,
                                LMMonoDepthVaryingFocalPoseRefiner)
from fastpose.scorers.sampson import (MonoDepthFocalPoseSampsonScorer,
                             MonoDepthPoseSampsonScorer)
from fastpose.solvers.monodepth import (MonoDepthP3PSolver, MonoDepthSharedFocalSolver,
                               MonoDepthShiftSolver,
                               MonoDepthVaryingFocalSolver)

_estimators = {}
_final_refiners = {}

_REFINER_CLS = {
    'calibrated': LMMonoDepthPoseRefiner,
    'calibrated-shift': LMMonoDepthShiftPoseRefiner,
    'shared-focal': LMMonoDepthSharedFocalPoseRefiner,
    'varying-focal': LMMonoDepthVaryingFocalPoseRefiner,
}


def _get_estimator(kind):
    if kind not in _estimators:
        if kind == 'calibrated':
            _estimators[kind] = RansacEstimator(MonoDepthP3PSolver(),
                                                MonoDepthPoseSampsonScorer(),
                                                LMMonoDepthPoseRefiner())
        elif kind == 'calibrated-shift':
            _estimators[kind] = RansacEstimator(MonoDepthShiftSolver(),
                                                MonoDepthPoseSampsonScorer(),
                                                LMMonoDepthShiftPoseRefiner())
        elif kind == 'shared-focal':
            _estimators[kind] = RansacEstimator(
                MonoDepthSharedFocalSolver(),
                MonoDepthFocalPoseSampsonScorer(),
                LMMonoDepthSharedFocalPoseRefiner())
        else:
            _estimators[kind] = RansacEstimator(
                MonoDepthVaryingFocalSolver(),
                MonoDepthFocalPoseSampsonScorer(),
                LMMonoDepthVaryingFocalPoseRefiner())
    return _estimators[kind]


def _get_final_refiner(kind, loss):
    # refiner for the final polish pass on RANSAC inliers only; `loss` is an
    # already-resolved Loss object (see refiners/losses.py). One refiner -
    # and one compiled LM kernel - is cached per (problem, loss type)
    key = (kind, type(loss))
    if key not in _final_refiners:
        _final_refiners[key] = _REFINER_CLS[kind](loss=loss)
    return _final_refiners[key]


def _monodepth_data(x1, x2, d1, d2, scale_reproj, weight_sampson):
    return (np.ascontiguousarray(x1[:, 0]), np.ascontiguousarray(x1[:, 1]),
            np.ascontiguousarray(x2[:, 0]), np.ascontiguousarray(x2[:, 1]),
            np.ascontiguousarray(d1), np.ascontiguousarray(d2),
            float(scale_reproj), float(weight_sampson))


def _check_inputs(x1, x2, d1, d2):
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    d1 = np.ascontiguousarray(d1, dtype=np.float64).ravel()
    d2 = np.ascontiguousarray(d2, dtype=np.float64).ravel()
    if not (len(x1) == len(x2) == len(d1) == len(d2)):
        raise ValueError("x1, x2, d1 and d2 must have the same length")
    return x1, x2, d1, d2


def _scale_reproj(max_error, max_reproj_error):
    if max_reproj_error is None or max_reproj_error <= 0.0:
        return 0.0
    return (max_error * max_error) / (max_reproj_error * max_reproj_error)


def _principal_point(pp):
    if pp is None:
        return np.zeros(2, dtype=np.float64)
    arr = np.asarray(pp, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError("principal points must be length-2 arrays")
    return arr


def estimate_relative_pose_with_monodepth(
        x1, x2, d1, d2, camera1=None, camera2=None, estimate_shift=False,
        iterations=1000, max_error=0.002, max_reproj_error=0.016,
        weight_sampson=1.0, seed=4578, min_iterations=None,
        success_prob=0.9999, lo_iterations=25,
        final_refinement_iterations=100, final_loss='cauchy'):
    # params:
    # x1, x2 - (n, 2) arrays of *calibrated* image points
    # d1, d2 - (n,) monocular depths per image (scale-invariant, or
    #          affine-invariant with estimate_shift=True)
    # camera1, camera2 - optional cameras (3x3 intrinsic matrix K, or a
    #          poselib.Camera) used to unproject x1/x2 from pixel
    #          coordinates into calibrated points; None (default) keeps x1,
    #          x2 as already-calibrated input. When supplied, max_error and
    #          max_reproj_error are also rescaled by the average focal
    #          length of the two cameras
    # estimate_shift - also estimate one depth shift per image (the depths
    #          enter as d + shift)
    # max_error - truncated Sampson threshold in calibrated units
    # max_reproj_error - reprojection threshold (same units) that sets the
    #          hybrid LO weighting; None or 0 disables the reprojection term
    # weight_sampson - weight of the Sampson term in the hybrid LO cost
    # final_refinement_iterations - LM step budget for the final robust-loss
    #          polish pass on the RANSAC inliers; independent of
    #          lo_iterations. Defaults to 100; 0 disables the pass
    # final_loss - robust loss minimized by that final pass: 'truncated',
    #          'cauchy' (default) or 'truncated_cauchy'; a Loss object from
    #          refiners/losses.py is accepted too. The first call for a given
    #          loss compiles its LM kernel
    # returns (model, info) with model = {'R', 't', 'scale', 'shift1',
    # 'shift2'} (scale * (d2 + shift2) * x2h = R ((d1 + shift1) * x1h) + t
    # for inliers) and info = {'inliers', 'num_inliers', 'model_score',
    # 'iterations', 'refinements'}; on total failure model holds the
    # identity pose with scale=1.0, shift1=shift2=0.0 and
    # info['num_inliers'] is 0
    x1, x2, d1, d2 = _check_inputs(x1, x2, d1, d2)
    loss = get_loss(final_loss)
    x1, x2, max_error, max_reproj_error = unproject_pair(
        camera1, camera2, x1, x2, max_error, max_reproj_error)
    data = _monodepth_data(x1, x2, d1, d2,
                           _scale_reproj(max_error, max_reproj_error),
                           weight_sampson)

    kind = 'calibrated-shift' if estimate_shift else 'calibrated'
    estimator = _get_estimator(kind)
    model, _, num_inliers, ransac_iterations = estimator.estimate(
        data, len(x1), max_error, iterations=iterations,
        min_iterations=min_iterations, success_prob=success_prob,
        lo_iterations=lo_iterations, seed=seed)

    if num_inliers == 0:
        return ({'R': np.eye(3), 't': np.zeros(3), 'scale': 1.0,
                'shift1': 0.0, 'shift2': 0.0},
                failure_info(len(x1), ransac_iterations))

    R = model[:9].reshape(3, 3).copy()
    t = model[9:12].copy()
    scale = float(model[12])
    shift1 = float(model[13])
    shift2 = float(model[14])
    score, inliers, num_inliers = MonoDepthPoseSampsonScorer.score_numpy(
        R, t, x1, x2, max_error)
    refined = False

    # final polish: robust-loss refinement restricted to the RANSAC inliers
    if final_refinement_iterations != 0 and num_inliers > 0:
        final_refiner = _get_final_refiner(kind, loss)
        inlier_data = _monodepth_data(
            x1[inliers], x2[inliers], d1[inliers], d2[inliers],
            _scale_reproj(max_error, max_reproj_error), weight_sampson)
        refined_model = np.empty(15)
        num_final_iterations = (final_refiner.num_iterations
                                if final_refinement_iterations is None
                                else final_refinement_iterations)
        if final_refiner.refine(inlier_data, model, refined_model, max_error ** 2,
                                num_final_iterations):
            R_c = refined_model[:9].reshape(3, 3).copy()
            t_c = refined_model[9:12].copy()
            scale_c = float(refined_model[12])
            shift1_c = float(refined_model[13])
            shift2_c = float(refined_model[14])
            score_c, inliers_c, num_inliers_c = MonoDepthPoseSampsonScorer.score_numpy(
                R_c, t_c, x1, x2, max_error)
            R, t, scale, shift1, shift2, score = (R_c, t_c, scale_c, shift1_c,
                                                   shift2_c, score_c)
            inliers, num_inliers = inliers_c, num_inliers_c
            refined = True

    return ({'R': R, 't': t, 'scale': scale, 'shift1': shift1, 'shift2': shift2},
            build_info(inliers, num_inliers, score, ransac_iterations, refined))


def estimate_shared_focal_relative_pose_with_monodepth(
        x1, x2, d1, d2, principal_point1=None, principal_point2=None,
        iterations=1000, max_error=2.0, max_reproj_error=16.0,
        weight_sampson=1.0, seed=4578, min_iterations=None,
        success_prob=0.9999, lo_iterations=25,
        final_refinement_iterations=100, final_loss='cauchy'):
    # x1, x2 in pixel coordinates, one unknown square-pixel focal length
    # shared by both cameras; principal_point* optional (cx, cy), zero if
    # omitted. Thresholds in pixels. final_refinement_iterations is the LM
    # step budget for the final robust-loss polish pass on the RANSAC
    # inliers, independent of lo_iterations; defaults to 100, 0 disables the
    # pass. final_loss picks that pass's robust loss - 'truncated', 'cauchy'
    # (default) or 'truncated_cauchy', or a Loss object from
    # refiners/losses.py. Returns (model, info) with model = {'R', 't', 'f',
    # 'scale'} and
    # info = {'inliers', 'num_inliers', 'model_score', 'iterations',
    # 'refinements'}; on total failure model holds the identity pose with
    # f=scale=1.0 and info['num_inliers'] is 0.
    x1, x2, d1, d2 = _check_inputs(x1, x2, d1, d2)
    loss = get_loss(final_loss)
    x1c = x1 - _principal_point(principal_point1)
    x2c = x2 - _principal_point(principal_point2)
    data = _monodepth_data(x1c, x2c, d1, d2,
                           _scale_reproj(max_error, max_reproj_error),
                           weight_sampson)

    estimator = _get_estimator('shared-focal')
    model, _, num_inliers, ransac_iterations = estimator.estimate(
        data, len(x1), max_error, iterations=iterations,
        min_iterations=min_iterations, success_prob=success_prob,
        lo_iterations=lo_iterations, seed=seed)

    if num_inliers == 0:
        return ({'R': np.eye(3), 't': np.zeros(3), 'f': 1.0, 'scale': 1.0},
                failure_info(len(x1), ransac_iterations))

    R = model[:9].reshape(3, 3).copy()
    t = model[9:12].copy()
    f = float(model[12])
    scale = float(model[14])
    score, inliers, num_inliers = MonoDepthFocalPoseSampsonScorer.score_numpy(
        R, t, f, f, x1c, x2c, max_error)
    refined = False

    # final polish: robust-loss refinement restricted to the RANSAC inliers
    if final_refinement_iterations != 0 and num_inliers > 0:
        final_refiner = _get_final_refiner('shared-focal', loss)
        inlier_data = _monodepth_data(
            x1c[inliers], x2c[inliers], d1[inliers], d2[inliers],
            _scale_reproj(max_error, max_reproj_error), weight_sampson)
        refined_model = np.empty(15)
        num_final_iterations = (final_refiner.num_iterations
                                if final_refinement_iterations is None
                                else final_refinement_iterations)
        if final_refiner.refine(inlier_data, model, refined_model, max_error ** 2,
                                num_final_iterations):
            R_c = refined_model[:9].reshape(3, 3).copy()
            t_c = refined_model[9:12].copy()
            f_c = float(refined_model[12])
            scale_c = float(refined_model[14])
            score_c, inliers_c, num_inliers_c = MonoDepthFocalPoseSampsonScorer.score_numpy(
                R_c, t_c, f_c, f_c, x1c, x2c, max_error)
            R, t, f, scale, score = R_c, t_c, f_c, scale_c, score_c
            inliers, num_inliers = inliers_c, num_inliers_c
            refined = True

    return ({'R': R, 't': t, 'f': f, 'scale': scale},
            build_info(inliers, num_inliers, score, ransac_iterations, refined))


def estimate_varying_focal_relative_pose_with_monodepth(
        x1, x2, d1, d2, principal_point1=None, principal_point2=None,
        iterations=1000, max_error=2.0, max_reproj_error=16.0,
        weight_sampson=1.0, seed=4578, min_iterations=None,
        success_prob=0.9999, lo_iterations=25,
        final_refinement_iterations=100, final_loss='cauchy'):
    # x1, x2 in pixel coordinates, one unknown square-pixel focal length per
    # camera; principal_point* optional (cx, cy), zero if omitted.
    # Thresholds in pixels. final_refinement_iterations is the LM step
    # budget for the final robust-loss polish pass on the RANSAC inliers,
    # independent of lo_iterations; defaults to 100, 0 disables the pass.
    # final_loss picks that pass's robust loss - 'truncated', 'cauchy'
    # (default) or 'truncated_cauchy', or a Loss object from
    # refiners/losses.py.
    # Returns (model, info) with model = {'R', 't', 'f1', 'f2', 'scale'} and
    # info = {'inliers', 'num_inliers', 'model_score', 'iterations',
    # 'refinements'}; on total failure model holds the identity pose with
    # f1=f2=scale=1.0 and info['num_inliers'] is 0.
    x1, x2, d1, d2 = _check_inputs(x1, x2, d1, d2)
    loss = get_loss(final_loss)
    x1c = x1 - _principal_point(principal_point1)
    x2c = x2 - _principal_point(principal_point2)
    data = _monodepth_data(x1c, x2c, d1, d2,
                           _scale_reproj(max_error, max_reproj_error),
                           weight_sampson)

    estimator = _get_estimator('varying-focal')
    model, _, num_inliers, ransac_iterations = estimator.estimate(
        data, len(x1), max_error, iterations=iterations,
        min_iterations=min_iterations, success_prob=success_prob,
        lo_iterations=lo_iterations, seed=seed)

    if num_inliers == 0:
        return ({'R': np.eye(3), 't': np.zeros(3), 'f1': 1.0, 'f2': 1.0,
                'scale': 1.0},
                failure_info(len(x1), ransac_iterations))

    R = model[:9].reshape(3, 3).copy()
    t = model[9:12].copy()
    f1 = float(model[12])
    f2 = float(model[13])
    scale = float(model[14])
    score, inliers, num_inliers = MonoDepthFocalPoseSampsonScorer.score_numpy(
        R, t, f1, f2, x1c, x2c, max_error)
    refined = False

    # final polish: robust-loss refinement restricted to the RANSAC inliers
    if final_refinement_iterations != 0 and num_inliers > 0:
        final_refiner = _get_final_refiner('varying-focal', loss)
        inlier_data = _monodepth_data(
            x1c[inliers], x2c[inliers], d1[inliers], d2[inliers],
            _scale_reproj(max_error, max_reproj_error), weight_sampson)
        refined_model = np.empty(15)
        num_final_iterations = (final_refiner.num_iterations
                                if final_refinement_iterations is None
                                else final_refinement_iterations)
        if final_refiner.refine(inlier_data, model, refined_model, max_error ** 2,
                                num_final_iterations):
            R_c = refined_model[:9].reshape(3, 3).copy()
            t_c = refined_model[9:12].copy()
            f1_c = float(refined_model[12])
            f2_c = float(refined_model[13])
            scale_c = float(refined_model[14])
            score_c, inliers_c, num_inliers_c = MonoDepthFocalPoseSampsonScorer.score_numpy(
                R_c, t_c, f1_c, f2_c, x1c, x2c, max_error)
            R, t, f1, f2, scale, score = R_c, t_c, f1_c, f2_c, scale_c, score_c
            inliers, num_inliers = inliers_c, num_inliers_c
            refined = True

    return ({'R': R, 't': t, 'f1': f1, 'f2': f2, 'scale': scale},
            build_info(inliers, num_inliers, score, ransac_iterations, refined))
