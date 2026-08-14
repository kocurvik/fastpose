"""Command-line warmup for the cached Numba kernels.

Warming up front matters for more than latency: numba's on-disk cache is not
safe against concurrent writers (the index read, data-file name allocation and
the two file writes in `_IndexDataCacheFile.save` are not one atomic step), so
several worker processes compiling the same kernel at once can leave an index
entry pointing at another key's compiled code. Run `fastpose-warmup`, or call
`warmup()` in the parent process before creating a `multiprocessing.Pool`, so
the workers only ever read the cache - or, with the default `fork` start
method, inherit the compiled kernels and never touch it at all.
"""

import argparse
import time

import numpy as np

from fastpose.estimators.absolute import estimate_absolute_pose
from fastpose.estimators.absolute_focal import estimate_absolute_pose_with_focal
from fastpose.estimators.essential import estimate_relative_pose
from fastpose.estimators.fundamental import estimate_fundamental
from fastpose.estimators.monodepth import (
    estimate_relative_pose_with_monodepth,
    estimate_shared_focal_relative_pose_with_monodepth,
    estimate_varying_focal_relative_pose_with_monodepth)
from fastpose.estimators.monodepth import (_get_final_refiner, _monodepth_data,
                                           _scale_reproj)
from fastpose.estimators.shared_focal import estimate_relative_pose_with_shared_focal
from fastpose.estimators.varying_focal import estimate_relative_pose_with_varying_focals
from fastpose.refiners.losses import LOSSES, TruncatedLoss, get_loss

# every `final_loss` the monodepth entry points accept that brings its own LM
# kernel, so a new loss in refiners/losses.py is warmed without touching this
# file. TruncatedLoss is excluded: its refiners keep the local-optimization
# kernel the RANSAC loop compiles anyway.
MONODEPTH_FINAL_LOSSES = tuple(name for name, cls in LOSSES.items()
                               if cls is not TruncatedLoss)


def _warm_monodepth_final_refiners(x1, x2, depth1, depth2, R, t,
                                   num_iterations):
    # The estimator calls only reach the final polish pass when RANSAC happens
    # to find inliers on the synthetic scene - with a low --iterations the
    # shift and varying-focal variants find none and their kernels stay cold.
    # A cold kernel is one that every worker of a multiprocessing pool would
    # compile (and race to cache) at once, so drive the refiners directly,
    # once per (problem, loss): the kernel compiles on the call regardless of
    # what it returns.
    num_iterations = max(1, num_iterations)
    refined = np.empty(15)

    pose = np.empty(15)
    pose[:9] = R.ravel()
    pose[9:12] = t

    calibrated_model = pose.copy()
    calibrated_model[12] = 1.0   # scale
    calibrated_model[13] = 0.0   # shift1
    calibrated_model[14] = 0.0   # shift2
    calibrated_data = _monodepth_data(x1, x2, depth1, depth2,
                                      _scale_reproj(0.002, 0.016), 1.0)

    focal = 1000.0
    focal_model = pose.copy()
    focal_model[12] = focal      # f1
    focal_model[13] = focal      # f2
    focal_model[14] = 1.0        # scale
    focal_data = _monodepth_data(x1 * focal, x2 * focal, depth1, depth2,
                                 _scale_reproj(2.0, 16.0), 1.0)

    problems = (
        ('calibrated', calibrated_data, calibrated_model, 0.002),
        ('calibrated-shift', calibrated_data, calibrated_model, 0.002),
        ('shared-focal', focal_data, focal_model, 2.0),
        ('varying-focal', focal_data, focal_model, 2.0),
    )
    for final_loss in MONODEPTH_FINAL_LOSSES:
        loss = get_loss(final_loss)
        for kind, data, model, max_error in problems:
            refiner = _get_final_refiner(kind, loss)
            refiner.refine(data, model, refined, max_error ** 2,
                           num_iterations)


def _synthetic_correspondences(num_points=32):
    rng = np.random.default_rng(0)
    points = np.empty((num_points, 3), dtype=np.float64)
    points[:, 0] = rng.uniform(-0.8, 0.8, num_points)
    points[:, 1] = rng.uniform(-0.5, 0.5, num_points)
    points[:, 2] = rng.uniform(3.0, 6.0, num_points)

    angle = np.deg2rad(4.0)
    ca = np.cos(angle)
    sa = np.sin(angle)
    R = np.array([[ca, 0.0, sa],
                  [0.0, 1.0, 0.0],
                  [-sa, 0.0, ca]], dtype=np.float64)
    t = np.array([0.25, 0.02, 0.01], dtype=np.float64)

    points2 = points @ R.T + t
    x1 = points[:, :2] / points[:, 2:3]
    x2 = points2[:, :2] / points2[:, 2:3]
    return np.ascontiguousarray(x1), np.ascontiguousarray(x2)


def warmup(problem="all", iterations=3, lo_iterations=1,
           final_refinement_iterations=1):
    x1, x2 = _synthetic_correspondences()

    if problem in ("all", "fundamental"):
        pixel_x1 = x1 * 1000.0 + 500.0
        pixel_x2 = x2 * 1000.0 + 500.0
        estimate_fundamental(
            pixel_x1,
            pixel_x2,
            iterations=iterations,
            min_iterations=iterations,
            max_error=2.0,
            seed=0,
            lo_iterations=lo_iterations,
        )

    if problem in ("all", "essential"):
        estimate_relative_pose(
            x1,
            x2,
            iterations=iterations,
            min_iterations=iterations,
            max_error=0.002,
            seed=0,
            lo_iterations=lo_iterations,
        )

    if problem in ("all", "absolute"):
        rng = np.random.default_rng(0)
        depth = rng.uniform(3.0, 6.0, size=len(x1))
        world = np.column_stack([x1 * depth[:, None], depth])
        estimate_absolute_pose(
            x1,
            world,
            iterations=iterations,
            min_iterations=iterations,
            max_error=0.002,
            seed=0,
            lo_iterations=lo_iterations,
        )

    if problem in ("all", "absolute-focal"):
        rng = np.random.default_rng(0)
        depth = rng.uniform(3.0, 6.0, size=len(x1))
        world = np.column_stack([x1 * depth[:, None], depth])
        estimate_absolute_pose_with_focal(
            x1 * 1000.0,
            world,
            iterations=iterations,
            min_iterations=iterations,
            max_error=2.0,
            seed=0,
            lo_iterations=lo_iterations,
        )

    if problem in ("all", "varying-focal"):
        pp1 = np.array([500.0, 480.0])
        pp2 = np.array([620.0, 510.0])
        pixel_x1 = x1 * 800.0 + pp1
        pixel_x2 = x2 * 1300.0 + pp2
        estimate_relative_pose_with_varying_focals(
            pixel_x1,
            pixel_x2,
            pp1,
            pp2,
            iterations=iterations,
            min_iterations=iterations,
            max_error=2.0,
            seed=0,
            lo_iterations=lo_iterations,
        )

    if problem in ("all", "shared-focal"):
        pp1 = np.array([500.0, 480.0])
        pp2 = np.array([620.0, 510.0])
        pixel_x1 = x1 * 1000.0 + pp1
        pixel_x2 = x2 * 1000.0 + pp2
        estimate_relative_pose_with_shared_focal(
            pixel_x1,
            pixel_x2,
            pp1,
            pp2,
            iterations=iterations,
            min_iterations=iterations,
            max_error=2.0,
            seed=0,
            lo_iterations=lo_iterations,
        )

    if problem in ("all", "monodepth"):
        rng = np.random.default_rng(0)
        depth1 = rng.uniform(3.0, 6.0, size=len(x1))
        world = np.column_stack([x1 * depth1[:, None], depth1])
        # exact depths of the warmup pose in camera 2 (R, t from
        # _synthetic_correspondences)
        angle = np.deg2rad(4.0)
        ca = np.cos(angle)
        sa = np.sin(angle)
        R = np.array([[ca, 0.0, sa],
                      [0.0, 1.0, 0.0],
                      [-sa, 0.0, ca]], dtype=np.float64)
        t = np.array([0.25, 0.02, 0.01], dtype=np.float64)
        depth2 = (world @ R.T + t)[:, 2]

        for estimate_shift in (False, True):
            estimate_relative_pose_with_monodepth(
                x1,
                x2,
                depth1,
                depth2,
                estimate_shift=estimate_shift,
                iterations=iterations,
                min_iterations=iterations,
                max_error=0.002,
                seed=0,
                lo_iterations=lo_iterations,
                final_refinement_iterations=final_refinement_iterations,
            )
        estimate_shared_focal_relative_pose_with_monodepth(
            x1 * 1000.0,
            x2 * 1000.0,
            depth1,
            depth2,
            iterations=iterations,
            min_iterations=iterations,
            max_error=2.0,
            seed=0,
            lo_iterations=lo_iterations,
            final_refinement_iterations=final_refinement_iterations,
        )
        estimate_varying_focal_relative_pose_with_monodepth(
            x1 * 1000.0,
            x2 * 1000.0,
            depth1,
            depth2,
            iterations=iterations,
            min_iterations=iterations,
            max_error=2.0,
            seed=0,
            lo_iterations=lo_iterations,
            final_refinement_iterations=final_refinement_iterations,
        )
        # one LM kernel is compiled per `final_loss`, and the calls above only
        # reach the ones the RANSAC run happened to need
        _warm_monodepth_final_refiners(x1, x2, depth1, depth2, R, t,
                                       final_refinement_iterations)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compile and cache fastpose's Numba kernels with small synthetic inputs.",
    )
    parser.add_argument(
        "--problem",
        choices=("all", "fundamental", "essential", "absolute",
                 "absolute-focal", "varying-focal", "shared-focal",
                 "monodepth"),
        default="all",
        help="Subset of kernels to warm up.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="RANSAC iterations to run for each selected problem.",
    )
    parser.add_argument(
        "--lo-iterations",
        type=int,
        default=1,
        help="Local optimization iterations to run during warmup.",
    )
    parser.add_argument(
        "--final-refinement-iterations",
        type=int,
        default=1,
        help="LM steps of the final polish pass to run during warmup; the "
             "pass is compiled per final_loss, so 0 leaves those kernels cold.",
    )
    args = parser.parse_args(argv)

    start = time.perf_counter()
    warmup(args.problem, args.iterations, args.lo_iterations,
           args.final_refinement_iterations)
    elapsed = time.perf_counter() - start
    print(f"fastpose warmup complete in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
