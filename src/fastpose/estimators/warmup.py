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
from tqdm import tqdm

from fastpose.estimators.absolute import estimate_absolute_pose
from fastpose.estimators.absolute_focal import estimate_absolute_pose_with_focal
from fastpose.estimators.essential import estimate_relative_pose
from fastpose.estimators.fundamental import estimate_fundamental
from fastpose.estimators.homography import estimate_homography
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


def _synthetic_planar_correspondences(num_points=32):
    """Pixel correspondences of a *planar* scene, for the homography warmup.

    `_synthetic_correspondences` draws points at random depths, which no
    homography relates - the 4-point solver would run but never find a model
    with enough inliers to trigger local optimization or the final polish, and
    those kernels would silently stay cold. These lie on one tilted plane, so
    an exact homography exists.
    """
    rng = np.random.default_rng(0)
    x1 = rng.uniform(-0.4, 0.4, size=(num_points, 2))
    n = np.array([0.3, -0.2, 1.0])
    n /= np.linalg.norm(n)
    d = 5.0
    R, t = _synthetic_pose()
    H = R + np.outer(t, n) / d
    x1h = np.column_stack([x1, np.ones(num_points)])
    x2h = x1h @ H.T
    x2 = x2h[:, :2] / x2h[:, 2:3]
    return (np.ascontiguousarray(x1 * 1000.0 + 500.0),
            np.ascontiguousarray(x2 * 1000.0 + 500.0))


def _synthetic_pose():
    # the (R, t) `_synthetic_correspondences` generates its second view with
    angle = np.deg2rad(4.0)
    ca = np.cos(angle)
    sa = np.sin(angle)
    R = np.array([[ca, 0.0, sa],
                  [0.0, 1.0, 0.0],
                  [-sa, 0.0, ca]], dtype=np.float64)
    t = np.array([0.25, 0.02, 0.01], dtype=np.float64)
    return R, t


def _synthetic_depths(x1):
    """Per-image depths of the synthetic scene, for the monodepth warmups.

    These points stay in the camera-1 frame - that is the frame monodepth
    works in, unlike the absolute-pose warmups (see _synthetic_world_points).
    """
    rng = np.random.default_rng(0)
    depth1 = rng.uniform(3.0, 6.0, size=len(x1))
    world = np.column_stack([x1 * depth1[:, None], depth1])
    R, t = _synthetic_pose()
    depth2 = (world @ R.T + t)[:, 2]
    return depth1, depth2


def _synthetic_world_points(x1):
    """World points for the absolute-pose warmups, in a frame that is not the
    camera's.

    The obvious construction - `column_stack([x1 * depth, depth])` - puts the
    world origin *at* the camera, so the pose to recover is (I, 0). P3P copes;
    **P4Pf does not**, and a zero translation made every hypothesis fail, so
    the absolute-focal warmup silently reached neither the local-optimization
    nor the final-polish kernel. Mapping the camera-frame points through the
    synthetic pose gives a nonzero translation and fixes both backends.
    """
    rng = np.random.default_rng(0)
    depth = rng.uniform(3.0, 6.0, size=len(x1))
    X_cam = np.column_stack([x1 * depth[:, None], depth])
    R, t = _synthetic_pose()
    return np.ascontiguousarray((X_cam - t) @ R)   # rows are R^T (X_cam - t)


def _warmup_steps(problem, iterations, lo_iterations,
                  final_refinement_iterations):
    """`(label, thunk)` pairs for the kernels `problem` selects, in run order.

    Splitting the warmup into named steps is what lets the CLI show a progress
    bar: the compile times are wildly uneven (the monodepth kernels dominate),
    so a bar over the steps is the only feedback available while a single
    `numba` compilation blocks for seconds.
    """
    x1, x2 = _synthetic_correspondences()
    steps = []

    def selected(name):
        return problem in ("all", name)

    if selected("fundamental"):
        pixel_x1 = x1 * 1000.0 + 500.0
        pixel_x2 = x2 * 1000.0 + 500.0
        steps.append(("fundamental", lambda: estimate_fundamental(
            pixel_x1,
            pixel_x2,
            iterations=iterations,
            min_iterations=iterations,
            max_error=2.0,
            seed=0,
            lo_iterations=lo_iterations,
        )))

    if selected("homography"):
        # a planar scene, so the 4-point solver has a homography to find and
        # the local-optimization and polish kernels are actually reached
        h_x1, h_x2 = _synthetic_planar_correspondences()
        steps.append(("homography", lambda: estimate_homography(
            h_x1,
            h_x2,
            iterations=iterations,
            min_iterations=iterations,
            max_error=2.0,
            seed=0,
            lo_iterations=lo_iterations,
            final_refinement_iterations=final_refinement_iterations,
        )))

    if selected("essential"):
        steps.append(("essential", lambda: estimate_relative_pose(
            x1,
            x2,
            iterations=iterations,
            min_iterations=iterations,
            max_error=0.002,
            seed=0,
            lo_iterations=lo_iterations,
        )))

    if selected("absolute"):
        world = _synthetic_world_points(x1)
        steps.append(("absolute", lambda: estimate_absolute_pose(
            x1,
            world,
            iterations=iterations,
            min_iterations=iterations,
            max_error=0.002,
            seed=0,
            lo_iterations=lo_iterations,
        )))

    if selected("absolute-focal"):
        world = _synthetic_world_points(x1)
        steps.append(("absolute-focal",
                      lambda: estimate_absolute_pose_with_focal(
                          x1 * 1000.0,
                          world,
                          iterations=iterations,
                          min_iterations=iterations,
                          max_error=2.0,
                          seed=0,
                          lo_iterations=lo_iterations,
                      )))

    if selected("varying-focal"):
        pp1 = np.array([500.0, 480.0])
        pp2 = np.array([620.0, 510.0])
        pixel_x1 = x1 * 800.0 + pp1
        pixel_x2 = x2 * 1300.0 + pp2
        steps.append(("varying-focal",
                      lambda: estimate_relative_pose_with_varying_focals(
                          pixel_x1,
                          pixel_x2,
                          pp1,
                          pp2,
                          iterations=iterations,
                          min_iterations=iterations,
                          max_error=2.0,
                          seed=0,
                          lo_iterations=lo_iterations,
                      )))

    if selected("shared-focal"):
        pp1 = np.array([500.0, 480.0])
        pp2 = np.array([620.0, 510.0])
        pixel_x1 = x1 * 1000.0 + pp1
        pixel_x2 = x2 * 1000.0 + pp2
        steps.append(("shared-focal",
                      lambda: estimate_relative_pose_with_shared_focal(
                          pixel_x1,
                          pixel_x2,
                          pp1,
                          pp2,
                          iterations=iterations,
                          min_iterations=iterations,
                          max_error=2.0,
                          seed=0,
                          lo_iterations=lo_iterations,
                      )))

    if selected("monodepth"):
        depth1, depth2 = _synthetic_depths(x1)
        R, t = _synthetic_pose()

        for estimate_shift in (False, True):
            label = "monodepth-shift" if estimate_shift else "monodepth"
            steps.append((label,
                          lambda estimate_shift=estimate_shift:
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
                          )))
        steps.append(("monodepth-shared-focal",
                      lambda: estimate_shared_focal_relative_pose_with_monodepth(
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
                      )))
        steps.append(("monodepth-varying-focal",
                      lambda: estimate_varying_focal_relative_pose_with_monodepth(
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
                      )))
        # one LM kernel is compiled per `final_loss`, and the calls above only
        # reach the ones the RANSAC run happened to need
        steps.append(("monodepth-final-refiners",
                      lambda: _warm_monodepth_final_refiners(
                          x1, x2, depth1, depth2, R, t,
                          final_refinement_iterations)))

    return steps


def _cuda_warmup_steps(problem, iterations, lo_iterations,
                       final_refinement_iterations):
    """Warmup steps for the CUDA backend, or `[]` when no device is usable.

    The GPU kernels are cached on disk the same way the CPU ones are
    (`cuda.jit(cache=True)`), so this buys the same thing `fastpose-warmup`
    buys elsewhere. It matters more here than on the CPU: a cold GPU estimate
    compiles the batched solver, the scorer and *two* LM kernels - one per
    loss, since the RANSAC-internal local optimization and the final polish
    pass use different ones - which is several seconds and is easy to mistake
    for the steady-state cost when benchmarking.

    `lo_iterations` and `final_refinement_iterations` are floored at 1: a zero
    skips the pass entirely and leaves its kernel cold, which is the one thing
    this function exists to prevent.
    """
    from fastpose import cuda as cuda_backend
    if not cuda_backend.is_available():
        return []

    x1, x2 = _synthetic_correspondences()
    world = _synthetic_world_points(x1)
    pp1 = np.array([500.0, 480.0])
    pp2 = np.array([620.0, 510.0])
    lo = max(1, lo_iterations)
    final = max(1, final_refinement_iterations)
    # a batch smaller than the point count keeps the warmup quick while still
    # compiling every kernel the driver launches, including a partial round
    common = dict(iterations=iterations, min_iterations=iterations, seed=0,
                  lo_iterations=lo, final_refinement_iterations=final,
                  device='cuda', batch=64)
    steps = []

    def selected(name):
        return problem in ("all", name)

    if selected("essential"):
        steps.append(("essential-cuda", lambda: estimate_relative_pose(
            x1, x2, max_error=0.002, **common)))

    if selected("absolute"):
        steps.append(("absolute-cuda", lambda: estimate_absolute_pose(
            x1, world, max_error=0.002, **common)))

    if selected("absolute-focal"):
        steps.append(("absolute-focal-cuda",
                      lambda: estimate_absolute_pose_with_focal(
                          x1 * 1000.0, world, max_error=2.0, **common)))

    if selected("fundamental"):
        steps.append(("fundamental-cuda", lambda: estimate_fundamental(
            x1 * 1000.0 + 500.0, x2 * 1000.0 + 500.0, max_error=2.0,
            **common)))

    if selected("homography"):
        h_x1, h_x2 = _synthetic_planar_correspondences()
        steps.append(("homography-cuda", lambda: estimate_homography(
            h_x1, h_x2, max_error=2.0, **common)))

    if selected("varying-focal"):
        steps.append(("varying-focal-cuda",
                      lambda: estimate_relative_pose_with_varying_focals(
                          x1 * 800.0 + pp1, x2 * 1300.0 + pp2, pp1, pp2,
                          max_error=2.0, **common)))

    if selected("shared-focal"):
        # much the slowest of the seven to compile: the 6-point solve kernel
        # inlines a 31x46 elimination template and a 15x15 Danilevsky chain
        steps.append(("shared-focal-cuda",
                      lambda: estimate_relative_pose_with_shared_focal(
                          x1 * 1000.0 + pp1, x2 * 1000.0 + pp2, pp1, pp2,
                          max_error=2.0, **common)))

    if selected("monodepth"):
        depth1, depth2 = _synthetic_depths(x1)
        for estimate_shift in (False, True):
            label = ("monodepth-shift-cuda" if estimate_shift
                     else "monodepth-cuda")
            steps.append((label,
                          lambda estimate_shift=estimate_shift:
                          estimate_relative_pose_with_monodepth(
                              x1, x2, depth1, depth2,
                              estimate_shift=estimate_shift, max_error=0.002,
                              **common)))
        steps.append(("monodepth-shared-focal-cuda",
                      lambda: estimate_shared_focal_relative_pose_with_monodepth(
                          x1 * 1000.0, x2 * 1000.0, depth1, depth2,
                          max_error=2.0, **common)))
        steps.append(("monodepth-varying-focal-cuda",
                      lambda: estimate_varying_focal_relative_pose_with_monodepth(
                          x1 * 1000.0, x2 * 1000.0, depth1, depth2,
                          max_error=2.0, **common)))

    return steps


def warmup(problem="all", iterations=3, lo_iterations=1,
           final_refinement_iterations=1, progress=False, device="cpu"):
    # device - 'cpu' (default) warms the numba CPU kernels, 'cuda' the GPU
    #     ones, 'all' both. 'cuda' and 'all' are no-ops when no CUDA device is
    #     available, so calling with 'all' is safe on a CPU-only machine.
    steps = []
    if device in ("cpu", "all"):
        steps.extend(_warmup_steps(problem, iterations, lo_iterations,
                                   final_refinement_iterations))
    if device in ("cuda", "all"):
        steps.extend(_cuda_warmup_steps(problem, iterations, lo_iterations,
                                        final_refinement_iterations))
    bar = tqdm(steps, desc="warming up", unit="kernel", disable=not progress)
    for label, run in bar:
        # before the call, not after: the postfix names the kernel currently
        # compiling, which is where the bar sits for seconds at a time
        bar.set_postfix_str(label)
        run()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compile and cache fastpose's Numba kernels with small synthetic inputs.",
    )
    parser.add_argument(
        "--problem",
        choices=("all", "fundamental", "homography", "essential",
                 "absolute", "absolute-focal", "varying-focal",
                 "shared-focal", "monodepth"),
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
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "all"),
        default="cpu",
        help="Which backend's kernels to warm up. 'cuda' and 'all' are "
             "no-ops when no CUDA device is available. Every problem is "
             "covered on both backends.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Do not draw the progress bar.",
    )
    args = parser.parse_args(argv)

    start = time.perf_counter()
    warmup(args.problem, args.iterations, args.lo_iterations,
           args.final_refinement_iterations, progress=not args.no_progress,
           device=args.device)
    elapsed = time.perf_counter() - start
    print(f"fastpose warmup complete in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
