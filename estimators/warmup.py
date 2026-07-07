"""Command-line warmup for the cached Numba kernels."""

import argparse
import time

import numpy as np

from estimators.absolute import estimate_absolute_pose
from estimators.absolute_focal import estimate_absolute_pose_with_focal
from estimators.essential import estimate_relative_pose
from estimators.fundamental import estimate_fundamental
from estimators.varying_focal import estimate_relative_pose_with_varying_focals


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


def warmup(problem="all", iterations=3, lo_iterations=1):
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compile and cache fastpose's Numba kernels with small synthetic inputs.",
    )
    parser.add_argument(
        "--problem",
        choices=("all", "fundamental", "essential", "absolute",
                 "absolute-focal", "varying-focal"),
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
    args = parser.parse_args(argv)

    start = time.perf_counter()
    warmup(args.problem, args.iterations, args.lo_iterations)
    elapsed = time.perf_counter() - start
    print(f"fastpose warmup complete in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
