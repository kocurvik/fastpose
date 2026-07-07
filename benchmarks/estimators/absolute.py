"""Benchmark P3P absolute pose against poselib.

Default run evaluates pose mAA vs mean runtime for RANSAC iteration budgets
[100, 200, 500, 1000] and writes the tradeoff plot to abspose_maa.png;
`python -m benchmarks.estimators.absolute scaling` runs the runtime-scaling
benchmark over match counts instead.

Pose error is max(rotation error in degrees, camera-center error as a
percentage of the mean scene depth), so mAA(10) pairs 1..10 degrees with
1..10 percent position error.
"""

import sys
import time

import numpy as np

from benchmarks.utils import (generate_abspose_data, pose_maa,
                              plot_maa_tradeoff, rotation_error_deg)
from estimators.absolute import estimate_absolute_pose

MEAN_SCENE_DEPTH = 7.0  # depths are drawn uniformly from [4, 10]


def _abs_pose_error(R_est, t_est, R_gt, t_gt):
    if R_est is None:
        return 180.0
    rot = rotation_error_deg(R_est, R_gt)
    c_est = -R_est.T @ t_est
    c_gt = -R_gt.T @ t_gt
    pos = 100.0 * float(np.linalg.norm(c_est - c_gt)) / MEAN_SCENE_DEPTH
    return max(rot, pos)


def evaluate_maa(num_scenes=100, num_samples=5000, noise_sigma=2.0,
                 outlier_ratio=0.3, iterations_list=(100, 200, 500, 1000),
                 focal=1000.0, image_size=2000.0, max_error=2.0,
                 plot_path='abspose_maa.png'):
    import poselib

    c = image_size / 2.0
    camera = {'model': 'PINHOLE', 'width': int(image_size), 'height': int(image_size),
              'params': [focal, focal, c, c]}

    print(f'generating {num_scenes} scenes '
          f'({num_samples} matches, {noise_sigma} px noise, {outlier_ratio:.0%} outliers)')
    scenes = []
    for i in range(num_scenes):
        g = np.random.default_rng(40000 + i)
        x, X, R_gt, t_gt = generate_abspose_data(
            g, num_samples, noise_sigma=noise_sigma, outlier_ratio=outlier_ratio,
            focal=focal, image_size=image_size)
        scenes.append((x, (x - c) / focal, X, R_gt, t_gt))

    # warm up the JIT so compilation time is not measured
    estimate_absolute_pose(scenes[0][1][:100], scenes[0][2][:100],
                           iterations=10, max_error=max_error / focal)

    methods = ['numba', 'numba+LO', 'poselib']
    results = {m: {'runtime_ms': [], 'maa': []} for m in methods}

    for iters in iterations_list:
        errors = {m: [] for m in methods}
        times = {m: [] for m in methods}
        for si, (x, xn, X, R_gt, t_gt) in enumerate(scenes):
            for method, lo in (('numba', 0), ('numba+LO', None)):
                start = time.perf_counter()
                R_est, t_est, num_inliers, inliers = estimate_absolute_pose(
                    xn, X, iterations=iters, max_error=max_error / focal,
                    lo_iterations=lo, seed=si)
                err = _abs_pose_error(R_est, t_est, R_gt, t_gt)
                times[method].append(time.perf_counter() - start)
                errors[method].append(err)

            # poselib 3.0 expects RANSAC options nested under 'ransac'
            # (a flat dict is silently ignored)
            ransac_options = {'ransac': {'max_reproj_error': max_error,
                                         'min_iterations': iters,
                                         'max_iterations': iters}}
            start = time.perf_counter()
            # poselib 3.0 returns an Image (pose + camera)
            image, info = poselib.estimate_absolute_pose(x, X, camera,
                                                         ransac_options)
            err = _abs_pose_error(image.pose.R, image.pose.t, R_gt, t_gt)
            times['poselib'].append(time.perf_counter() - start)
            errors['poselib'].append(err)

        print(f'--- {iters} iterations ---')
        for m in methods:
            rt = float(np.mean(times[m])) * 1000.0
            maa = pose_maa(errors[m])
            results[m]['runtime_ms'].append(rt)
            results[m]['maa'].append(maa)
            print(f'{m:10s} mAA(10)={maa:.4f}  mean runtime={rt:8.2f} ms')

    plot_maa_tradeoff(results, methods, iterations_list, plot_path,
                      'Absolute pose: accuracy vs runtime \N{EM DASH} '
                      'point labels are RANSAC iterations')
    print(f'\nplot written to {plot_path}')
    return results


def run_scaling_benchmark():
    # runtime scaling over match counts at a fixed iteration budget
    import poselib

    iterations = 1000
    max_error = 2.0  # pixels
    focal = 1000.0
    image_size = 2000.0
    repeats = 5

    camera = {'model': 'PINHOLE', 'width': int(image_size), 'height': int(image_size),
              'params': [focal, focal, image_size / 2, image_size / 2]}
    # poselib 3.0 expects RANSAC options nested under 'ransac'
    ransac_options = {'ransac': {
        'max_reproj_error': max_error,
        'max_iterations': iterations,
        'min_iterations': iterations,
    }}

    rng = np.random.default_rng(0)
    x_w, X_w, _, _ = generate_abspose_data(rng, 100)
    xn_w = (x_w - image_size / 2) / focal
    estimate_absolute_pose(xn_w, X_w, iterations=10, max_error=max_error / focal)

    for num_samples in [1000, 10000, 50000]:
        rng = np.random.default_rng(0)
        x, X, R_gt, t_gt = generate_abspose_data(rng, num_samples)
        xn = (x - image_size / 2) / focal
        print(f'=== {num_samples} matches, {iterations} iterations ===')

        for label, lo in [('numba', 0), ('numba+LO', None)]:
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                R_est, t_est, num_inliers, inliers = estimate_absolute_pose(
                    xn, X, iterations=iterations, max_error=max_error / focal,
                    lo_iterations=lo)
                times.append(time.perf_counter() - start)
            rot_err = rotation_error_deg(R_est, R_gt)
            print(f'{label:11s} {min(times):.4f}s, inliers={num_inliers}/{num_samples}, '
                  f'rot err={rot_err:.3f} deg')

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            # poselib 3.0 returns an Image (pose + camera)
            image, info = poselib.estimate_absolute_pose(x, X, camera,
                                                         ransac_options)
            times.append(time.perf_counter() - start)
        rot_err = rotation_error_deg(image.pose.R, R_gt)
        print(f'{"poselib":11s} {min(times):.4f}s, inliers={info["num_inliers"]}/{num_samples}, '
              f'rot err={rot_err:.3f} deg')
        print()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'scaling':
        run_scaling_benchmark()
    else:
        evaluate_maa()
