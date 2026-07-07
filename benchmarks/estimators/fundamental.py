"""Benchmark fundamental matrix estimation against poselib.

Default run evaluates pose mAA(10) vs mean runtime for RANSAC iteration
budgets [100, 200, 500, 1000] and writes the tradeoff plot to
fundamental_maa.png; `python -m benchmarks.estimators.fundamental scaling`
runs the runtime-scaling benchmark over match counts instead.
"""

import sys
import time

import numpy as np

from benchmarks.utils import (generate_data, generate_relpose_data, pose_maa,
                              plot_maa_tradeoff, rotation_error_deg,
                              translation_error_deg)
from estimators.essential import motion_from_essential
from estimators.fundamental import estimate_fundamental, estimate_fundamental_numba
from scorers.sampson import SampsonScorer


def _pose_error_from_f(F, inliers, x1n, x2n, K, R_gt, t_gt):
    if F is None or inliers is None or np.count_nonzero(inliers) == 0:
        return 180.0
    E = K.T @ F @ K
    R_est, t_est = motion_from_essential(E, x1n[inliers], x2n[inliers])
    return max(rotation_error_deg(R_est, R_gt), translation_error_deg(t_est, t_gt))


def evaluate_maa(num_scenes=100, num_samples=20000, noise_sigma=1.0,
                 outlier_ratio=0.5, iterations_list=(100, 200, 500, 1000),
                 focal=1000.0, image_size=2000.0, max_error=2.0,
                 plot_path='fundamental_maa.png'):
    import poselib

    c = image_size / 2.0
    K = np.array([[focal, 0.0, c], [0.0, focal, c], [0.0, 0.0, 1.0]])

    print(f'generating {num_scenes} scenes '
          f'({num_samples} matches, {noise_sigma} px noise, {outlier_ratio:.0%} outliers)')
    scenes = []
    for i in range(num_scenes):
        rng = np.random.default_rng(20000 + i)
        x1, x2, R_gt, t_gt = generate_relpose_data(
            rng, num_samples, noise_sigma=noise_sigma, outlier_ratio=outlier_ratio,
            focal=focal, image_size=image_size)
        scenes.append((x1, x2, (x1 - c) / focal, (x2 - c) / focal, R_gt, t_gt))

    # warm up the JIT so compilation time is not measured
    estimate_fundamental_numba(scenes[0][0][:100], scenes[0][1][:100],
                               iterations=10, max_error=max_error)

    methods = ['numba', 'numba+LO', 'poselib']
    results = {m: {'runtime_ms': [], 'maa': []} for m in methods}

    for iters in iterations_list:
        errors = {m: [] for m in methods}
        times = {m: [] for m in methods}
        for si, (x1, x2, x1n, x2n, R_gt, t_gt) in enumerate(scenes):
            for method, lo in (('numba', 0), ('numba+LO', None)):
                start = time.perf_counter()
                F, num_inliers, inliers = estimate_fundamental_numba(
                    x1, x2, iterations=iters, max_error=max_error,
                    lo_iterations=lo, seed=si)
                times[method].append(time.perf_counter() - start)
                errors[method].append(_pose_error_from_f(F, inliers, x1n, x2n,
                                                         K, R_gt, t_gt))

            # poselib 3.0 expects RANSAC options nested under 'ransac'
            # (a flat dict is silently ignored)
            ransac_options = {'ransac': {'max_epipolar_error': max_error,
                                         'min_iterations': iters,
                                         'max_iterations': iters}}
            start = time.perf_counter()
            F, info = poselib.estimate_fundamental(x1, x2, ransac_options)
            times['poselib'].append(time.perf_counter() - start)
            _, inliers, num_inliers = SampsonScorer.score_numpy(F, x1, x2, max_error)
            errors['poselib'].append(_pose_error_from_f(F, inliers, x1n, x2n,
                                                        K, R_gt, t_gt))

        print(f'--- {iters} iterations ---')
        for m in methods:
            rt = float(np.mean(times[m])) * 1000.0
            maa = pose_maa(errors[m])
            results[m]['runtime_ms'].append(rt)
            results[m]['maa'].append(maa)
            print(f'{m:10s} mAA(10)={maa:.4f}  mean runtime={rt:8.2f} ms')

    plot_maa_tradeoff(results, methods, iterations_list, plot_path,
                      'Fundamental matrix to pose: accuracy vs runtime - '
                      'point labels are RANSAC iterations')
    print(f'\nplot written to {plot_path}')
    return results


def run_scaling_benchmark():
    # benchmark numpy / numba backends against poselib on synthetic data
    import poselib

    iterations = 1000
    max_error = 2.0
    repeats = 5

    # poselib 3.0 expects RANSAC options nested under 'ransac'
    # (a flat dict is silently ignored)
    ransac_options = {'ransac': {
        'max_epipolar_error': max_error,
        'max_iterations': iterations,
        'min_iterations': iterations,
    }}

    # warm up the JIT so compilation time is not measured
    rng = np.random.default_rng(0)
    x1_w, x2_w = generate_data(rng, 100)
    estimate_fundamental_numba(x1_w, x2_w, iterations=10)

    for num_samples in [1000, 10000, 50000]:
        rng = np.random.default_rng(0)
        x1, x2 = generate_data(rng, num_samples)
        print(f'=== {num_samples} matches, {iterations} iterations ===')

        if num_samples <= 1000:
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                F, num_inliers, inliers = estimate_fundamental(
                    x1, x2, iterations=iterations, max_error=max_error)
                times.append(time.perf_counter() - start)
            print(f'numpy:      {min(times):.4f}s, inliers={num_inliers}/{num_samples}')

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            F, num_inliers, inliers = estimate_fundamental_numba(
                x1, x2, iterations=iterations, max_error=max_error, lo_iterations=0)
            times.append(time.perf_counter() - start)
        print(f'numba:      {min(times):.4f}s, inliers={num_inliers}/{num_samples}')

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            F, num_inliers, inliers = estimate_fundamental_numba(
                x1, x2, iterations=iterations, max_error=max_error)
            times.append(time.perf_counter() - start)
        print(f'numba+LO:   {min(times):.4f}s, inliers={num_inliers}/{num_samples}')

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            F, info = poselib.estimate_fundamental(x1, x2, ransac_options)
            times.append(time.perf_counter() - start)
        print(f'poselib:    {min(times):.4f}s, inliers={info["num_inliers"]}/{num_samples}')
        print()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'scaling':
        run_scaling_benchmark()
    else:
        evaluate_maa()
