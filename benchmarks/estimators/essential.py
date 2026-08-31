"""Benchmark 5-point relative pose (essential matrix) against poselib.

Default run evaluates pose mAA(10) vs mean runtime for RANSAC iteration
budgets [100, 200, 500, 1000] and writes the tradeoff plot to
relpose_maa.png; `python -m benchmarks.estimators.essential scaling` runs
the runtime-scaling benchmark over match counts instead.
"""

import sys
import time

import numpy as np

from benchmarks.utils import (generate_relpose_data, pose_maa,
                              plot_maa_tradeoff, rotation_error_deg,
                              translation_error_deg)
from fastpose.estimators.essential import estimate_relative_pose


def _pose_error(model, est_info, R_gt, t_gt):
    if est_info['num_inliers'] == 0:
        return 180.0
    return max(rotation_error_deg(model['R'], R_gt),
               translation_error_deg(model['t'], t_gt))


def evaluate_maa(num_scenes=100, num_samples=5000, noise_sigma=4.0,
                 outlier_ratio=0.2, iterations_list=(100, 200, 500, 1000),
                 focal=1000.0, image_size=2000.0, max_error=2.0,
                 plot_path='relpose_maa.png'):
    import poselib

    c = image_size / 2.0
    camera = {'model': 'PINHOLE', 'width': int(image_size), 'height': int(image_size),
              'params': [focal, focal, c, c]}

    print(f'generating {num_scenes} scenes '
          f'({num_samples} matches, {noise_sigma} px noise, {outlier_ratio:.0%} outliers)')
    scenes = []
    for i in range(num_scenes):
        g = np.random.default_rng(10000 + i)
        x1, x2, R_gt, t_gt = generate_relpose_data(
            g, num_samples, noise_sigma=noise_sigma, outlier_ratio=outlier_ratio,
            focal=focal, image_size=image_size)
        scenes.append((x1, x2, (x1 - c) / focal, (x2 - c) / focal, R_gt, t_gt))

    # warm up the JIT so compilation time is not measured
    estimate_relative_pose(scenes[0][2][:100], scenes[0][3][:100],
                           iterations=10, max_error=max_error / focal)

    methods = ['numba', 'numba+LO', 'poselib']
    results = {m: {'runtime_ms': [], 'maa': []} for m in methods}

    for iters in iterations_list:
        errors = {m: [] for m in methods}
        times = {m: [] for m in methods}
        for si, (x1, x2, x1n, x2n, R_gt, t_gt) in enumerate(scenes):
            for method, lo in (('numba', 0), ('numba+LO', None)):
                start = time.perf_counter()
                model, est_info = estimate_relative_pose(
                    x1n, x2n, iterations=iters, max_error=max_error / focal,
                    lo_iterations=lo, seed=si)
                err = _pose_error(model, est_info, R_gt, t_gt)
                times[method].append(time.perf_counter() - start)
                errors[method].append(err)

            # poselib 3.0 takes the iteration budget nested under 'ransac' (a
            # flat dict is silently ignored) but the inlier threshold as a
            # top-level 'max_error'; 'max_epipolar_error' is silently ignored
            # in both places, leaving poselib on its default threshold.
            ransac_options = {'max_error': max_error,
                              'ransac': {'min_iterations': iters,
                                         'max_iterations': iters}}
            start = time.perf_counter()
            pose, info = poselib.estimate_relative_pose(x1, x2, camera, camera,
                                                        ransac_options)
            err = max(rotation_error_deg(pose.R, R_gt),
                      translation_error_deg(pose.t, t_gt))
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
                      'Relative pose: accuracy vs runtime \N{EM DASH} '
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
    # poselib 3.0 takes the iteration budget nested under 'ransac' but the
    # inlier threshold as a top-level 'max_error'; 'max_epipolar_error' is
    # silently ignored in both places.
    ransac_options = {
        'max_error': max_error,
        'ransac': {
            'max_iterations': iterations,
            'min_iterations': iterations,
        },
    }

    rng = np.random.default_rng(0)
    x1_w, x2_w, _, _ = generate_relpose_data(rng, 100)
    x1n_w = (x1_w - image_size / 2) / focal
    x2n_w = (x2_w - image_size / 2) / focal
    estimate_relative_pose(x1n_w, x2n_w, iterations=10, max_error=max_error / focal)

    for num_samples in [1000, 10000, 50000]:
        rng = np.random.default_rng(0)
        x1, x2, R_gt, t_gt = generate_relpose_data(rng, num_samples)
        x1n = (x1 - image_size / 2) / focal
        x2n = (x2 - image_size / 2) / focal
        print(f'=== {num_samples} matches, {iterations} iterations ===')

        for label, lo in [('numba', 0), ('numba+LO', None)]:
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                model, est_info = estimate_relative_pose(
                    x1n, x2n, iterations=iterations, max_error=max_error / focal,
                    lo_iterations=lo)
                times.append(time.perf_counter() - start)
            rot_err = rotation_error_deg(model['R'], R_gt)
            print(f'{label:11s} {min(times):.4f}s, '
                  f'inliers={est_info["num_inliers"]}/{num_samples}, '
                  f'rot err={rot_err:.3f} deg')

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            pose, info = poselib.estimate_relative_pose(x1, x2, camera, camera,
                                                        ransac_options)
            times.append(time.perf_counter() - start)
        rot_err = rotation_error_deg(pose.R, R_gt)
        print(f'{"poselib":11s} {min(times):.4f}s, inliers={info["num_inliers"]}/{num_samples}, '
              f'rot err={rot_err:.3f} deg')
        print()


def run_cuda_scaling_benchmark():
    # CPU vs GPU over both axes the GPU driver is supposed to win on: match
    # count and iteration budget. Both are swept because they trade off
    # differently - match count makes each round more expensive, iteration
    # count adds rounds, and only the second amortizes the fixed per-round
    # launch and readback cost.
    from fastpose import cuda as cuda_backend
    if not cuda_backend.is_available():
        print(f'no usable CUDA device: {cuda_backend.unavailable_reason()}')
        return

    max_error = 2.0  # pixels
    focal = 1000.0
    image_size = 2000.0
    repeats = 3

    # Warm both backends before timing anything. On the GPU this compiles the
    # batched solver, the scorer and both LM kernels (one per loss); a single
    # untimed call is not enough, which is easy to mistake for a slow GPU.
    rng = np.random.default_rng(0)
    x1_w, x2_w, _, _ = generate_relpose_data(rng, 500)
    x1n_w = (x1_w - image_size / 2) / focal
    x2n_w = (x2_w - image_size / 2) / focal
    for device in ('cpu', 'cuda'):
        for _ in range(2):
            estimate_relative_pose(x1n_w, x2n_w, iterations=50,
                                   min_iterations=50,
                                   max_error=max_error / focal, seed=0,
                                   device=device)

    header = (f'{"matches":>8} {"iters":>7} | {"cpu":>9} {"cuda":>9} '
              f'{"speedup":>8} | {"inl cpu":>8} {"inl gpu":>8} | '
              f'{"err cpu":>8} {"err gpu":>8}')
    print(header)
    print('-' * len(header))

    for num_samples in [2000, 16000, 50000]:
        for iterations in [1000, 5000, 20000]:
            rng = np.random.default_rng(0)
            x1, x2, R_gt, t_gt = generate_relpose_data(
                rng, num_samples, noise_sigma=0.5, outlier_ratio=0.3,
                focal=focal, image_size=image_size)
            x1n = (x1 - image_size / 2) / focal
            x2n = (x2 - image_size / 2) / focal

            results = {}
            for device in ('cpu', 'cuda'):
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    model, est_info = estimate_relative_pose(
                        x1n, x2n, iterations=iterations,
                        min_iterations=iterations,
                        max_error=max_error / focal, seed=0, device=device)
                    times.append(time.perf_counter() - start)
                err = max(rotation_error_deg(model['R'], R_gt),
                          translation_error_deg(model['t'], t_gt))
                results[device] = (min(times), est_info['num_inliers'], err)

            cpu_t, cpu_inl, cpu_err = results['cpu']
            gpu_t, gpu_inl, gpu_err = results['cuda']
            print(f'{num_samples:>8} {iterations:>7} | {cpu_t * 1e3:8.1f}ms '
                  f'{gpu_t * 1e3:8.1f}ms {cpu_t / gpu_t:7.2f}x | '
                  f'{cpu_inl:>8} {gpu_inl:>8} | {cpu_err:8.4f} {gpu_err:8.4f}')
        print()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'scaling':
        run_scaling_benchmark()
    elif len(sys.argv) > 1 and sys.argv[1] == 'cuda-scaling':
        run_cuda_scaling_benchmark()
    else:
        evaluate_maa()
