"""Benchmark homography estimation against poselib.

Default run evaluates transfer-error mAA(10) vs mean runtime for RANSAC
iteration budgets [100, 200, 500, 1000] and writes the tradeoff plot to
homography_maa.png; `python -m benchmarks.estimators.homography scaling` runs
the runtime-scaling benchmark over match counts instead.

The accuracy metric is the *median symmetric transfer error in pixels* of the
estimated H over the ground-truth inlier correspondences, scored with mAA over
1..10 px the same way the pose benchmarks score degrees. There is no pose to
compare - `motion_from_homography` would need the calibration and would add
its own four-fold ambiguity - and the transfer error is what a homography is
actually used for.

Both libraries are run at the same `max_error`, and it means the same thing
in both: poselib's homography estimator thresholds the forward transfer
distance, and fastpose thresholds the *average* of the forward and backward
ones, which coincides with it wherever the two directions agree. Measured over
12 scenes at 2000 matches, the two inlier sets agree on 98.7-100% of points
across thresholds from 1 to 6 px.
"""

import sys
import time

import numpy as np

from benchmarks.utils import (generate_homography_data, plot_maa_tradeoff,
                              pose_maa)
from fastpose.estimators.homography import estimate_homography
from fastpose.scorers.transfer import symmetric_transfer_errors_numpy

# a failed estimate is charged an error well past the largest mAA threshold
FAILURE_ERROR = 1e3


def _transfer_error(H, x1, x2, inlier_mask):
    # median symmetric transfer error, in pixels, over the true inliers
    if H is None or not np.all(np.isfinite(H)):
        return FAILURE_ERROR
    errors = symmetric_transfer_errors_numpy(H, x1[inlier_mask],
                                             x2[inlier_mask])
    if not np.any(np.isfinite(errors)):
        return FAILURE_ERROR
    return float(np.sqrt(np.median(errors)))


def evaluate_maa(num_scenes=100, num_samples=5000, noise_sigma=2.0,
                 outlier_ratio=0.4, iterations_list=(100, 200, 500, 1000),
                 focal=1000.0, image_size=2000.0, max_error=4.0,
                 plot_path='homography_maa.png'):
    import poselib

    print(f'generating {num_scenes} planar scenes '
          f'({num_samples} matches, {noise_sigma} px noise, '
          f'{outlier_ratio:.0%} outliers)')
    scenes = []
    for i in range(num_scenes):
        rng = np.random.default_rng(40000 + i)
        x1, x2, H_gt, mask = generate_homography_data(
            rng, num_samples, noise_sigma=noise_sigma,
            outlier_ratio=outlier_ratio, focal=focal, image_size=image_size,
            return_gt=True)
        scenes.append((x1, x2, H_gt, mask))

    # warm up the JIT so compilation time is not measured
    estimate_homography(scenes[0][0][:100], scenes[0][1][:100], iterations=10,
                        max_error=max_error)

    methods = ['numba', 'numba+LO', 'poselib']
    results = {m: {'runtime_ms': [], 'maa': []} for m in methods}

    for iters in iterations_list:
        errors = {m: [] for m in methods}
        times = {m: [] for m in methods}
        for si, (x1, x2, H_gt, mask) in enumerate(scenes):
            for method, lo in (('numba', 0), ('numba+LO', None)):
                start = time.perf_counter()
                model, est_info = estimate_homography(
                    x1, x2, iterations=iters, max_error=max_error,
                    lo_iterations=lo, seed=si)
                times[method].append(time.perf_counter() - start)
                if est_info['num_inliers'] == 0:
                    errors[method].append(FAILURE_ERROR)
                else:
                    errors[method].append(
                        _transfer_error(model['H'], x1, x2, mask))

            # poselib 3.0 takes the iteration budget nested under 'ransac' but
            # the inlier threshold as a top-level 'max_error'; see the note in
            # benchmarks/estimators/fundamental.py
            ransac_options = {'max_error': max_error,
                              'ransac': {'min_iterations': iters,
                                         'max_iterations': iters}}
            start = time.perf_counter()
            H, info = poselib.estimate_homography(x1, x2, ransac_options)
            times['poselib'].append(time.perf_counter() - start)
            errors['poselib'].append(_transfer_error(H, x1, x2, mask))

        print(f'--- {iters} iterations ---')
        for m in methods:
            rt = float(np.mean(times[m])) * 1000.0
            maa = pose_maa(errors[m])
            results[m]['runtime_ms'].append(rt)
            results[m]['maa'].append(maa)
            print(f'{m:10s} mAA(10px)={maa:.4f}  mean runtime={rt:8.2f} ms')

    plot_maa_tradeoff(results, methods, iterations_list, plot_path,
                      'Homography: median transfer error vs runtime - '
                      'point labels are RANSAC iterations')
    print(f'\nplot written to {plot_path}')
    return results


def run_scaling_benchmark():
    import poselib

    iterations = 1000
    max_error = 4.0
    repeats = 5

    ransac_options = {
        'max_error': max_error,
        'ransac': {
            'max_iterations': iterations,
            'min_iterations': iterations,
        },
    }

    # warm up the JIT so compilation time is not measured
    rng = np.random.default_rng(0)
    x1_w, x2_w = generate_homography_data(rng, 100)
    estimate_homography(x1_w, x2_w, iterations=10)

    for num_samples in [1000, 10000, 50000]:
        rng = np.random.default_rng(0)
        x1, x2 = generate_homography_data(rng, num_samples)
        print(f'=== {num_samples} matches, {iterations} iterations ===')

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model, est_info = estimate_homography(
                x1, x2, iterations=iterations, max_error=max_error,
                lo_iterations=0)
            times.append(time.perf_counter() - start)
        print(f'numba:      {min(times):.4f}s, '
              f'inliers={est_info["num_inliers"]}/{num_samples}')

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model, est_info = estimate_homography(
                x1, x2, iterations=iterations, max_error=max_error)
            times.append(time.perf_counter() - start)
        print(f'numba+LO:   {min(times):.4f}s, '
              f'inliers={est_info["num_inliers"]}/{num_samples}')

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            H, info = poselib.estimate_homography(x1, x2, ransac_options)
            times.append(time.perf_counter() - start)
        print(f'poselib:    {min(times):.4f}s, '
              f'inliers={info["num_inliers"]}/{num_samples}')
        print()


def run_cuda_scaling_benchmark():
    from fastpose import cuda as cuda_backend
    if not cuda_backend.is_available():
        print('no usable CUDA device: '
              f'{cuda_backend.unavailable_reason()}')
        return

    max_error = 4.0
    repeats = 3

    rng = np.random.default_rng(0)
    x1_w, x2_w = generate_homography_data(rng, 200)
    for device in ('cpu', 'cuda'):
        estimate_homography(x1_w, x2_w, iterations=10, device=device)

    for num_samples in [2000, 16000]:
        for iterations in [1000, 10000]:
            rng = np.random.default_rng(0)
            x1, x2 = generate_homography_data(rng, num_samples)
            print(f'=== {num_samples} matches, {iterations} iterations ===')
            for device in ('cpu', 'cuda'):
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    _, est_info = estimate_homography(
                        x1, x2, iterations=iterations,
                        min_iterations=iterations, max_error=max_error,
                        seed=0, device=device)
                    times.append(time.perf_counter() - start)
                print(f'{device:6s}: {min(times):.4f}s, '
                      f'inliers={est_info["num_inliers"]}/{num_samples}')
            print()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'scaling':
        run_scaling_benchmark()
    elif len(sys.argv) > 1 and sys.argv[1] == 'cuda-scaling':
        run_cuda_scaling_benchmark()
    else:
        evaluate_maa()
