"""Benchmark monodepth relative pose estimation against poselib.

Default run evaluates the calibrated estimators (P3P variant, 3-point shift
variant on affine depths) as pose mAA(10) vs mean runtime for RANSAC
iteration budgets [100, 200, 500, 1000] against poselib's
estimate_monodepth_relative_pose, and writes monodepth_maa.png.

`python -m benchmarks.estimators.monodepth shared-focal` and
`python -m benchmarks.estimators.monodepth varying-focal` evaluate the
unknown-focal variants (adding the focal error to the report);
`python -m benchmarks.estimators.monodepth scaling` runs the
runtime-scaling benchmark over match counts for all four variants.

Thresholds follow the poselib defaults used in the estimators: 2 px Sampson
scoring error and 16 px reprojection error for the hybrid local
optimization (scaled by 1/f for the calibrated problems).
"""

import sys
import time

import numpy as np

from benchmarks.utils import (generate_monodepth_relpose_data, pose_maa,
                              plot_maa_tradeoff, rotation_error_deg,
                              translation_error_deg)
from estimators.monodepth import (
    estimate_relative_pose_with_monodepth,
    estimate_shared_focal_relative_pose_with_monodepth,
    estimate_varying_focal_relative_pose_with_monodepth)

FOCAL = 1000.0
FOCAL1 = 800.0
FOCAL2 = 1300.0
ALPHA1 = 0.7
ALPHA2 = 1.4
NOISE_SIGMA = 2.0     # px
DEPTH_NOISE = 0.02    # relative
OUTLIER_RATIO = 0.3
MAX_ERROR = 2.0       # px, Sampson
MAX_REPROJ_ERROR = 16.0  # px


def _pose_error(R_est, t_est, R_gt, t_gt):
    if R_est is None:
        return 180.0
    return max(rotation_error_deg(R_est, R_gt),
               translation_error_deg(t_est, t_gt))


def _generate_scenes(num_scenes, num_samples, mode, with_shift=False):
    scenes = []
    beta1 = 0.3 * ALPHA1 if with_shift else 0.0
    beta2 = -0.2 * ALPHA2 if with_shift else 0.0
    if mode == 'calibrated':
        f1 = f2 = 1.0
        noise = NOISE_SIGMA / FOCAL
    elif mode == 'shared-focal':
        f1 = f2 = FOCAL
        noise = NOISE_SIGMA
    else:
        f1, f2 = FOCAL1, FOCAL2
        noise = NOISE_SIGMA
    for i in range(num_scenes):
        rng = np.random.default_rng(60000 + i)
        scenes.append(generate_monodepth_relpose_data(
            rng, num_samples, noise_sigma=noise, outlier_ratio=OUTLIER_RATIO,
            depth_noise=DEPTH_NOISE, focal1=f1, focal2=f2,
            alpha1=ALPHA1, beta1=beta1, alpha2=ALPHA2, beta2=beta2))
    return scenes


def _poselib_ransac_opt(iters, max_errors, estimate_shift=False):
    return {'ransac': {'min_iterations': iters, 'max_iterations': iters},
            'max_errors': max_errors, 'estimate_shift': estimate_shift}


def evaluate_maa(num_scenes=100, num_samples=5000,
                 iterations_list=(100, 200, 500, 1000),
                 plot_path='monodepth_maa.png'):
    import poselib

    print(f'generating {num_scenes} calibrated monodepth scenes '
          f'({num_samples} matches, {NOISE_SIGMA} px noise, '
          f'{100 * DEPTH_NOISE:.0f}% depth noise, {OUTLIER_RATIO:.0%} outliers)')
    scenes = _generate_scenes(num_scenes, num_samples, 'calibrated')
    shift_scenes = _generate_scenes(num_scenes, num_samples, 'calibrated',
                                    with_shift=True)

    max_error = MAX_ERROR / FOCAL
    max_reproj = MAX_REPROJ_ERROR / FOCAL
    cam = {'model': 'SIMPLE_PINHOLE', 'width': -1, 'height': -1,
           'params': [1.0, 0.0, 0.0]}

    # warm up the JIT so compilation time is not measured
    estimate_relative_pose_with_monodepth(
        scenes[0][0][:100], scenes[0][1][:100], scenes[0][2][:100],
        scenes[0][3][:100], iterations=10, max_error=max_error,
        max_reproj_error=max_reproj)
    estimate_relative_pose_with_monodepth(
        shift_scenes[0][0][:100], shift_scenes[0][1][:100],
        shift_scenes[0][2][:100], shift_scenes[0][3][:100],
        estimate_shift=True, iterations=10, max_error=max_error,
        max_reproj_error=max_reproj)

    methods = ['md-p3p', 'md-p3p+LO', 'md-shift+LO', 'poselib-md',
               'poselib-md-shift']
    results = {m: {'runtime_ms': [], 'maa': []} for m in methods}

    for iters in iterations_list:
        errors = {m: [] for m in methods}
        times = {m: [] for m in methods}
        scale_errs = []
        for si in range(num_scenes):
            x1, x2, d1, d2, R_gt, t_gt = scenes[si]
            for method, lo in (('md-p3p', 0), ('md-p3p+LO', None)):
                start = time.perf_counter()
                R_est, t_est, scale, _, _, num_inliers, _ = (
                    estimate_relative_pose_with_monodepth(
                        x1, x2, d1, d2, iterations=iters,
                        max_error=max_error, max_reproj_error=max_reproj,
                        lo_iterations=lo, seed=si))
                times[method].append(time.perf_counter() - start)
                errors[method].append(_pose_error(R_est, t_est, R_gt, t_gt))
                if method == 'md-p3p+LO' and scale is not None:
                    scale_errs.append(abs(scale - ALPHA1 / ALPHA2) * ALPHA2 / ALPHA1)

            opt = _poselib_ransac_opt(iters, [max_reproj, max_error])
            start = time.perf_counter()
            geom, info = poselib.estimate_monodepth_relative_pose(
                x1, x2, d1, d2, cam, cam, opt)
            times['poselib-md'].append(time.perf_counter() - start)
            errors['poselib-md'].append(
                _pose_error(geom.pose.R, geom.pose.t, R_gt, t_gt))

            # shift variants on the affine-depth scenes
            x1, x2, d1, d2, R_gt, t_gt = shift_scenes[si]
            start = time.perf_counter()
            R_est, t_est, scale, _, _, num_inliers, _ = (
                estimate_relative_pose_with_monodepth(
                    x1, x2, d1, d2, estimate_shift=True, iterations=iters,
                    max_error=max_error, max_reproj_error=max_reproj,
                    seed=si))
            times['md-shift+LO'].append(time.perf_counter() - start)
            errors['md-shift+LO'].append(_pose_error(R_est, t_est, R_gt, t_gt))

            opt = _poselib_ransac_opt(iters, [max_reproj, max_error],
                                      estimate_shift=True)
            start = time.perf_counter()
            geom, info = poselib.estimate_monodepth_relative_pose(
                x1, x2, d1, d2, cam, cam, opt)
            times['poselib-md-shift'].append(time.perf_counter() - start)
            errors['poselib-md-shift'].append(
                _pose_error(geom.pose.R, geom.pose.t, R_gt, t_gt))

        print(f'--- {iters} iterations ---')
        for m in methods:
            rt = float(np.mean(times[m])) * 1000.0
            maa = pose_maa(errors[m])
            results[m]['runtime_ms'].append(rt)
            results[m]['maa'].append(maa)
            print(f'{m:17s} mAA(10)={maa:.4f}  mean runtime={rt:8.2f} ms')
        if scale_errs:
            print(f'md-p3p+LO median scale error: {np.median(scale_errs):.3%}')

    plot_maa_tradeoff(results, methods, iterations_list, plot_path,
                      'Calibrated monodepth relative pose: accuracy vs '
                      'runtime \N{EM DASH} point labels are RANSAC iterations')
    print(f'\nplot written to {plot_path}')
    return results


def evaluate_focal_maa(shared, num_scenes=100, num_samples=5000,
                       iterations_list=(100, 200, 500, 1000)):
    import poselib

    mode = 'shared-focal' if shared else 'varying-focal'
    plot_path = f'monodepth_{mode.replace("-", "_")}_maa.png'
    print(f'generating {num_scenes} {mode} monodepth scenes '
          f'({num_samples} matches, {NOISE_SIGMA} px noise, '
          f'{100 * DEPTH_NOISE:.0f}% depth noise, {OUTLIER_RATIO:.0%} outliers)')
    scenes = _generate_scenes(num_scenes, num_samples, mode)

    if shared:
        estimate = estimate_shared_focal_relative_pose_with_monodepth
        poselib_estimate = poselib.estimate_monodepth_shared_focal_relative_pose
        poselib_args = (np.zeros(2),)
        gt_focals = (FOCAL, FOCAL)
    else:
        estimate = estimate_varying_focal_relative_pose_with_monodepth
        poselib_estimate = poselib.estimate_monodepth_varying_focal_relative_pose
        poselib_args = (np.zeros(2), np.zeros(2))
        gt_focals = (FOCAL1, FOCAL2)

    estimate(scenes[0][0][:100], scenes[0][1][:100], scenes[0][2][:100],
             scenes[0][3][:100], iterations=10, max_error=MAX_ERROR,
             max_reproj_error=MAX_REPROJ_ERROR)

    short = 'md-sf' if shared else 'md-vf'
    methods = [short, short + '+LO', 'poselib']
    results = {m: {'runtime_ms': [], 'maa': []} for m in methods}

    for iters in iterations_list:
        errors = {m: [] for m in methods}
        times = {m: [] for m in methods}
        focal_errs = []
        poselib_focal_errs = []
        for si in range(num_scenes):
            x1, x2, d1, d2, R_gt, t_gt = scenes[si]
            for method, lo in ((short, 0), (short + '+LO', None)):
                start = time.perf_counter()
                out = estimate(x1, x2, d1, d2, iterations=iters,
                               max_error=MAX_ERROR,
                               max_reproj_error=MAX_REPROJ_ERROR,
                               lo_iterations=lo, seed=si)
                times[method].append(time.perf_counter() - start)
                R_est, t_est = out[0], out[1]
                errors[method].append(_pose_error(R_est, t_est, R_gt, t_gt))
                if method.endswith('+LO') and R_est is not None:
                    if shared:
                        focal_errs.append(abs(out[2] - FOCAL) / FOCAL)
                    else:
                        focal_errs.append(max(
                            abs(out[2] - FOCAL1) / FOCAL1,
                            abs(out[3] - FOCAL2) / FOCAL2))

            opt = _poselib_ransac_opt(iters, [MAX_REPROJ_ERROR, MAX_ERROR])
            start = time.perf_counter()
            pair, info = poselib_estimate(x1, x2, d1, d2, *poselib_args, opt)
            times['poselib'].append(time.perf_counter() - start)
            pose = pair.geometry.pose
            errors['poselib'].append(_pose_error(pose.R, pose.t, R_gt, t_gt))
            poselib_focal_errs.append(max(
                abs(pair.camera1.focal() - gt_focals[0]) / gt_focals[0],
                abs(pair.camera2.focal() - gt_focals[1]) / gt_focals[1]))

        print(f'--- {iters} iterations ---')
        for m in methods:
            rt = float(np.mean(times[m])) * 1000.0
            maa = pose_maa(errors[m])
            results[m]['runtime_ms'].append(rt)
            results[m]['maa'].append(maa)
            print(f'{m:10s} mAA(10)={maa:.4f}  mean runtime={rt:8.2f} ms')
        if focal_errs:
            print(f'{short}+LO median focal error: {np.median(focal_errs):.3%}, '
                  f'poselib: {np.median(poselib_focal_errs):.3%}')

    plot_maa_tradeoff(results, methods, iterations_list, plot_path,
                      f'{mode} monodepth relative pose: accuracy vs runtime '
                      '\N{EM DASH} point labels are RANSAC iterations')
    print(f'\nplot written to {plot_path}')
    return results


def run_scaling_benchmark():
    # runtime scaling over match counts at a fixed iteration budget
    iterations = 1000
    repeats = 5

    variants = [
        ('md-p3p', 'calibrated', False,
         lambda x1, x2, d1, d2, iters, seed: estimate_relative_pose_with_monodepth(
             x1, x2, d1, d2, iterations=iters, max_error=MAX_ERROR / FOCAL,
             max_reproj_error=MAX_REPROJ_ERROR / FOCAL, seed=seed)),
        ('md-shift', 'calibrated', True,
         lambda x1, x2, d1, d2, iters, seed: estimate_relative_pose_with_monodepth(
             x1, x2, d1, d2, estimate_shift=True, iterations=iters,
             max_error=MAX_ERROR / FOCAL,
             max_reproj_error=MAX_REPROJ_ERROR / FOCAL, seed=seed)),
        ('md-shared-f', 'shared-focal', False,
         lambda x1, x2, d1, d2, iters, seed:
         estimate_shared_focal_relative_pose_with_monodepth(
             x1, x2, d1, d2, iterations=iters, max_error=MAX_ERROR,
             max_reproj_error=MAX_REPROJ_ERROR, seed=seed)),
        ('md-varying-f', 'varying-focal', False,
         lambda x1, x2, d1, d2, iters, seed:
         estimate_varying_focal_relative_pose_with_monodepth(
             x1, x2, d1, d2, iterations=iters, max_error=MAX_ERROR,
             max_reproj_error=MAX_REPROJ_ERROR, seed=seed)),
    ]

    for label, mode, with_shift, run in variants:
        scenes = _generate_scenes(1, 100, mode, with_shift=with_shift)
        run(scenes[0][0], scenes[0][1], scenes[0][2], scenes[0][3], 10, 0)

    for num_samples in [1000, 10000, 50000]:
        print(f'=== {num_samples} matches, {iterations} iterations ===')
        for label, mode, with_shift, run in variants:
            x1, x2, d1, d2, R_gt, t_gt = _generate_scenes(
                1, num_samples, mode, with_shift=with_shift)[0]
            times = []
            out = None
            for _ in range(repeats):
                start = time.perf_counter()
                out = run(x1, x2, d1, d2, iterations, 0)
                times.append(time.perf_counter() - start)
            R_est, t_est = out[0], out[1]
            num_inliers = out[-2]
            err = _pose_error(R_est, t_est, R_gt, t_gt)
            print(f'{label:13s} {min(times):.4f}s, '
                  f'inliers={num_inliers}/{num_samples}, '
                  f'pose err={err:.3f} deg')
        print()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'scaling':
        run_scaling_benchmark()
    elif len(sys.argv) > 1 and sys.argv[1] == 'shared-focal':
        evaluate_focal_maa(shared=True)
    elif len(sys.argv) > 1 and sys.argv[1] == 'varying-focal':
        evaluate_focal_maa(shared=False)
    else:
        evaluate_maa()
