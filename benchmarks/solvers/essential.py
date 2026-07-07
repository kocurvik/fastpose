"""Accuracy and runtime of the 5-point minimal solver on synthetic
noise-free samples, compared against poselib's relpose_5pt.

For each trial a random relative pose and 5 exact calibrated correspondences
are generated. Both solvers output cheirality-checked poses; per trial the
returned pose closest to the ground truth is evaluated on three metrics:
the essential matrix error (Frobenius distance of the unit-normalized
E = [t]_x R, sign-invariant), the max algebraic epipolar residual on the
sample, and the rotation / translation-direction errors. The translation
metric is sign-sensitive so a wrong cheirality choice (wrong R candidate or
t sign) shows as a ~180 degree error.

The runtime of every solver call is measured from Python (best of `repeats`
measurements per sample, warmed up beforehand, data generation excluded);
it includes the respective call overhead (numba dispatch / pybind
conversion), i.e. what a Python RANSAC loop would actually pay. poselib's
relpose_5pt expects unit-norm bearing vectors, which are precomputed
outside the timed section.

Run with: python -m benchmarks.solvers.essential
"""

import math
import time

import numpy as np

from benchmarks.utils import (max_algebraic_residual, report_runtime,
                              report_solver_accuracy, rotation_error_deg, skew)
from estimators.utils import point_columns
from solvers.essential import FivePointSolver


def _signed_translation_error_deg(t_est, t_gt):
    # sign-sensitive direction error: a wrong cheirality sign shows as ~180
    # degrees (unlike the sign-invariant metric used for the estimators)
    c = float(np.dot(t_est, t_gt)) / (np.linalg.norm(t_est) * np.linalg.norm(t_gt))
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def _evaluate_poses(poses, x1n, x2n, E_gt, R_gt, t_gt,
                    errors, residuals, rot_errors, trans_errors):
    # record, over the returned poses, the best essential matrix error (with
    # the algebraic residual of that model) and the best pose error (rotation
    # and sign-sensitive translation of the pose minimizing their max)
    gt = E_gt.ravel()
    best_e = np.inf
    best_e_model = None
    best_pose = np.inf
    best_rot = 180.0
    best_trans = 180.0
    for R, t in poses:
        e = (skew(t) @ R).ravel()
        e = e / np.linalg.norm(e)
        e_err = min(np.linalg.norm(e - gt), np.linalg.norm(e + gt))
        if e_err < best_e:
            best_e = e_err
            best_e_model = e
        rot = rotation_error_deg(R, R_gt)
        trans = _signed_translation_error_deg(t, t_gt)
        if max(rot, trans) < best_pose:
            best_pose = max(rot, trans)
            best_rot = rot
            best_trans = trans
    errors.append(best_e)
    residuals.append(max_algebraic_residual(best_e_model, x1n, x2n))
    rot_errors.append(best_rot)
    trans_errors.append(best_trans)


def _report_pose_errors(rot_errors, trans_errors):
    for label, vals in (('rotation error (deg)         ', rot_errors),
                        ('translation dir. error (deg) ', trans_errors)):
        med, p90, p99, mx = np.percentile(vals, [50, 90, 99, 100])
        print(f'{label} median={med:.2e}  p90={p90:.2e}  p99={p99:.2e}  max={mx:.2e}')


def run(num_trials=1000, seed=0, repeats=5):
    import poselib

    from benchmarks.utils import generate_exact_essential_sample

    repeats = max(1, repeats)
    solver = FivePointSolver()
    sample = np.arange(solver.sample_size, dtype=np.int64)
    models = np.empty((solver.max_models, solver.num_params))
    workspace = np.empty(solver.workspace_size)

    trials = []
    for trial in range(num_trials):
        rng = np.random.default_rng(seed + trial)
        x1n, x2n, E_gt, R_gt, t_gt = generate_exact_essential_sample(rng)
        # unit-norm bearing vectors for poselib (precomputed, not timed)
        b1 = np.column_stack([x1n, np.ones(len(x1n))])
        b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
        b2 = np.column_stack([x2n, np.ones(len(x2n))])
        b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
        trials.append((point_columns(x1n, x2n), x1n, x2n, b1, b2,
                       E_gt, R_gt, t_gt))

    print(f'=== 5-point relative pose solver: {num_trials} noise-free minimal samples ===')

    # ------------------------------------------------------------------
    # fastpose numba solver
    # ------------------------------------------------------------------
    solver.solve(trials[0][0], sample, models, workspace)  # warm up the JIT

    errors, residuals, rot_errors, trans_errors, times = [], [], [], [], []
    num_failures = 0
    for data, x1n, x2n, _, _, E_gt, R_gt, t_gt in trials:
        best_dt = np.inf
        for _ in range(repeats):
            start = time.perf_counter()
            num_models = solver.solve(data, sample, models, workspace)
            best_dt = min(best_dt, time.perf_counter() - start)
        times.append(best_dt)
        if num_models == 0:
            num_failures += 1
            continue
        poses = [(models[m, :9].reshape(3, 3).copy(), models[m, 9:12].copy())
                 for m in range(num_models)]
        _evaluate_poses(poses, x1n, x2n, E_gt, R_gt, t_gt,
                        errors, residuals, rot_errors, trans_errors)

    report_solver_accuracy('fastpose 5-point solver (numba)', errors, residuals,
                           num_failures, num_trials)
    _report_pose_errors(rot_errors, trans_errors)
    report_runtime(f'runtime per solve call (best of {repeats}):', times)
    times_fastpose = times

    # ------------------------------------------------------------------
    # poselib relpose_5pt
    # ------------------------------------------------------------------
    poselib.relpose_5pt(trials[0][3], trials[0][4])  # warm up

    errors, residuals, rot_errors, trans_errors, times = [], [], [], [], []
    num_failures = 0
    for _, x1n, x2n, b1, b2, E_gt, R_gt, t_gt in trials:
        best_dt = np.inf
        for _ in range(repeats):
            start = time.perf_counter()
            pl_poses = poselib.relpose_5pt(b1, b2)
            best_dt = min(best_dt, time.perf_counter() - start)
        times.append(best_dt)
        if len(pl_poses) == 0:
            num_failures += 1
            continue
        poses = [(p.R, p.t) for p in pl_poses]
        _evaluate_poses(poses, x1n, x2n, E_gt, R_gt, t_gt,
                        errors, residuals, rot_errors, trans_errors)

    print()
    report_solver_accuracy('poselib relpose_5pt', errors, residuals,
                           num_failures, num_trials)
    _report_pose_errors(rot_errors, trans_errors)
    report_runtime(f'runtime per solve call (best of {repeats}):', times)

    med_fastpose = float(np.median(times_fastpose)) * 1e6
    med_poselib = float(np.median(times)) * 1e6
    print(f'\nmedian runtime per solve call: fastpose {med_fastpose:.1f} us, '
          f'poselib {med_poselib:.1f} us '
          f'(ratio fastpose/poselib: {med_fastpose / med_poselib:.2f})')


if __name__ == '__main__':
    run()
