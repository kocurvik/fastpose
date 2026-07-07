"""Accuracy and runtime of the 7-point minimal solver on synthetic
noise-free samples.

For each trial a random rank-2 fundamental matrix and 7 exact
correspondences are generated; the solver kernel is called directly and the
closest returned model is compared against the ground truth (Frobenius
distance of the unit-normalized matrices, sign-invariant). The runtime of
every solver call is measured from Python (best of `repeats` measurements
per sample, JIT warmed up beforehand, data generation excluded).

Run with: python -m benchmarks.solvers.fundamental
"""

import time

import numpy as np

from benchmarks.utils import (best_model_error, generate_exact_fundamental_sample,
                              max_algebraic_residual, report_runtime,
                              report_solver_accuracy)
from estimators.utils import point_columns
from solvers.fundamental import SevenPointSolver


def run(num_trials=1000, seed=0, repeats=5):
    repeats = max(1, repeats)
    solver = SevenPointSolver()
    sample = np.arange(solver.sample_size, dtype=np.int64)
    models = np.empty((solver.max_models, solver.num_params))
    workspace = np.empty(solver.workspace_size)

    trials = []
    for trial in range(num_trials):
        rng = np.random.default_rng(seed + trial)
        x1, x2, F_gt = generate_exact_fundamental_sample(rng)
        trials.append((point_columns(x1, x2), x1, x2, F_gt))

    # warm up the JIT so compilation time is not measured
    solver.solve(trials[0][0], sample, models, workspace)

    print(f'=== 7-point fundamental solver: {num_trials} noise-free minimal samples ===')
    errors = []
    residuals = []
    times = []
    num_failures = 0
    for data, x1, x2, F_gt in trials:
        best_dt = np.inf
        for _ in range(repeats):
            start = time.perf_counter()
            num_models = solver.solve(data, sample, models, workspace)
            best_dt = min(best_dt, time.perf_counter() - start)
        times.append(best_dt)
        if num_models == 0:
            num_failures += 1
            continue
        err, idx = best_model_error(models, num_models, F_gt)
        errors.append(err)
        residuals.append(max_algebraic_residual(models[idx], x1, x2))

    report_solver_accuracy('fastpose 7-point solver (numba)', errors, residuals,
                           num_failures, num_trials)
    report_runtime(f'runtime per solve call (best of {repeats}):', times)
    return errors, residuals, times, num_failures


if __name__ == '__main__':
    run()
