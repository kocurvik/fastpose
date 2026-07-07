"""Shared benchmark utilities: pose error metrics, synthetic data
generation (noisy scenes and noise-free minimal samples), accuracy
reporting and the mAA-vs-runtime plot.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# error metrics
# ---------------------------------------------------------------------------

def rotation_error_deg(R_est, R_gt):
    c = (np.trace(R_est @ R_gt.T) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def translation_error_deg(t_est, t_gt):
    # angular error of the translation direction, sign-invariant
    c = abs(float(np.dot(t_est, t_gt))) / (np.linalg.norm(t_est) * np.linalg.norm(t_gt))
    return math.degrees(math.acos(min(1.0, c)))


def pose_maa(errors, max_threshold=10):
    # mean Average Accuracy: mean over thresholds 1..max_threshold degrees of
    # the fraction of runs whose pose error max(rot, trans) is below threshold
    errors = np.asarray(errors)
    return float(np.mean([(errors < t).mean() for t in range(1, max_threshold + 1)]))


def skew(t):
    return np.array([[0.0, -t[2], t[1]],
                     [t[2], 0.0, -t[0]],
                     [-t[1], t[0], 0.0]])


# ---------------------------------------------------------------------------
# synthetic data generation
# ---------------------------------------------------------------------------

def _random_rank2_f(rng):
    u, _, vt = np.linalg.svd(rng.normal(size=(3, 3)))
    F = u @ np.diag([1.0, 0.35, 0.0]) @ vt
    return F / np.linalg.norm(F)


def _points_on_epipolar_lines(rng, F, num_samples):
    # exact correspondences for F in [-1, 1]^2 coordinates: x1 drawn
    # uniformly, x2 placed on the epipolar line of x1 (solving for the
    # better-conditioned coordinate)
    x1 = rng.uniform(-1.0, 1.0, size=(num_samples, 2))
    x1_h = np.column_stack([x1, np.ones(num_samples)])
    lines = x1_h @ F.T

    x2 = np.empty_like(x1)
    random_coord = rng.uniform(-1.0, 1.0, size=num_samples)
    solve_y = np.abs(lines[:, 1]) > np.abs(lines[:, 0])

    x2[solve_y, 0] = random_coord[solve_y]
    x2[solve_y, 1] = -(lines[solve_y, 0] * x2[solve_y, 0] + lines[solve_y, 2]) / lines[solve_y, 1]

    x2[~solve_y, 1] = random_coord[~solve_y]
    x2[~solve_y, 0] = -(lines[~solve_y, 1] * x2[~solve_y, 1] + lines[~solve_y, 2]) / lines[~solve_y, 0]
    return x1, x2


def generate_data(rng, num_samples, noise_sigma=1.0, outlier_ratio=0.2,
                  image_size=1000.0, return_gt=False):
    # synthetic correspondences in pixel-scale coordinates [0, image_size]^2
    # with Gaussian pixel noise on inliers and uniformly drawn outliers;
    # the returned true_F refers to the pre-pixel [-1, 1] coordinates
    true_F = _random_rank2_f(rng)
    x1, x2 = _points_on_epipolar_lines(rng, true_F, num_samples)

    # map from [-1, 1] to pixel coordinates, then corrupt in pixel space
    x1 = (x1 + 1.0) * (0.5 * image_size)
    x2 = (x2 + 1.0) * (0.5 * image_size)

    x2 += rng.normal(scale=noise_sigma, size=x2.shape)

    num_outliers = int(num_samples * outlier_ratio)
    outlier_idxs = rng.choice(num_samples, num_outliers, replace=False)
    inlier_mask = np.ones(num_samples, dtype=np.bool_)
    inlier_mask[outlier_idxs] = False
    x2[outlier_idxs] = rng.uniform(0.0, image_size, size=(num_outliers, 2))

    if return_gt:
        return x1, x2, true_F, inlier_mask
    return x1, x2


def generate_relpose_data(rng, num_samples, noise_sigma=1.0, outlier_ratio=0.2,
                          focal=1000.0, image_size=2000.0):
    # synthetic two-view correspondences in pixel coordinates with a known
    # relative pose; returns pixel points plus the ground truth (R, t)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = 0.2
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    t = rng.normal(size=3)
    t *= 0.5 / np.linalg.norm(t)

    x1n = np.empty((0, 2))
    x2n = np.empty((0, 2))
    while len(x1n) < num_samples:
        m = 2 * num_samples
        xn = rng.uniform(-0.45, 0.45, size=(m, 2))
        depth = rng.uniform(4.0, 10.0, size=m)
        X = np.column_stack([xn * depth[:, None], depth])
        X2 = X @ R.T + t
        proj = X2[:, :2] / X2[:, 2:3]
        valid = (X2[:, 2] > 0.5) & (np.abs(proj[:, 0]) < 0.48) & (np.abs(proj[:, 1]) < 0.48)
        x1n = np.concatenate([x1n, xn[valid]])
        x2n = np.concatenate([x2n, proj[valid]])
    x1n = x1n[:num_samples]
    x2n = x2n[:num_samples]

    c = image_size / 2.0
    x1 = focal * x1n + c
    x2 = focal * x2n + c
    x2 += rng.normal(scale=noise_sigma, size=x2.shape)

    num_outliers = int(num_samples * outlier_ratio)
    outlier_idxs = rng.choice(num_samples, num_outliers, replace=False)
    x2[outlier_idxs] = rng.uniform(0.0, image_size, size=(num_outliers, 2))

    return x1, x2, R, t


def generate_varying_focal_relpose_data(rng, num_samples, noise_sigma=1.0,
                                        outlier_ratio=0.2, focal1=800.0,
                                        focal2=1300.0, pp1=(500.0, 480.0),
                                        pp2=(620.0, 510.0)):
    # synthetic two-view correspondences with different focal lengths and
    # known principal points. Noise is added in second-image pixel units.
    y1, y2, R, t = generate_relpose_data(
        rng, num_samples, noise_sigma=0.0, outlier_ratio=0.0,
        focal=1.0, image_size=2.0)
    pp1 = np.asarray(pp1, dtype=np.float64)
    pp2 = np.asarray(pp2, dtype=np.float64)
    x1 = focal1 * (y1 - 1.0) + pp1
    x2 = focal2 * (y2 - 1.0) + pp2
    x2 += rng.normal(scale=noise_sigma, size=x2.shape)

    num_outliers = int(num_samples * outlier_ratio)
    if num_outliers:
        outlier_idxs = rng.choice(num_samples, num_outliers, replace=False)
        x2[outlier_idxs, 0] = rng.uniform(0.0, 2.0 * pp2[0], size=num_outliers)
        x2[outlier_idxs, 1] = rng.uniform(0.0, 2.0 * pp2[1], size=num_outliers)

    return x1, x2, R, t, pp1, pp2


def generate_shared_focal_relpose_data(rng, num_samples, noise_sigma=1.0,
                                       outlier_ratio=0.2, focal=1000.0,
                                       pp1=(500.0, 480.0),
                                       pp2=(620.0, 510.0)):
    # synthetic two-view correspondences with one shared focal length and
    # known principal points. Noise is added in second-image pixel units.
    return generate_varying_focal_relpose_data(
        rng, num_samples, noise_sigma=noise_sigma,
        outlier_ratio=outlier_ratio, focal1=focal, focal2=focal,
        pp1=pp1, pp2=pp2)


def generate_abspose_data(rng, num_samples, noise_sigma=1.0, outlier_ratio=0.2,
                          focal=1000.0, image_size=2000.0):
    # synthetic 2D-3D correspondences in pixel coordinates with a known
    # absolute pose (X_cam = R X_world + t); returns pixel points, world
    # points and the ground truth (R, t)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.0, math.pi)
    K = skew(axis)
    R = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    t = rng.normal(size=3)

    # sample points inside the camera frustum, then map them to world space
    xn = rng.uniform(-0.45, 0.45, size=(num_samples, 2))
    depth = rng.uniform(4.0, 10.0, size=num_samples)
    X_cam = np.column_stack([xn * depth[:, None], depth])
    X = (X_cam - t) @ R  # rows are R^T (X_cam - t)

    c = image_size / 2.0
    x = focal * xn + c
    x += rng.normal(scale=noise_sigma, size=x.shape)

    num_outliers = int(num_samples * outlier_ratio)
    if num_outliers:
        outlier_idxs = rng.choice(num_samples, num_outliers, replace=False)
        x[outlier_idxs] = rng.uniform(0.0, image_size, size=(num_outliers, 2))

    return x, X, R, t


def generate_exact_fundamental_sample(rng, num_samples=7):
    # noise-free correspondences in [-1, 1]^2 for a random rank-2 F
    F = _random_rank2_f(rng)
    x1, x2 = _points_on_epipolar_lines(rng, F, num_samples)
    return x1, x2, F


def generate_exact_essential_sample(rng, num_samples=5, focal=1000.0,
                                    image_size=2000.0):
    # noise-free *calibrated* correspondences from a random relative pose;
    # returns the ground-truth essential matrix E = [t]_x R (unit norm)
    x1, x2, R, t = generate_relpose_data(rng, num_samples, noise_sigma=0.0,
                                         outlier_ratio=0.0, focal=focal,
                                         image_size=image_size)
    c = image_size / 2.0
    x1n = (x1 - c) / focal
    x2n = (x2 - c) / focal
    E = skew(t) @ R
    return x1n, x2n, E / np.linalg.norm(E), R, t


# ---------------------------------------------------------------------------
# solver accuracy helpers
# ---------------------------------------------------------------------------

def best_model_error(models, num_models, gt):
    # smallest Frobenius distance between any returned model and the ground
    # truth, both unit-normalized, minimized over the sign ambiguity;
    # returns (error, index of the closest model)
    gt = gt.ravel() / np.linalg.norm(gt)
    best = np.inf
    best_idx = -1
    for m in range(num_models):
        model = models[m] / np.linalg.norm(models[m])
        err = min(np.linalg.norm(model - gt), np.linalg.norm(model + gt))
        if err < best:
            best = err
            best_idx = m
    return best, best_idx


def max_algebraic_residual(model, x1, x2):
    # max |x2^T F x1| over the sample for the unit-normalized model
    F = (model / np.linalg.norm(model)).reshape(3, 3)
    ones = np.ones(len(x1))
    x1h = np.column_stack([x1, ones])
    x2h = np.column_stack([x2, ones])
    return float(np.max(np.abs(np.sum(x2h * (x1h @ F.T), axis=1))))


def report_runtime(label, times_s):
    # per-call runtime statistics in microseconds
    t = np.asarray(times_s) * 1e6
    med, p90, mx = np.percentile(t, [50, 90, 100])
    print(f'{label} median={med:.1f} us  mean={t.mean():.1f} us  '
          f'p90={p90:.1f} us  max={mx:.1f} us')


def report_solver_accuracy(name, errors, residuals, num_failures, num_trials):
    errors = np.asarray(errors)
    residuals = np.asarray(residuals)
    print(f'--- {name} ---')
    print(f'no model returned: {num_failures}')
    if len(errors) == 0:
        return
    recovered = int(np.count_nonzero(errors < 1e-6))
    print(f'ground truth recovered (model error < 1e-6): {recovered}/{num_trials}')
    for label, vals in (('model error vs ground truth ', errors),
                        ('max |algebraic residual|    ', residuals)):
        med, p90, p99, mx = np.percentile(vals, [50, 90, 99, 100])
        print(f'{label} median={med:.2e}  p90={p90:.2e}  p99={p99:.2e}  max={mx:.2e}')


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_maa_tradeoff(results, methods, iterations_list, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, NullFormatter

    surface = '#fcfcfb'
    ink = '#0b0b0b'
    ink2 = '#52514e'
    muted = '#898781'
    grid = '#e1e0d9'
    baseline = '#c3c2b7'
    colors = {'numba': '#2a78d6', 'numba+LO': '#1baf7a', 'poselib': '#eda100',
              'fundamental+gtK': '#2a78d6', 'varying-f': '#8b5cf6',
              'varying-f+LO': '#1baf7a', 'p3p+gtf': '#2a78d6',
              'p4pf': '#8b5cf6', 'p4pf+LO': '#1baf7a',
              'shared-f': '#d14f2a', 'shared-f+LO': '#1baf7a'}
    markers = {'numba': 'o', 'numba+LO': 's', 'poselib': '^',
               'fundamental+gtK': 'o', 'varying-f': 'D',
               'varying-f+LO': 's', 'p3p+gtf': 'o', 'p4pf': 'D',
               'p4pf+LO': 's', 'shared-f': 'P', 'shared-f+LO': 's'}

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)

    for m in methods:
        rt = results[m]['runtime_ms']
        maa = results[m]['maa']
        ax.plot(rt, maa, color=colors[m], marker=markers[m], linewidth=2,
                markersize=7, label=m, zorder=3)
        for x, y, iters in zip(rt, maa, iterations_list):
            ax.annotate(f'{iters}', (x, y), textcoords='offset points',
                        xytext=(0, -13), ha='center', fontsize=8, color=muted,
                        zorder=2)
        ax.annotate(m, (rt[-1], maa[-1]), textcoords='offset points',
                    xytext=(9, 2), fontsize=9, color=ink2, zorder=4)

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.margins(x=0.18)
    ax.set_xlabel('mean runtime per image pair (ms)', color=ink2, fontsize=10)
    ax.set_ylabel('pose mAA(10\N{DEGREE SIGN})', color=ink2, fontsize=10)
    ax.set_title(title, color=ink, fontsize=11, pad=12)
    ax.grid(True, which='major', color=grid, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted, labelsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(baseline)
    ax.legend(frameon=False, loc='lower right', fontsize=9, labelcolor=ink2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
