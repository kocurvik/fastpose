"""Refiners on synthetic two-view data: starting from a perturbed
ground-truth model on noisy outlier-free correspondences, refinement must
succeed, lower the truncated Sampson (MSAC) cost to at least ground-truth
quality, and keep the model on its manifold (rank 2 for the fundamental
matrix; R in SO(3) and unit t for the pose-parametrized essential refiner).
"""

import numpy as np

from benchmarks.utils import (generate_relpose_data, rotation_error_deg, skew,
                              translation_error_deg)
from fastpose.refiners.essential import LMEssentialRefiner
from fastpose.refiners.fundamental import LMFundamentalRefiner
from fastpose.scorers.sampson import PoseSampsonScorer, SampsonScorer

FOCAL = 1000.0
IMAGE_SIZE = 2000.0
MAX_ERROR = 4.0 / FOCAL  # 4 px Sampson threshold in calibrated units


def make_scene(seed, num_samples=500, noise_sigma=1.0):
    rng = np.random.default_rng(seed)
    x1, x2, R, t = generate_relpose_data(rng, num_samples, noise_sigma=noise_sigma,
                                         outlier_ratio=0.0, focal=FOCAL,
                                         image_size=IMAGE_SIZE)
    c = IMAGE_SIZE / 2.0
    x1n = (x1 - c) / FOCAL
    x2n = (x2 - c) / FOCAL
    E_gt = skew(t) @ R
    E_gt /= np.linalg.norm(E_gt)
    data = (np.ascontiguousarray(x1n[:, 0]), np.ascontiguousarray(x1n[:, 1]),
            np.ascontiguousarray(x2n[:, 0]), np.ascontiguousarray(x2n[:, 1]))
    return x1n, x2n, data, E_gt, R, t


def truncated_score_f(model, x1n, x2n):
    score, inliers, num_inliers = SampsonScorer.score_numpy(
        model.reshape(3, 3), x1n, x2n, MAX_ERROR)
    return score, inliers, num_inliers


def truncated_score_pose(R, t, x1n, x2n):
    score, inliers, num_inliers = PoseSampsonScorer.score_numpy(
        R, t, x1n, x2n, MAX_ERROR)
    return score, inliers, num_inliers


def rotation_from_axis_angle(w):
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    K = skew(w / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def test_fundamental_refiner_improves_perturbed_model():
    x1n, x2n, data, E_gt, R, t = make_scene(0)
    rng = np.random.default_rng(100)
    model0 = E_gt.ravel() + 0.002 * rng.normal(size=9)
    model0 /= np.linalg.norm(model0)

    score0, _, num_inliers0 = truncated_score_f(model0, x1n, x2n)
    assert num_inliers0 > 50  # sanity: the start point sees enough inliers

    refined = np.empty(9)
    assert LMFundamentalRefiner.refine(data, model0, refined, MAX_ERROR ** 2, 50)

    score_ref, _, _ = truncated_score_f(refined, x1n, x2n)
    score_gt, _, _ = truncated_score_f(E_gt.ravel(), x1n, x2n)
    assert score_ref < score0
    assert score_ref <= score_gt * 1.05  # at least ground-truth quality

    # the factorized parametrization keeps F rank 2 by construction
    sv = np.linalg.svd(refined.reshape(3, 3), compute_uv=False)
    assert sv[2] < 1e-12 * sv[0]


def test_essential_refiner_improves_perturbed_pose():
    x1n, x2n, data, E_gt, R_gt, t_gt = make_scene(1)
    rng = np.random.default_rng(200)
    t_gt_u = t_gt / np.linalg.norm(t_gt)

    # perturb the ground-truth pose
    R0 = R_gt @ rotation_from_axis_angle(0.003 * rng.normal(size=3))
    t0 = t_gt_u + 0.003 * rng.normal(size=3)
    t0 /= np.linalg.norm(t0)
    model0 = np.concatenate([R0.ravel(), t0])

    score0, _, num_inliers0 = truncated_score_pose(R0, t0, x1n, x2n)
    assert num_inliers0 > 50

    refined = np.empty(12)
    assert LMEssentialRefiner.refine(data, model0, refined, MAX_ERROR ** 2, 50)
    R_ref = refined[:9].reshape(3, 3)
    t_ref = refined[9:12]

    # the retraction keeps R a rotation and t on the unit sphere
    np.testing.assert_allclose(R_ref.T @ R_ref, np.eye(3), atol=1e-9)
    assert np.linalg.det(R_ref) > 0.0
    assert abs(np.linalg.norm(t_ref) - 1.0) < 1e-9

    score_ref, _, _ = truncated_score_pose(R_ref, t_ref, x1n, x2n)
    score_gt, _, _ = truncated_score_pose(R_gt, t_gt_u, x1n, x2n)
    assert score_ref < score0
    assert score_ref <= score_gt * 1.05

    # the refined pose is directly comparable to the ground truth
    assert rotation_error_deg(R_ref, R_gt) < 0.5
    assert translation_error_deg(t_ref, t_gt) < 1.0


def test_refiner_does_not_degrade_ground_truth_start():
    # starting exactly at the ground truth, refinement may only improve
    # (or keep) the truncated cost
    x1n, x2n, data, E_gt, R_gt, t_gt = make_scene(2)
    t_gt_u = t_gt / np.linalg.norm(t_gt)

    score_gt, _, _ = truncated_score_f(E_gt.ravel(), x1n, x2n)
    refined = np.empty(9)
    assert LMFundamentalRefiner.refine(data, E_gt.ravel().copy(), refined,
                                       MAX_ERROR ** 2, 50)
    score_ref, _, _ = truncated_score_f(refined, x1n, x2n)
    assert score_ref <= score_gt * (1.0 + 1e-9)

    pose_gt = np.concatenate([R_gt.ravel(), t_gt_u])
    score_gt, _, _ = truncated_score_pose(R_gt, t_gt_u, x1n, x2n)
    refined = np.empty(12)
    assert LMEssentialRefiner.refine(data, pose_gt, refined, MAX_ERROR ** 2, 50)
    score_ref, _, _ = truncated_score_pose(refined[:9].reshape(3, 3),
                                           refined[9:12], x1n, x2n)
    assert score_ref <= score_gt * (1.0 + 1e-9)
