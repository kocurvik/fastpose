"""CUDA backend tests: every kernel checked against its CPU counterpart.

The whole module skips when no usable CUDA device is present, so the suite
still runs on CPU-only machines.

The parity tolerances are not arbitrary. Solver and refiner kernels are
compiled from the same source for both backends, but NVVM and LLVM contract
and reassociate `fastmath` expressions differently, so agreement is to a few
ulps rather than bit-exact. The scorer is a tree reduction on the GPU against
a sequential sum on the CPU, which is a summation-order difference of the same
size. Inlier *counts* are integers and are expected to agree exactly.
"""

import numpy as np
import pytest

from benchmarks.utils import (generate_relpose_data, rotation_error_deg,
                              translation_error_deg)
from fastpose import cuda as cuda_backend

pytestmark = pytest.mark.skipif(
    not cuda_backend.is_available(),
    reason=f"no usable CUDA device: {cuda_backend.unavailable_reason()}")

FOCAL = 1000.0
IMAGE_SIZE = 2000.0
PP = np.array([IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0])


def make_scene(seed, num_samples=2000, noise_sigma=0.5, outlier_ratio=0.3):
    rng = np.random.default_rng(seed)
    x1, x2, R, t = generate_relpose_data(
        rng, num_samples, noise_sigma=noise_sigma, outlier_ratio=outlier_ratio,
        focal=FOCAL, image_size=IMAGE_SIZE)
    return (x1 - PP) / FOCAL, (x2 - PP) / FOCAL, R, t


def columns(x1, x2):
    return tuple(np.ascontiguousarray(c) for c in
                 (x1[:, 0], x1[:, 1], x2[:, 0], x2[:, 1]))


def draw_samples(rng, n, count):
    return np.stack([rng.choice(n, 5, replace=False)
                     for _ in range(count)]).astype(np.int64)


# ---------------------------------------------------------------------------
# minimal solver
# ---------------------------------------------------------------------------

def test_cuda_solver_matches_cpu_solver():
    from numba import cuda
    from fastpose.cuda.solvers import MAX_MODELS, allocate_models, solve_batch
    from fastpose.solvers.essential import FivePointSolver, _solve_5pt

    x1, x2, _, _ = make_scene(0, num_samples=500, noise_sigma=0.0,
                              outlier_ratio=0.0)
    data = columns(x1, x2)
    rng = np.random.default_rng(7)
    batch = 128
    samples = draw_samples(rng, len(x1), batch)

    cpu_models = np.zeros((batch, MAX_MODELS, 12))
    cpu_counts = np.zeros(batch, dtype=np.int64)
    workspace = np.empty(FivePointSolver.workspace_size)
    for i in range(batch):
        m = np.zeros((MAX_MODELS, 12))
        cpu_counts[i] = _solve_5pt(data, samples[i], m, workspace)
        cpu_models[i] = m

    d_data = tuple(cuda.to_device(c) for c in data)
    d_models, d_counts = allocate_models(batch)
    solve_batch(d_data, cuda.to_device(samples), d_models, d_counts)
    cuda.synchronize()
    gpu_models = d_models.copy_to_host()
    gpu_counts = d_counts.copy_to_host()

    # the solver's branches are all on comparisons against tolerances well
    # above the codegen difference, so the *number* of models must agree
    np.testing.assert_array_equal(cpu_counts, gpu_counts)
    assert cpu_counts.sum() > 0
    for i in range(batch):
        k = cpu_counts[i]
        if k:
            np.testing.assert_allclose(gpu_models[i, :k], cpu_models[i, :k],
                                       rtol=1e-6, atol=1e-6)


def test_cuda_solver_recovers_exact_pose_on_clean_scene():
    from numba import cuda
    from fastpose.cuda.solvers import allocate_models, solve_batch

    x1, x2, R_gt, t_gt = make_scene(1, num_samples=200, noise_sigma=0.0,
                                    outlier_ratio=0.0)
    data = columns(x1, x2)
    rng = np.random.default_rng(3)
    batch = 64
    d_data = tuple(cuda.to_device(c) for c in data)
    d_models, d_counts = allocate_models(batch)
    solve_batch(d_data, cuda.to_device(draw_samples(rng, len(x1), batch)),
                d_models, d_counts)
    cuda.synchronize()
    models = d_models.copy_to_host()
    counts = d_counts.copy_to_host()

    best = min(max(rotation_error_deg(models[i, m, :9].reshape(3, 3), R_gt),
                   translation_error_deg(models[i, m, 9:12], t_gt))
               for i in range(batch) for m in range(counts[i]))
    assert best < 1e-6


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------

def test_cuda_scorer_matches_cpu_scorer():
    from numba import cuda
    from fastpose.cuda.scorers import ScoreBuffers, score_batch
    from fastpose.cuda.solvers import MAX_MODELS, allocate_models, solve_batch
    from fastpose.scorers.sampson import pose_sampson_cheirality_score

    x1, x2, _, _ = make_scene(2, num_samples=3000)
    data = columns(x1, x2)
    max_error_sq = (2.0 / FOCAL) ** 2
    rng = np.random.default_rng(11)
    batch = 128

    d_data = tuple(cuda.to_device(c) for c in data)
    d_models, d_counts = allocate_models(batch)
    solve_batch(d_data, cuda.to_device(draw_samples(rng, len(x1), batch)),
                d_models, d_counts)
    cuda.synchronize()
    models = d_models.copy_to_host()
    counts = d_counts.copy_to_host()

    buffers = ScoreBuffers(batch)
    score_batch(d_data, d_models, d_counts, max_error_sq, buffers, batch)
    cuda.synchronize()
    g_score, g_idx, g_inl, g_max_inl, g_max_idx = buffers.to_host(batch)

    checked = 0
    for i in range(batch):
        if counts[i] == 0:
            assert g_idx[i] == -1
            continue
        checked += 1
        # a 1e300 bound disables the CPU scorer's early bail-out, which is
        # what the GPU kernel has no equivalent of
        scored = [pose_sampson_cheirality_score(models[i, m], data,
                                                max_error_sq, 1e300)
                  for m in range(counts[i])]
        costs = [s for s, _ in scored]
        best = int(np.argmin(costs))

        # The GPU scorer is float32 per point (float64 accumulator), so a
        # point sitting within float32 rounding of the threshold can flip.
        # Measured over 200 real models at 16k matches: counts agree on
        # ~99% of models and never differ by more than one point.
        assert abs(int(g_inl[i]) - scored[best][1]) <= 1
        assert abs(int(g_max_inl[i]) - max(k for _, k in scored)) <= 1
        np.testing.assert_allclose(g_score[i], costs[best], rtol=1e-5)

        # what actually matters is that the *selection* is as good, not that
        # the index is identical - two models can score within rounding of
        # each other, in which case either is a correct pick
        assert costs[int(g_idx[i])] <= costs[best] * (1.0 + 1e-5)
    assert checked > 0


# ---------------------------------------------------------------------------
# local optimization
# ---------------------------------------------------------------------------

def test_cuda_lm_matches_cpu_lm():
    from numba import cuda
    from fastpose.cuda.refiners import (COL_SUCCESS, RefineBuffers,
                                        refine_batch)
    from fastpose.cuda.solvers import MAX_MODELS
    from fastpose.refiners.essential import LMEssentialRefiner
    from fastpose.refiners.losses import TruncatedLoss
    from fastpose.refiners.utils import LO_INLIER_SCALE
    from fastpose.solvers.essential import FivePointSolver, _solve_5pt

    x1, x2, _, _ = make_scene(4, num_samples=3000)
    data = columns(x1, x2)
    max_error_sq = (2.0 / FOCAL) ** 2
    lo_iterations = 25

    rng = np.random.default_rng(13)
    workspace = np.empty(FivePointSolver.workspace_size)
    candidates = []
    while len(candidates) < 16:
        sample = rng.choice(len(x1), 5, replace=False).astype(np.int64)
        m = np.zeros((MAX_MODELS, 12))
        for k in range(_solve_5pt(data, sample, m, workspace)):
            if len(candidates) < 16:
                candidates.append(m[k].copy())
    candidates = np.array(candidates)
    k = len(candidates)

    refiner = LMEssentialRefiner(relaxed_inlier_scale=LO_INLIER_SCALE)
    cpu_refined = np.zeros((k, 12))
    cpu_ok = np.zeros(k, dtype=np.int64)
    for i in range(k):
        out = np.empty(12)
        cpu_ok[i] = int(refiner.refine(data, candidates[i], out, max_error_sq,
                                       lo_iterations))
        if cpu_ok[i]:
            cpu_refined[i] = out

    d_data = tuple(cuda.to_device(c) for c in data)
    d_models = cuda.to_device(candidates.reshape(k, 1, 12))
    buffers = RefineBuffers(k, len(x1))
    refine_batch(d_data, d_models, buffers, np.arange(k, dtype=np.int64),
                 np.zeros(k, dtype=np.int64), k, max_error_sq, lo_iterations,
                 TruncatedLoss())
    cuda.synchronize()
    result = buffers.refined.copy_to_host()
    gpu_refined = result[:, 0, :12]
    gpu_ok = (result[:, 0, COL_SUCCESS] != 0.0).astype(np.int64)

    np.testing.assert_array_equal(cpu_ok, gpu_ok)
    assert cpu_ok.sum() > 0
    both = cpu_ok == 1

    # The GPU LM evaluates residuals and the jacobian in float32 (accumulating
    # JtJ/Jtr and solving the 5x5 system in float64), so the converged pose
    # differs at float32 level rather than matching bit for bit.
    np.testing.assert_allclose(gpu_refined[both], cpu_refined[both],
                               rtol=1e-3, atol=1e-4)

    # The assertion that matters: scored in full float64, the mixed-precision
    # result must be as good a minimizer as the float64 CPU refiner's. This is
    # what would catch a genuine precision regression - an element-wise
    # tolerance alone would not distinguish "rounded differently" from
    # "converged somewhere worse".
    from fastpose.scorers.sampson import pose_sampson_cheirality_score
    for i in np.nonzero(both)[0]:
        cpu_cost = pose_sampson_cheirality_score(cpu_refined[i], data,
                                                 max_error_sq, 1e300)[0]
        gpu_cost = pose_sampson_cheirality_score(gpu_refined[i], data,
                                                 max_error_sq, 1e300)[0]
        assert gpu_cost <= cpu_cost * (1.0 + 1e-4)


def test_cuda_lm_improves_the_cost_it_minimizes():
    from numba import cuda
    from fastpose.cuda.refiners import (COL_SUCCESS, RefineBuffers,
                                        refine_batch)
    from fastpose.cuda.solvers import MAX_MODELS
    from fastpose.refiners.losses import TruncatedLoss
    from fastpose.scorers.sampson import pose_sampson_cheirality_score
    from fastpose.solvers.essential import FivePointSolver, _solve_5pt

    x1, x2, _, _ = make_scene(5, num_samples=2000)
    data = columns(x1, x2)
    max_error_sq = (2.0 / FOCAL) ** 2

    rng = np.random.default_rng(17)
    workspace = np.empty(FivePointSolver.workspace_size)
    candidates = []
    while len(candidates) < 8:
        sample = rng.choice(len(x1), 5, replace=False).astype(np.int64)
        m = np.zeros((MAX_MODELS, 12))
        for j in range(_solve_5pt(data, sample, m, workspace)):
            if len(candidates) < 8:
                candidates.append(m[j].copy())
    candidates = np.array(candidates)
    k = len(candidates)

    d_data = tuple(cuda.to_device(c) for c in data)
    buffers = RefineBuffers(k, len(x1))
    refine_batch(d_data, cuda.to_device(candidates.reshape(k, 1, 12)),
                 buffers, np.arange(k, dtype=np.int64),
                 np.zeros(k, dtype=np.int64), k, max_error_sq, 25,
                 TruncatedLoss())
    cuda.synchronize()
    result = buffers.refined.copy_to_host()
    refined = result[:, 0, :12]
    ok = result[:, 0, COL_SUCCESS] != 0.0

    improved = 0
    for i in range(k):
        if not ok[i]:
            continue
        before, _ = pose_sampson_cheirality_score(candidates[i], data,
                                                  max_error_sq, 1e300)
        after, _ = pose_sampson_cheirality_score(refined[i], data,
                                                 max_error_sq, 1e300)
        assert after <= before * (1.0 + 1e-9)
        improved += int(after < before)
    assert improved > 0


# ---------------------------------------------------------------------------
# end-to-end estimator
# ---------------------------------------------------------------------------

def test_cuda_estimator_recovers_exact_pose():
    from fastpose.estimators.essential import estimate_relative_pose

    x1, x2, R_gt, t_gt = make_scene(0, num_samples=200, noise_sigma=0.0,
                                    outlier_ratio=0.0)
    model, info = estimate_relative_pose(
        x1, x2, iterations=200, min_iterations=200, max_error=1e-3, seed=0,
        lo_iterations=0, device='cuda')

    assert info['num_inliers'] == len(x1)
    assert np.all(info['inliers'])
    assert rotation_error_deg(model['R'], R_gt) < 1e-4
    assert translation_error_deg(model['t'], t_gt) < 1e-4


def test_cuda_estimator_agrees_with_cpu_on_outlier_scene():
    from fastpose.estimators.essential import estimate_relative_pose

    x1, x2, R_gt, t_gt = make_scene(1, num_samples=4000)
    kwargs = dict(iterations=2000, min_iterations=2000,
                  max_error=2.0 / FOCAL, seed=0)
    cpu_model, cpu_info = estimate_relative_pose(x1, x2, device='cpu', **kwargs)
    gpu_model, gpu_info = estimate_relative_pose(x1, x2, device='cuda', **kwargs)

    # the two drivers draw different samples and optimize different
    # candidates, so they are compared on the quality of what they return
    # rather than on identical output
    assert gpu_info['num_inliers'] >= cpu_info['num_inliers'] * 0.98
    assert rotation_error_deg(gpu_model['R'], R_gt) < 1.0
    assert translation_error_deg(gpu_model['t'], t_gt) < 1.0
    assert rotation_error_deg(gpu_model['R'], cpu_model['R']) < 1.0


def test_cuda_estimator_info_fields():
    from fastpose.estimators.essential import estimate_relative_pose

    x1, x2, _, _ = make_scene(2, num_samples=1000)
    model, info = estimate_relative_pose(
        x1, x2, iterations=500, min_iterations=500, max_error=2.0 / FOCAL,
        seed=0, device='cuda')

    assert set(info) == {'inliers', 'num_inliers', 'model_score',
                         'iterations', 'refinements'}
    assert info['inliers'].dtype == np.bool_
    assert info['inliers'].shape == (len(x1),)
    assert np.count_nonzero(info['inliers']) == info['num_inliers']
    assert info['model_score'] >= 0.0
    assert info['iterations'] > 0
    assert info['refinements'] in (0, 1)


def test_cuda_estimator_is_reproducible_for_a_seed():
    from fastpose.estimators.essential import estimate_relative_pose

    x1, x2, _, _ = make_scene(3, num_samples=1500)
    kwargs = dict(iterations=1000, min_iterations=1000,
                  max_error=2.0 / FOCAL, seed=99, device='cuda')
    a_model, a_info = estimate_relative_pose(x1, x2, **kwargs)
    b_model, b_info = estimate_relative_pose(x1, x2, **kwargs)

    np.testing.assert_array_equal(a_model['R'], b_model['R'])
    np.testing.assert_array_equal(a_model['t'], b_model['t'])
    assert a_info['num_inliers'] == b_info['num_inliers']


def test_cuda_estimator_failure_returns_generic_pose():
    from fastpose.estimators.essential import estimate_relative_pose

    rng = np.random.default_rng(3)
    # max_error=0.0 means no correspondence can satisfy the inlier threshold
    x1 = rng.uniform(-0.5, 0.5, size=(50, 2))
    x2 = rng.uniform(-0.5, 0.5, size=(50, 2))

    model, info = estimate_relative_pose(
        x1, x2, iterations=100, min_iterations=100, max_error=0.0, seed=0,
        device='cuda')

    assert info['num_inliers'] == 0
    assert info['refinements'] == 0
    assert not np.any(info['inliers'])
    np.testing.assert_array_equal(model['R'], np.eye(3))
    np.testing.assert_array_equal(model['t'], np.zeros(3))


def test_cuda_estimator_respects_batch():
    from fastpose.estimators.essential import estimate_relative_pose

    x1, x2, R_gt, _ = make_scene(6, num_samples=1200)
    model, info = estimate_relative_pose(
        x1, x2, iterations=900, min_iterations=900, max_error=2.0 / FOCAL,
        seed=0, device='cuda', batch=256)

    # 900 iterations at batch=256 is four rounds, the last a partial one
    assert info['iterations'] == 900
    assert rotation_error_deg(model['R'], R_gt) < 1.0


def test_cuda_driver_refines_at_most_one_candidate_per_round():
    # the local-optimization budget is controlled by the gate (a minimal model
    # must improve the best minimal score or inlier count), not by a candidate
    # count. Refining more than one per round measured as pure waste, so the
    # driver refines exactly one - this pins that down.
    import fastpose.cuda.ransac as ransac_module
    from fastpose.estimators.essential import estimate_relative_pose

    calls = []
    original = ransac_module.refine_prepared

    def spy(data, buffers, num_candidates, *args, **kwargs):
        calls.append(num_candidates)
        return original(data, buffers, num_candidates, *args, **kwargs)

    x1, x2, _, _ = make_scene(7, num_samples=3000)
    ransac_module.refine_prepared = spy
    try:
        estimate_relative_pose(
            x1, x2, iterations=4000, min_iterations=4000,
            max_error=2.0 / FOCAL, seed=0, device='cuda', batch=512)
    finally:
        ransac_module.refine_prepared = original

    assert calls, "local optimization never ran"
    assert set(calls) == {1}, f"expected one candidate per launch, got {calls}"
    # and the gate must actually gate: far fewer launches than rounds
    assert len(calls) < 4000 // 512


def test_unknown_device_raises():
    from fastpose.estimators.essential import estimate_relative_pose

    x1, x2, _, _ = make_scene(0, num_samples=50)
    with pytest.raises(ValueError, match="device must be"):
        estimate_relative_pose(x1, x2, device='tpu')
