"""CUDA backend tests: every kernel checked against its CPU counterpart.

The whole module skips when no usable CUDA device is present, so the suite
still runs on CPU-only machines.

Layout
------
The kernel-level tests are parameterized over every ported problem. A `Case`
carries the one thing that differs between them - the scene, the data layout,
and the three CPU references (minimal solver, scorer, LM refiner) the GPU
kernels must agree with - so `test_cuda_solver_matches_cpu` and friends are
written once. The end-to-end tests stay on `estimate_relative_pose`, which is
where the driver, the adaptive termination and the local-optimization gate
live; those are problem-agnostic, and a per-problem smoke test covers the
plumbing of the rest.

The parity tolerances are not arbitrary. Solver and refiner kernels are
compiled from the same source for both backends, but NVVM and LLVM contract
and reassociate `fastmath` expressions differently, so agreement is to a few
ulps rather than bit-exact. The scorer is a tree reduction on the GPU against
a sequential sum on the CPU, which is a summation-order difference of the same
size. Inlier *counts* are integers and are expected to agree exactly.

The scorer and the LM kernels are built with `real=float32`. They must
therefore be handed the **float32** coordinate columns: passing the float64
ones compiles fine and silently promotes the whole per-point chain back to
double, which means the mixed-precision path under test is never exercised.
"""

import numpy as np
import pytest

from benchmarks.utils import (generate_abspose_data, generate_data,
                              generate_homography_data,
                              generate_monodepth_relpose_data,
                              generate_relpose_data,
                              generate_shared_focal_relpose_data,
                              generate_varying_focal_relpose_data,
                              max_algebraic_residual, max_transfer_residual,
                              rotation_error_deg, translation_error_deg)
from fastpose import cuda as cuda_backend

pytestmark = pytest.mark.skipif(
    not cuda_backend.is_available(),
    reason=f"no usable CUDA device: {cuda_backend.unavailable_reason()}")

FOCAL = 1000.0
IMAGE_SIZE = 2000.0
PP = np.array([IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0])

VF_FOCAL1 = 800.0
VF_FOCAL2 = 1300.0
VF_PP1 = (500.0, 480.0)
VF_PP2 = (620.0, 510.0)


# ---------------------------------------------------------------------------
# problem cases
# ---------------------------------------------------------------------------

def _solver_models_allclose(cpu, gpu):
    # the default: both backends compile the same source, and NVVM's and
    # LLVM's `fastmath` associations differ by a few ulps
    np.testing.assert_allclose(gpu, cpu, rtol=1e-6, atol=1e-6)


def _solver_models_shared_focal(cpu, gpu):
    # The 6-point solver cannot be held to an elementwise tolerance, and
    # loosening one until it passes would assert nothing.
    #
    # It is the one solver built *without* fastmath, because its pipeline -
    # 31x46 elimination template, 15x15 Danilevsky, Sturm isolation - is
    # ill-conditioned enough to amplify last-bit differences into the 8th
    # significant digit (see solvers/shared_focal.py). NVVM associates
    # differently from LLVM regardless of that flag, so the per-model relative
    # difference is extremely heavy-tailed: measured over 512 samples, median
    # 2.5e-12 and p90 2.1e-10, but a maximum of 5.5e-2 on the near-degenerate
    # samples the ill-conditioning bites hardest.
    #
    # So this asserts the shape of that distribution instead. A real bug -
    # a transposed index, a wrong constant, a truncated table - moves the
    # median and the bulk, not just the tail.
    denom = np.maximum(np.abs(cpu), 1e-12)
    rel = np.max(np.abs(cpu - gpu) / denom, axis=1)
    assert np.median(rel) < 1e-9, f"median relative difference {np.median(rel)}"
    within = float(np.mean(rel < 1e-6))
    assert within > 0.95, f"only {within:.1%} of models agree to 1e-6"


class Case():
    """One problem's scene plus the CPU references the GPU must match.

    `data` is the tuple the CPU solver/scorer/refiner take, `cols` the
    coordinate columns the device kernels take. They differ only for
    varying-focal, whose principal points ride in `params` on the GPU (see
    Instructions.md section 2.4 on the `cuda-backend` branch) and in the
    `data` tuple on the CPU.
    """

    def __init__(self, name, data, cols, params, max_error_sq, solver,
                 cpu_solve, cpu_score, refiner, error,
                 check_solver_models=None, lm_cost=None):
        self.name = name
        self.check_solver_models = (check_solver_models
                                    or _solver_models_allclose)
        # what the LM actually minimizes, which is the scorer's functional for
        # every problem except monodepth - there it is the *hybrid* of the
        # Sampson error and the symmetric reprojection error, so asking the
        # refiner to improve the Sampson score alone would be asking for the
        # wrong thing. Same signature as a scorer.
        self.lm_cost = lm_cost or cpu_score
        self.data = data
        self.cols = cols
        self.params = params
        self.max_error_sq = max_error_sq
        self.sample_size = solver.sample_size
        self.num_params = solver.num_params
        self.max_models = solver.max_models
        self.workspace_size = solver.workspace_size
        self.cpu_solve = cpu_solve
        self.cpu_score = cpu_score
        self.refiner = refiner
        self.error = error
        self.num_points = cols[0].shape[0]


def _essential_case(seed, n, noise, outliers):
    from fastpose.estimators.utils import point_columns
    from fastpose.refiners.essential import LMEssentialRefiner
    from fastpose.refiners.utils import LO_INLIER_SCALE
    from fastpose.scorers.sampson import pose_sampson_cheirality_score
    from fastpose.solvers.essential import FivePointSolver, _solve_5pt

    rng = np.random.default_rng(seed)
    x1, x2, R, t = generate_relpose_data(
        rng, n, noise_sigma=noise, outlier_ratio=outliers, focal=FOCAL,
        image_size=IMAGE_SIZE)
    cols = point_columns((x1 - PP) / FOCAL, (x2 - PP) / FOCAL)

    def error(m):
        return max(rotation_error_deg(m[:9].reshape(3, 3), R),
                   translation_error_deg(m[9:12], t))

    return Case('essential', cols, cols, None, (2.0 / FOCAL) ** 2,
                FivePointSolver, _solve_5pt, pose_sampson_cheirality_score,
                LMEssentialRefiner(relaxed_inlier_scale=LO_INLIER_SCALE),
                error)


def _abspose_columns(x, X):
    return (np.ascontiguousarray(x[:, 0]), np.ascontiguousarray(x[:, 1]),
            np.ascontiguousarray(X[:, 0]), np.ascontiguousarray(X[:, 1]),
            np.ascontiguousarray(X[:, 2]))


def _absolute_case(seed, n, noise, outliers):
    from fastpose.refiners.absolute import LMAbsolutePoseRefiner
    from fastpose.scorers.reprojection import reprojection_score
    from fastpose.solvers.p3p import P3PSolver, _solve_p3p

    rng = np.random.default_rng(seed)
    x, X, R, t = generate_abspose_data(
        rng, n, noise_sigma=noise, outlier_ratio=outliers, focal=FOCAL,
        image_size=IMAGE_SIZE)
    cols = _abspose_columns((x - PP) / FOCAL, X)

    def error(m):
        # absolute pose is metric, so the translation is compared as a vector
        return max(rotation_error_deg(m[:9].reshape(3, 3), R),
                   float(np.linalg.norm(m[9:12] - t)))

    return Case('absolute', cols, cols, None, (2.0 / FOCAL) ** 2, P3PSolver,
                _solve_p3p, reprojection_score, LMAbsolutePoseRefiner(), error)


def _absolute_focal_case(seed, n, noise, outliers):
    from fastpose.refiners.absolute_focal import LMAbsolutePoseFocalRefiner
    from fastpose.scorers.reprojection import focal_reprojection_score
    from fastpose.solvers.p4pf import P4PFSolver, _solve_p4pf

    rng = np.random.default_rng(seed)
    x, X, R, t = generate_abspose_data(
        rng, n, noise_sigma=noise, outlier_ratio=outliers, focal=FOCAL,
        image_size=IMAGE_SIZE)
    # the P4Pf model is [R | t | f] with the residual in principal-point-
    # centered pixels, so the threshold is in pixels too
    cols = _abspose_columns(x - PP, X)

    def error(m):
        return max(rotation_error_deg(m[:9].reshape(3, 3), R),
                   float(np.linalg.norm(m[9:12] - t)),
                   abs(float(m[12]) - FOCAL) / FOCAL)

    return Case('absolute-focal', cols, cols, None, 2.0 ** 2, P4PFSolver,
                _solve_p4pf, focal_reprojection_score,
                LMAbsolutePoseFocalRefiner(), error)


def _fundamental_case(seed, n, noise, outliers):
    from fastpose.estimators.utils import normalize_points, point_columns
    from fastpose.refiners.fundamental import LMFundamentalRefiner
    from fastpose.scorers.sampson import sampson_score
    from fastpose.solvers.fundamental import SevenPointSolver, _solve_7pt

    rng = np.random.default_rng(seed)
    x1, x2 = generate_data(rng, n, noise_sigma=noise, outlier_ratio=outliers,
                           image_size=IMAGE_SIZE)
    x1n, x2n, _, scale = normalize_points(x1, x2)
    cols = point_columns(x1n, x2n)

    def error(m):
        # there is no pose to compare here; on a noiseless scene the recovered
        # F must satisfy the epipolar constraint on every correspondence
        return max_algebraic_residual(m, x1n, x2n)

    return Case('fundamental', cols, cols, None, (2.0 * scale) ** 2,
                SevenPointSolver, _solve_7pt, sampson_score,
                LMFundamentalRefiner(), error)


def _homography_case(seed, n, noise, outliers):
    from fastpose.estimators.utils import normalize_points, point_columns
    from fastpose.refiners.homography import LMHomographyRefiner
    from fastpose.scorers.transfer import symmetric_transfer_score
    from fastpose.solvers.homography import FourPointSolver, _solve_h4p

    rng = np.random.default_rng(seed)
    x1, x2 = generate_homography_data(rng, n, noise_sigma=noise,
                                      outlier_ratio=outliers, focal=FOCAL,
                                      image_size=IMAGE_SIZE)
    x1n, x2n, _, scale = normalize_points(x1, x2)
    cols = point_columns(x1n, x2n)

    def error(m):
        # there is no pose to compare here; on a noiseless scene the recovered
        # H must transfer every correspondence onto its match, both ways
        return max_transfer_residual(m, x1n, x2n)

    return Case('homography', cols, cols, None, (2.0 * scale) ** 2,
                FourPointSolver, _solve_h4p, symmetric_transfer_score,
                LMHomographyRefiner(), error)


def _varying_focal_case(seed, n, noise, outliers):
    from fastpose.estimators.utils import normalize_points, point_columns
    from fastpose.refiners.utils import LO_INLIER_SCALE
    from fastpose.refiners.varying_focal import LMVaryingFocalPoseRefiner
    from fastpose.scorers.sampson import varying_focal_pose_sampson_score
    from fastpose.solvers.varying_focal import (SevenPointVaryingFocalSolver,
                                                _solve_varying_focal_7pt)

    rng = np.random.default_rng(seed)
    x1, x2, R, t, pp1, pp2 = generate_varying_focal_relpose_data(
        rng, n, noise_sigma=noise, outlier_ratio=outliers, focal1=VF_FOCAL1,
        focal2=VF_FOCAL2, pp1=VF_PP1, pp2=VF_PP2)
    x1n, x2n, T, scale = normalize_points(x1, x2)
    pp1n = np.array([scale * pp1[0] + T[0, 2], scale * pp1[1] + T[1, 2]])
    pp2n = np.array([scale * pp2[0] + T[0, 2], scale * pp2[1] + T[1, 2]])
    cols = point_columns(x1n, x2n)
    # the CPU solver, scorer and refiner all take the principal points inside
    # the data tuple; on the GPU they travel in `params`
    data = cols + (float(pp1n[0]), float(pp1n[1]),
                   float(pp2n[0]), float(pp2n[1]))
    params = np.array([pp1n[0], pp1n[1], pp2n[0], pp2n[1]])

    def error(m):
        return max(rotation_error_deg(m[:9].reshape(3, 3), R),
                   translation_error_deg(m[9:12], t),
                   abs(float(m[12]) / scale - VF_FOCAL1) / VF_FOCAL1,
                   abs(float(m[13]) / scale - VF_FOCAL2) / VF_FOCAL2)

    return Case('varying-focal', data, cols, params, (2.0 * scale) ** 2,
                SevenPointVaryingFocalSolver, _solve_varying_focal_7pt,
                varying_focal_pose_sampson_score,
                LMVaryingFocalPoseRefiner(
                    relaxed_inlier_scale=LO_INLIER_SCALE),
                error)


def _shared_focal_case(seed, n, noise, outliers):
    from fastpose.estimators.utils import normalize_points, point_columns
    from fastpose.refiners.shared_focal import LMSharedFocalPoseRefiner
    from fastpose.refiners.utils import LO_INLIER_SCALE
    from fastpose.scorers.sampson import shared_focal_pose_sampson_score
    from fastpose.solvers.shared_focal import (SixPointSharedFocalSolver,
                                               _solve_shared_focal_6pt)

    rng = np.random.default_rng(seed)
    x1, x2, R, t, pp1, pp2 = generate_shared_focal_relpose_data(
        rng, n, noise_sigma=noise, outlier_ratio=outliers, focal=FOCAL,
        pp1=VF_PP1, pp2=VF_PP2)
    # the 31x46 elimination template is far too ill-conditioned for raw pixel
    # coordinates - the estimator normalizes first and so must this
    x1n, x2n, T, scale = normalize_points(x1, x2)
    pp1n = np.array([scale * pp1[0] + T[0, 2], scale * pp1[1] + T[1, 2]])
    pp2n = np.array([scale * pp2[0] + T[0, 2], scale * pp2[1] + T[1, 2]])
    cols = point_columns(x1n, x2n)
    data = cols + (float(pp1n[0]), float(pp1n[1]),
                   float(pp2n[0]), float(pp2n[1]))
    params = np.array([pp1n[0], pp1n[1], pp2n[0], pp2n[1]])

    def error(m):
        # the model carries the one focal twice; both must come back
        return max(rotation_error_deg(m[:9].reshape(3, 3), R),
                   translation_error_deg(m[9:12], t),
                   abs(float(m[12]) / scale - FOCAL) / FOCAL,
                   abs(float(m[13]) / scale - FOCAL) / FOCAL)

    return Case('shared-focal', data, cols, params, (2.0 * scale) ** 2,
                SixPointSharedFocalSolver, _solve_shared_focal_6pt,
                shared_focal_pose_sampson_score,
                LMSharedFocalPoseRefiner(
                    relaxed_inlier_scale=LO_INLIER_SCALE),
                error, check_solver_models=_solver_models_shared_focal)


def _monodepth_scene(seed, n, noise, outliers, focal, shift):
    # alpha1 = 1 keeps the recovered translation equal to the ground truth
    # one: the model solves scale (d2 + shift2) x2h = R (d1 + shift1) x1h + t,
    # so t comes back scaled by alpha1. The affine depth corruption then gives
    # ground truth scale = alpha1/alpha2 and shift_i = beta_i/alpha_i.
    # `noise` is in pixels, as it is for every other case here; the calibrated
    # problems work in calibrated units, so it is divided by the focal there
    f = FOCAL if focal else 1.0
    x1, x2, d1, d2, R, t = generate_monodepth_relpose_data(
        np.random.default_rng(seed), n,
        noise_sigma=noise if focal else noise / FOCAL,
        outlier_ratio=outliers, depth_noise=0.01 if noise else 0.0,
        focal1=f, focal2=f, alpha1=1.0, beta1=0.05 if shift else 0.0,
        alpha2=0.8, beta2=-0.03 if shift else 0.0)
    return x1, x2, d1, d2, R, t


def _monodepth_case(name, num_tangent, focal, shift, solver, cpu_solve,
                    refiner_cls):
    from fastpose.refiners.monodepth import (monodepth_focal_hybrid_cost,
                                             monodepth_hybrid_cost)
    from fastpose.scorers.sampson import (monodepth_focal_pose_sampson_score,
                                          monodepth_pose_sampson_score)

    max_error = 2.0 if focal else 0.002
    max_reproj_error = 16.0 if focal else 0.016
    scale_reproj = (max_error / max_reproj_error) ** 2

    def build(seed, n, noise, outliers):
        x1, x2, d1, d2, R, t = _monodepth_scene(seed, n, noise, outliers,
                                                focal, shift)
        cols = (np.ascontiguousarray(x1[:, 0]), np.ascontiguousarray(x1[:, 1]),
                np.ascontiguousarray(x2[:, 0]), np.ascontiguousarray(x2[:, 1]),
                np.ascontiguousarray(d1), np.ascontiguousarray(d2))
        # the CPU tuple carries the two hybrid weights; on the GPU they ride
        # in `params` and, from there, in the derived form
        data = cols + (scale_reproj, 1.0)
        params = np.array([scale_reproj, 1.0])

        def error(m):
            errs = [rotation_error_deg(m[:9].reshape(3, 3), R),
                    float(np.linalg.norm(m[9:12] - t))]
            if focal:
                errs.append(abs(float(m[12]) - FOCAL) / FOCAL)
                errs.append(abs(float(m[13]) - FOCAL) / FOCAL)
                errs.append(abs(float(m[14]) - 1.25) / 1.25)
            else:
                errs.append(abs(float(m[12]) - 1.25) / 1.25)
                if shift:
                    # the model needs d + shift = alpha z, so the shift that
                    # undoes `d = alpha z + beta` is -beta, not beta/alpha
                    errs.append(abs(float(m[13]) + 0.05))
                    errs.append(abs(float(m[14]) - 0.03))
            return max(errs)

        return Case(name, data, cols, params, max_error ** 2, solver,
                    cpu_solve, (monodepth_focal_pose_sampson_score if focal
                                else monodepth_pose_sampson_score),
                    refiner_cls(), error,
                    lm_cost=(monodepth_focal_hybrid_cost if focal
                             else monodepth_hybrid_cost))

    return build


def _make_monodepth_builders():
    from fastpose.refiners.monodepth import (
        LMMonoDepthPoseRefiner, LMMonoDepthSharedFocalPoseRefiner,
        LMMonoDepthShiftPoseRefiner, LMMonoDepthVaryingFocalPoseRefiner)
    from fastpose.solvers.monodepth import (
        MonoDepthP3PSolver, MonoDepthSharedFocalSolver, MonoDepthShiftSolver,
        MonoDepthVaryingFocalSolver, _solve_monodepth_p3p,
        _solve_monodepth_shared_focal_3pt, _solve_monodepth_shift_3pt,
        _solve_monodepth_varying_focal_3pt)

    return {
        'monodepth': _monodepth_case(
            'monodepth', 7, False, False, MonoDepthP3PSolver,
            _solve_monodepth_p3p, LMMonoDepthPoseRefiner),
        'monodepth-shift': _monodepth_case(
            'monodepth-shift', 9, False, True, MonoDepthShiftSolver,
            _solve_monodepth_shift_3pt, LMMonoDepthShiftPoseRefiner),
        'monodepth-shared-focal': _monodepth_case(
            'monodepth-shared-focal', 8, True, False,
            MonoDepthSharedFocalSolver, _solve_monodepth_shared_focal_3pt,
            LMMonoDepthSharedFocalPoseRefiner),
        'monodepth-varying-focal': _monodepth_case(
            'monodepth-varying-focal', 9, True, False,
            MonoDepthVaryingFocalSolver, _solve_monodepth_varying_focal_3pt,
            LMMonoDepthVaryingFocalPoseRefiner),
    }


BUILDERS = {
    'essential': _essential_case,
    'absolute': _absolute_case,
    'absolute-focal': _absolute_focal_case,
    'fundamental': _fundamental_case,
    'homography': _homography_case,
    'varying-focal': _varying_focal_case,
    'shared-focal': _shared_focal_case,
}
BUILDERS.update(_make_monodepth_builders())

PROBLEMS = tuple(BUILDERS)


def make_case(name, seed, num_samples, noise_sigma=0.5, outlier_ratio=0.3):
    return BUILDERS[name](seed, num_samples, noise_sigma, outlier_ratio)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def draw_samples(rng, n, count, sample_size):
    return np.stack([rng.choice(n, sample_size, replace=False)
                     for _ in range(count)]).astype(np.int64)


def device_columns(case):
    from numba import cuda
    d64 = tuple(cuda.to_device(c) for c in case.cols)
    # the scorer and the LM are compiled with real=float32; handing them the
    # float64 columns would promote their whole per-point chain back to double
    d32 = tuple(cuda.to_device(c.astype(np.float32)) for c in case.cols)
    return d64, d32


def device_params(problem, case):
    from numba import cuda
    host = (problem.default_params if case.params is None else
            np.ascontiguousarray(case.params, dtype=np.float64))
    return cuda.to_device(host)


def cpu_solve_all(case, samples):
    models = np.zeros((len(samples), case.max_models, case.num_params))
    counts = np.zeros(len(samples), dtype=np.int64)
    workspace = np.empty(case.workspace_size)
    for i in range(len(samples)):
        m = np.zeros((case.max_models, case.num_params))
        counts[i] = case.cpu_solve(case.data, samples[i], m, workspace)
        models[i] = m
    return models, counts


def gpu_solve_all(problem, case, d_data, d_params, samples):
    from numba import cuda
    d_models, d_counts = problem.allocate_models(len(samples))
    problem.solve_batch(d_data, d_params, cuda.to_device(samples), d_models,
                        d_counts)
    cuda.synchronize()
    return d_models, d_counts


def cpu_candidates(case, seed, wanted):
    # minimal models to hand the LM, drawn the way the driver would
    rng = np.random.default_rng(seed)
    workspace = np.empty(case.workspace_size)
    out = []
    attempts = 0
    while len(out) < wanted and attempts < 400:
        attempts += 1
        sample = rng.choice(case.num_points, case.sample_size,
                            replace=False).astype(np.int64)
        m = np.zeros((case.max_models, case.num_params))
        for k in range(case.cpu_solve(case.data, sample, m, workspace)):
            if len(out) < wanted:
                out.append(m[k].copy())
    assert out, f"{case.name}: the solver produced no candidates"
    return np.array(out)


# ---------------------------------------------------------------------------
# minimal solver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', PROBLEMS)
def test_cuda_solver_matches_cpu_solver(name):
    from fastpose.cuda.registry import get_problem

    case = make_case(name, 0, 500, noise_sigma=0.0, outlier_ratio=0.0)
    problem = get_problem(name)
    rng = np.random.default_rng(7)
    batch = 128
    samples = draw_samples(rng, case.num_points, batch, case.sample_size)

    cpu_models, cpu_counts = cpu_solve_all(case, samples)
    d_data, _ = device_columns(case)
    d_models, d_counts = gpu_solve_all(problem, case,
                                       d_data, device_params(problem, case),
                                       samples)
    gpu_models = d_models.copy_to_host()
    gpu_counts = d_counts.copy_to_host()

    # the solvers' branches are all on comparisons against tolerances well
    # above the codegen difference, so the *number* of models must agree -
    # this holds for every problem, shared-focal included, and is the check
    # that catches a structural bug
    np.testing.assert_array_equal(cpu_counts, gpu_counts)
    assert cpu_counts.sum() > 0
    cpu_flat = np.concatenate([cpu_models[i, :cpu_counts[i]]
                               for i in range(batch) if cpu_counts[i]])
    gpu_flat = np.concatenate([gpu_models[i, :cpu_counts[i]]
                               for i in range(batch) if cpu_counts[i]])
    case.check_solver_models(cpu_flat, gpu_flat)


@pytest.mark.parametrize('name', PROBLEMS)
def test_cuda_solver_is_exact_on_a_clean_scene(name):
    from fastpose.cuda.registry import get_problem

    case = make_case(name, 1, 200, noise_sigma=0.0, outlier_ratio=0.0)
    problem = get_problem(name)
    rng = np.random.default_rng(3)
    batch = 64
    samples = draw_samples(rng, case.num_points, batch, case.sample_size)

    d_data, _ = device_columns(case)
    d_models, d_counts = gpu_solve_all(problem, case,
                                       d_data, device_params(problem, case),
                                       samples)
    models = d_models.copy_to_host()
    counts = d_counts.copy_to_host()

    assert counts.sum() > 0
    best = min(case.error(models[i, m])
               for i in range(batch) for m in range(counts[i]))
    assert best < 1e-6, f"{name}: best model error {best}"


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', PROBLEMS)
def test_cuda_scorer_matches_cpu_scorer(name):
    from numba import cuda

    from fastpose.cuda.ransac import score_batch
    from fastpose.cuda.registry import get_problem
    from fastpose.cuda.scoring import ScoreBuffers

    case = make_case(name, 2, 3000)
    problem = get_problem(name)
    rng = np.random.default_rng(11)
    batch = 128
    samples = draw_samples(rng, case.num_points, batch, case.sample_size)

    d_data, d_data32 = device_columns(case)
    d_params = device_params(problem, case)
    d_models, d_counts = gpu_solve_all(problem, case, d_data, d_params,
                                       samples)
    models = d_models.copy_to_host()
    counts = d_counts.copy_to_host()

    buffers = ScoreBuffers(batch)
    score_batch(problem, d_data32, d_params, d_models, d_counts,
                case.max_error_sq, buffers, batch)
    cuda.synchronize()
    g_score, g_idx, g_inl, g_max_inl, _ = buffers.to_host(batch)

    checked = 0
    for i in range(batch):
        if counts[i] == 0:
            assert g_idx[i] == -1
            continue
        checked += 1
        # a 1e300 bound disables the CPU scorer's early bail-out, so both
        # sides score every correspondence
        scored = [case.cpu_score(models[i, m], case.data, case.max_error_sq,
                                 1e300)
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

def run_gpu_lm(problem, case, candidates, num_iterations):
    from numba import cuda

    from fastpose.cuda.lm import RefineBuffers
    from fastpose.cuda.ransac import refine_prepared
    from fastpose.refiners.losses import TruncatedLoss

    k = len(candidates)
    _, d_data32 = device_columns(case)
    d_params = device_params(problem, case)
    buffers = RefineBuffers(k, case.num_points, problem.num_params)
    buffers.init_models[:k].copy_to_device(
        np.ascontiguousarray(candidates, dtype=np.float64))
    refine_prepared(problem, d_data32, d_params, buffers, k,
                    case.max_error_sq, num_iterations, TruncatedLoss())
    cuda.synchronize()
    result = buffers.refined[:k].copy_to_host()
    refined = result[:, 0, :problem.num_params]
    ok = (result[:, 0, problem.num_params] != 0.0).astype(np.int64)
    return refined, ok


@pytest.mark.parametrize('name', PROBLEMS)
def test_cuda_lm_matches_cpu_lm(name):
    from fastpose.cuda.registry import get_problem

    case = make_case(name, 4, 3000)
    problem = get_problem(name)
    lo_iterations = 25
    candidates = cpu_candidates(case, 13, 16)
    k = len(candidates)

    cpu_refined = np.zeros((k, case.num_params))
    cpu_ok = np.zeros(k, dtype=np.int64)
    for i in range(k):
        out = np.empty(case.num_params)
        cpu_ok[i] = int(case.refiner.refine(case.data, candidates[i], out,
                                            case.max_error_sq, lo_iterations))
        if cpu_ok[i]:
            cpu_refined[i] = out

    gpu_refined, gpu_ok = run_gpu_lm(problem, case, candidates, lo_iterations)

    np.testing.assert_array_equal(cpu_ok, gpu_ok)
    assert cpu_ok.sum() > 0
    both = np.nonzero(cpu_ok == 1)[0]

    # Scored in full float64, the mixed-precision result must be as good a
    # minimizer as the float64 CPU refiner's - an element-wise tolerance
    # alone cannot tell "rounded differently" from "converged somewhere
    # worse". But "as good on every candidate" is not a true property of
    # these costs, and asserting it would be asserting something false:
    #
    # they are not convex, and a minimal sample can put the LM on a start
    # from which the descent is chaotic. Measured on monodepth-shared-focal
    # candidate 5, perturbing the *initial model* by 1e-7 relative and
    # re-running the **CPU** refiner alone gives ~25 distinct minima spanning
    # 3.2% in cost; candidates 3 and 8 hold a single basin to 4e-9. On such a
    # start float32-vs-float64 arithmetic simply picks a different minimum -
    # the GPU landed 0.7% *below* the CPU under numba 0.65 and 0.8% above it
    # under 0.67.
    #
    # So this asserts the three things that are true, each of which a genuine
    # jacobian or precision bug would break: nearly every candidate reaches
    # the same basin, the parameters match there, and there is no systematic
    # drift.
    cpu_costs = np.array([case.lm_cost(cpu_refined[i], case.data,
                                       case.max_error_sq, 1e300)[0]
                          for i in both])
    gpu_costs = np.array([case.lm_cost(gpu_refined[i], case.data,
                                       case.max_error_sq, 1e300)[0]
                          for i in both])
    rel = (gpu_costs - cpu_costs) / np.abs(cpu_costs)
    same = both[np.abs(rel) <= 1e-6]

    assert len(same) >= 0.75 * len(both), (
        f"{name}: only {len(same)}/{len(both)} candidates reached the same "
        f"minimum on both backends")
    assert np.median(rel) <= 1e-6, (
        f"{name}: the GPU refiner is systematically worse "
        f"(median relative cost {np.median(rel):g})")
    # the chaotic starts stay bounded: 5% is well outside the 3.2% spread the
    # CPU alone shows on one, and far inside anything a broken jacobian does
    assert rel.max() <= 0.05, (
        f"{name}: worst candidate converged {rel.max():.1%} worse on the GPU")

    # The GPU LM evaluates residuals and the jacobian in float32 (accumulating
    # JtJ/Jtr and solving the T x T system in float64), so the converged model
    # differs at float32 level rather than matching bit for bit.
    np.testing.assert_allclose(gpu_refined[same], cpu_refined[same],
                               rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize('name', PROBLEMS)
def test_cuda_lm_improves_the_cost_it_minimizes(name):
    from fastpose.cuda.registry import get_problem

    case = make_case(name, 5, 2000)
    problem = get_problem(name)
    candidates = cpu_candidates(case, 17, 8)
    refined, ok = run_gpu_lm(problem, case, candidates, 25)

    assert ok.sum() > 0
    improved = 0
    for i in range(len(candidates)):
        if not ok[i]:
            continue
        before, _ = case.lm_cost(candidates[i], case.data,
                                 case.max_error_sq, 1e300)
        after, _ = case.lm_cost(refined[i], case.data, case.max_error_sq,
                                1e300)
        assert after <= before * (1.0 + 1e-9)
        improved += int(after < before)
    assert improved > 0


# ---------------------------------------------------------------------------
# end-to-end, per problem
# ---------------------------------------------------------------------------

def _cuda_kwargs(iterations):
    return dict(iterations=iterations, min_iterations=iterations, seed=0)


def _estimate_pair(name, iterations=2000):
    # runs the problem's estimator entry point on both devices with matching
    # arguments; returns (cpu_result, gpu_result), each (model, info)
    rng = np.random.default_rng(21)
    kwargs = _cuda_kwargs(iterations)
    if name == 'essential':
        from fastpose.estimators.essential import estimate_relative_pose
        x1, x2, R, _ = generate_relpose_data(rng, 3000, noise_sigma=0.5,
                                             outlier_ratio=0.3, focal=FOCAL,
                                             image_size=IMAGE_SIZE)
        args = ((x1 - PP) / FOCAL, (x2 - PP) / FOCAL)
        kwargs['max_error'] = 2.0 / FOCAL
        fn, gt = estimate_relative_pose, R
    elif name == 'absolute':
        from fastpose.estimators.absolute import estimate_absolute_pose
        x, X, R, _ = generate_abspose_data(rng, 3000, noise_sigma=0.5,
                                           outlier_ratio=0.3, focal=FOCAL,
                                           image_size=IMAGE_SIZE)
        args = ((x - PP) / FOCAL, X)
        kwargs['max_error'] = 2.0 / FOCAL
        fn, gt = estimate_absolute_pose, R
    elif name == 'absolute-focal':
        from fastpose.estimators.absolute_focal import \
            estimate_absolute_pose_with_focal
        x, X, R, _ = generate_abspose_data(rng, 3000, noise_sigma=0.5,
                                           outlier_ratio=0.3, focal=FOCAL,
                                           image_size=IMAGE_SIZE)
        args = (x - PP, X)
        kwargs['max_error'] = 2.0
        fn, gt = estimate_absolute_pose_with_focal, R
    elif name == 'fundamental':
        from fastpose.estimators.fundamental import estimate_fundamental
        x1, x2 = generate_data(rng, 3000, noise_sigma=0.5, outlier_ratio=0.3,
                               image_size=IMAGE_SIZE)
        args = (x1, x2)
        kwargs['max_error'] = 2.0
        fn, gt = estimate_fundamental, None
    elif name == 'homography':
        from fastpose.estimators.homography import estimate_homography
        x1, x2 = generate_homography_data(rng, 3000, noise_sigma=0.5,
                                          outlier_ratio=0.3, focal=FOCAL,
                                          image_size=IMAGE_SIZE)
        args = (x1, x2)
        kwargs['max_error'] = 2.0
        fn, gt = estimate_homography, None
    elif name == 'varying-focal':
        from fastpose.estimators.varying_focal import \
            estimate_relative_pose_with_varying_focals
        x1, x2, R, _, pp1, pp2 = generate_varying_focal_relpose_data(
            rng, 3000, noise_sigma=0.5, outlier_ratio=0.3, focal1=VF_FOCAL1,
            focal2=VF_FOCAL2, pp1=VF_PP1, pp2=VF_PP2)
        args = (x1, x2, pp1, pp2)
        kwargs['max_error'] = 2.0
        fn, gt = estimate_relative_pose_with_varying_focals, R
    elif name == 'shared-focal':
        from fastpose.estimators.shared_focal import \
            estimate_relative_pose_with_shared_focal
        x1, x2, R, _, pp1, pp2 = generate_shared_focal_relpose_data(
            rng, 3000, noise_sigma=0.5, outlier_ratio=0.3, focal=FOCAL,
            pp1=VF_PP1, pp2=VF_PP2)
        args = (x1, x2, pp1, pp2)
        kwargs['max_error'] = 2.0
        fn, gt = estimate_relative_pose_with_shared_focal, R
    elif name.startswith('monodepth'):
        from fastpose.estimators.monodepth import (
            estimate_relative_pose_with_monodepth,
            estimate_shared_focal_relative_pose_with_monodepth,
            estimate_varying_focal_relative_pose_with_monodepth)
        focal = name.endswith('focal')
        x1, x2, d1, d2, R, _ = _monodepth_scene(
            31, 3000, 0.5, 0.3, focal, name == 'monodepth-shift')
        args = (x1, x2, d1, d2)
        kwargs['max_error'] = 2.0 if focal else 0.002
        gt = R
        if name == 'monodepth-shared-focal':
            fn = estimate_shared_focal_relative_pose_with_monodepth
        elif name == 'monodepth-varying-focal':
            fn = estimate_varying_focal_relative_pose_with_monodepth
        else:
            fn = estimate_relative_pose_with_monodepth
            kwargs['estimate_shift'] = name == 'monodepth-shift'
    else:
        raise AssertionError(name)

    cpu = fn(*args, device='cpu', **kwargs)
    gpu = fn(*args, device='cuda', **kwargs)
    return cpu, gpu, gt


@pytest.mark.parametrize('name', PROBLEMS)
def test_cuda_estimator_matches_cpu_estimator(name):
    (_, cpu_info), (gpu_model, gpu_info), R_gt = _estimate_pair(name)

    # the two drivers draw different samples and optimize different
    # candidates, so they are compared on the quality of what they return
    # rather than on identical output
    assert gpu_info['num_inliers'] >= cpu_info['num_inliers'] * 0.98
    assert gpu_info['inliers'].shape == (3000,)
    if R_gt is not None:
        assert rotation_error_deg(gpu_model['R'], R_gt) < 1.0


# ---------------------------------------------------------------------------
# end-to-end driver behaviour (problem-agnostic; exercised on `essential`)
# ---------------------------------------------------------------------------

def make_scene(seed, num_samples=2000, noise_sigma=0.5, outlier_ratio=0.3):
    rng = np.random.default_rng(seed)
    x1, x2, R, t = generate_relpose_data(
        rng, num_samples, noise_sigma=noise_sigma, outlier_ratio=outlier_ratio,
        focal=FOCAL, image_size=IMAGE_SIZE)
    return (x1 - PP) / FOCAL, (x2 - PP) / FOCAL, R, t


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

    def spy(problem, data32, params, buffers, num_candidates, *args, **kwargs):
        calls.append(num_candidates)
        return original(problem, data32, params, buffers, num_candidates,
                        *args, **kwargs)

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
    # and the gate must actually gate: far fewer launches than rounds, plus
    # the one call the final Cauchy polish makes
    assert len(calls) < 4000 // 512


def test_unknown_device_raises():
    from fastpose.estimators.essential import estimate_relative_pose

    x1, x2, _, _ = make_scene(0, num_samples=50)
    with pytest.raises(ValueError, match="device must be"):
        estimate_relative_pose(x1, x2, device='tpu')


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_registry_returns_one_problem_object_per_name():
    from fastpose.cuda.registry import get_problem, problem_names

    assert set(PROBLEMS) <= set(problem_names())
    for name in PROBLEMS:
        p = get_problem(name)
        assert p.name == name
        assert get_problem(name) is p


def test_registry_rejects_an_unknown_name():
    from fastpose.cuda.registry import get_problem

    with pytest.raises(ValueError, match="unknown CUDA problem"):
        get_problem('no-such-problem')


# ---------------------------------------------------------------------------
# warmup
# ---------------------------------------------------------------------------

def test_cuda_warmup_covers_every_ported_problem():
    # a kernel the warmup misses is one that every worker of a
    # multiprocessing pool compiles - and races to cache - at once, which is
    # how numba's on-disk cache index ends up pointing at another key's code
    from fastpose.estimators.warmup import _cuda_warmup_steps

    steps = _cuda_warmup_steps('all', 3, 1, 1)
    # one step per problem, plus the one that covers the monodepth polish
    # kernels the per-problem calls cannot reach (see the next test)
    assert ({label for label, _ in steps}
            == {f'{n}-cuda' for n in PROBLEMS}
            | {'monodepth-final-refiners-cuda'})


def test_cuda_warmup_reaches_the_final_polish_kernel():
    # the polish pass is a *second* LM kernel per problem, built for the
    # Cauchy loss. It is only reached when the estimate finds inliers, so a
    # warmup scene too small or too degenerate to find any would leave it
    # cold - silently.
    from fastpose.cuda.registry import get_problem
    from fastpose.estimators.warmup import _cuda_warmup_steps
    from fastpose.refiners.losses import CauchyLoss, TruncatedLoss

    for label, run in _cuda_warmup_steps('all', 3, 1, 1):
        run()
    for name in PROBLEMS:
        built = set(get_problem(name)._lm_kernels)
        assert TruncatedLoss in built, f"{name}: no local-optimization kernel"
        assert CauchyLoss in built, f"{name}: final polish kernel left cold"


def test_cuda_warmup_covers_every_selectable_monodepth_final_loss():
    # The monodepth entry points are the only ones that let the caller pick
    # the polish pass's loss, and `CudaProblem.lm_kernel` memoizes one kernel
    # per loss *type*. The per-problem estimate calls only reach the default
    # one, so asserting Cauchy alone (above) passed while
    # `final_loss='truncated_cauchy'` still compiled its kernel on the call -
    # the exact cache race the warmup exists to prevent. Its CPU counterpart
    # is test_warmup.py's monodepth check.
    from fastpose.cuda.registry import get_problem
    from fastpose.estimators.warmup import (MONODEPTH_FINAL_LOSSES,
                                            _cuda_warmup_steps)
    from fastpose.refiners.losses import LOSSES

    for label, run in _cuda_warmup_steps('monodepth', 3, 1, 1):
        run()
    expected = {LOSSES[name] for name in MONODEPTH_FINAL_LOSSES}
    for name in (n for n in PROBLEMS if n.startswith('monodepth')):
        missing = expected - set(get_problem(name)._lm_kernels)
        assert not missing, (
            "%s: warmup left these polish kernels cold: %s"
            % (name, sorted(c.__name__ for c in missing)))
