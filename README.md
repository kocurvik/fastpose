# fastpose

Proof of concept for a fast Python-based backend for robust relative pose estimation,
benchmarked against [PoseLib](https://github.com/PoseLib/PoseLib) (C++). Implements
fundamental matrix estimation (7-point solver) and calibrated relative pose /
essential matrix estimation (5-point solver) inside a shared LO-RANSAC engine with
truncated Sampson (MSAC) scoring.

## Architecture

The engine (`estimators/ransac.py`) is problem-agnostic LO-RANSAC compiled with
numba. A camera pose problem plugs in three components, each a thin class wrapping
a numba kernel with a fixed signature:

- **Solver** (`solvers/`) — minimal solver:
  `solve(data, sample, models, workspace) -> num_models`
  plus metadata (`sample_size`, `num_params`, `max_models`, `workspace_size`).
- **Scorer** (`scorers/`) — truncated robust score:
  `score(model, data, max_error_sq, best_score) -> (score, num_inliers)`.
- **Refiner** (`refiners/`, optional) — non-minimal refit for local optimization:
  `refine(data, model, refined, max_error_sq, num_iterations) -> success`.

`data` is an opaque tuple of contiguous arrays chosen by the problem (for F:
four coordinate columns, SIMD-friendly). The engine handles sampling, model
selection, local optimization, final refinement and adaptive termination, and is
specialized per problem by `RansacEstimator(solver, scorer, refiner)`.

### Layout

```
solvers/       minimal solvers: 7-point F, 5-point E, 7-point varying-focal
               and 6-point shared-focal relative pose, P3P and P4Pf absolute
               pose, four 3-point monodepth relative pose variants; shared
               helpers in solvers/utils.py
scorers/       robust scorers: truncated Sampson (MSAC) for flat 3x3 models,
               for pose models [R|t] (E = [t]_x R assembled on the fly), for
               varying/shared-focal pose models [R|t|f1|f2] and for the
               monodepth model layouts; truncated reprojection error for
               absolute pose models [R|t] and unknown-focal models [R|t|f]
refiners/      local optimization; refiners/lm.py is the generic LM engine,
               refiners/utils.py the shared factorization/jacobian machinery,
               refiners/{fundamental,essential,varying_focal,shared_focal,
               absolute,absolute_focal,monodepth}.py the per-problem kernels
estimators/    the RANSAC engine (estimators/ransac.py) and the full pipelines
               estimate_fundamental / estimate_relative_pose /
               estimate_relative_pose_with_varying_focals /
               estimate_relative_pose_with_shared_focal /
               estimate_absolute_pose / estimate_absolute_pose_with_focal /
               estimate_relative_pose_with_monodepth /
               estimate_{shared,varying}_focal_relative_pose_with_monodepth;
               shared helpers in estimators/utils.py
benchmarks/    benchmarks/estimators compares against poselib (mAA + runtime
               scaling); benchmarks/solvers evaluates solver accuracy on
               synthetic noise-free minimal samples; shared metrics, data
               generation and plotting in benchmarks/utils.py
tests/         finite-difference checks of the refiner jacobians (tests/jac)
               and end-to-end estimator tests on synthetic scenes
```

Adding a new problem (P3P, ...) means writing the three kernels and their wrapper —
the RANSAC loop is reused unchanged. The 5-point problem demonstrates the reuse:
it shares the Sampson scorer verbatim and only adds a solver and a refiner.

### Shared LM refiner engine

`refiners/lm.py` contains the Levenberg-Marquardt loop itself (damping schedule,
damped normal equations, accept/reject, convergence) once, compiled per problem
via closure specialization like the RANSAC driver. A refinement problem only
defines four small kernels over a flat state vector: `init_state` (model →
state), `state_to_model` (state → model), `accumulate` (residuals + jacobian →
normal equations) and `apply_step` (retraction). Both epipolar refiners share
the Sampson jacobian accumulation from `refiners/utils.py`; the fundamental
refiner optimizes the factorized state `F = U diag(1, sigma, 0) V^T` (7 tangent
parameters), the essential refiner optimizes the pose `[R | t]` directly
(5 tangent parameters: rotation + translation direction on the unit sphere).

## Fundamental matrix backend

- **7-point nullspace via Gaussian elimination** with partial pivoting instead of a
  LAPACK SVD call — much faster for a 7×9 matrix.
- **Closed-form cubic solver** (Cardano/trigonometric + one Newton polish step) for
  the determinant constraint instead of `np.roots`' companion-matrix eigendecomposition.
- **Fused single-pass MSAC scoring** with an exact early bail-out: the truncated score
  only grows, so scoring stops as soon as the partial score exceeds the best score so
  far (checked per 512-point chunk to keep the loop SIMD-vectorized). This is what
  keeps scoring cheap for dense matchers with many correspondences.
- **LO-RANSAC with Levenberg-Marquardt**: whenever a new best model is found, LM
  minimizes the truncated Sampson error over its inliers, with F parametrized by
  its SVD factorization `F = U diag(1, sigma, 0) V^T` (7 tangent parameters: two
  rotation updates + sigma), analogous to poselib's `FactorizedFundamentalMatrix`.
  Rank 2 holds by construction; a final refinement runs after the loop.
- **Hartley-style normalization** (single shared isotropic transform, threshold scaled
  accordingly) for numerical conditioning on pixel coordinates.
- Optional adaptive termination (`min_iterations < iterations`) with the standard
  inlier-ratio-based iteration bound.

## Benchmark

Synthetic pixel-scale correspondences (1000×1000 image, 1 px noise, 20% outliers),
1000 fixed RANSAC iterations, 2 px Sampson threshold, best of 5 runs
(Windows, Python 3.10, numba 0.65):

| matches | numpy   | numba    | numba+LO | poselib | speedup vs poselib |
|--------:|--------:|---------:|---------:|--------:|-------------------:|
|   1 000 | 0.252 s | 0.0050 s | 0.0053 s | 0.019 s | 3.5× |
|  10 000 | —       | 0.023 s  | 0.026 s  | 0.248 s | 9.5× |
|  50 000 | —       | 0.117 s  | 0.137 s  | 1.125 s | 8.2× |

Local optimization costs little (it only runs when a new best model is found)
and pays off on harder data — with 2 px noise and 40% outliers (2000 matches,
same seeds): plain RANSAC 638–1165 inliers, LO-RANSAC 896–1183, poselib
509–1025.

## 5-point relative pose backend

`solvers/essential.py` + `refiners/essential.py` add calibrated relative pose
estimation on the same engine:

- **Stewénius-style 5-point solver**: 4-dimensional nullspace by Gaussian
  elimination + Gram-Schmidt, the ten cubic constraints expanded into a 10×20
  matrix via precomputed monomial multiplication tables, Gauss-Jordan reduction
  and the 10×10 action matrix. Real eigenvalues are extracted the way poselib
  does it — characteristic polynomial via Danilevsky's method, Sturm-sequence
  bracketing with bisection + Newton polish — because LAPACK's nonsymmetric
  eigensolver is not available inside numba kernels.
- **Direct pose output with cheirality check**: each E candidate is decomposed
  inside the solver kernel into the (R, t) candidate with the best cheirality
  count on the minimal sample, so RANSAC models are poses `[R | t]` (12 flat
  parameters) and the estimator returns (R, t) directly — no post-hoc
  `motion_from_essential` on the winning model.
- **Pose Sampson scorer**: the same truncated Sampson form as for F, with
  `E = [t]_x R` assembled on the fly from the pose model.
- **LM refiner on the pose directly**: 5 tangent parameters — a minimal,
  gauge-free parametrization of the essential manifold — 3 for the rotation
  update `R exp([w]_x)` and 2 for the translation direction in an orthonormal
  basis of the plane orthogonal to t, retracted back to the unit sphere. Reuses
  the shared LM engine and Sampson jacobian accumulation from `refiners/`.
- `motion_from_essential` is kept for decomposing externally estimated E/F
  matrices (used by the fundamental matrix benchmark).

Benchmark (same protocol; 20% outliers, 1 px noise, f = 1000 px, 2 px threshold,
timing includes pose decomposition; poselib `estimate_relative_pose` with PINHOLE
cameras):

| matches | numba    | numba+LO | poselib | speedup vs poselib | rot err (LO / poselib) |
|--------:|---------:|---------:|--------:|-------------------:|-----------------------:|
|   1 000 | 0.023 s  | 0.024 s  | 0.027 s | 1.1× | 0.021° / 0.036° |
|  10 000 | 0.076 s  | 0.078 s  | 0.213 s | 2.7× | 0.008° / 0.015° |
|  50 000 | 0.302 s  | 0.342 s  | 1.178 s | 3.4× | 0.003° / 0.004° |

(The 5-point solver is ~10× more expensive per iteration than the 7-point one,
so the advantage at small n is smaller; scoring still dominates for dense
matches. Inlier counts are not directly comparable to poselib for this problem
since poselib additionally applies cheirality filtering during scoring.)

## Varying-focal relative pose backend

`estimate_relative_pose_with_varying_focals` estimates relative pose plus two
unknown focal lengths from pixel correspondences with known principal points,
on the same engine:

- **Minimal solver** (`solvers/varying_focal.py`): standard 7-point fundamental
  matrix hypotheses, focal lengths from the Bougnoux formula (square pixels,
  known principal points), then E = K2^T F K1 decomposed with the shared
  closed-form essential decomposition and cheirality check into pose models
  `[R | t | f1 | f2]` (14 flat parameters).
- **Scorer**: the truncated Sampson error of the induced fundamental matrix
  F = K2^-T E K1^-1, evaluated in the original pixel coordinates.
- **LM refiner** (`refiners/varying_focal.py`): 7 tangent parameters — rotation
  (3), translation direction (2) and the two log-focals — with the Sampson
  jacobian built from a central-difference tangent basis of the induced F.
- Hypotheses whose Bougnoux focal estimates are non-positive are rejected
  inside the solver.

## Absolute pose (P3P) backend

`estimate_absolute_pose` estimates the pose `[R | t]` (with
`lambda * (x, y, 1) = R X + t`) from calibrated 2D-3D correspondences:

- **P3P solver of Ding et al.** (`solvers/p3p.py`, "Revisiting the P3P
  Problem", CVPR 2023), ported from poselib's `p3p.cc`: the three
  correspondences reduce to a single cubic; the rank-2 conic pencil splits
  into two lines, each giving a quadratic in a depth ratio, and the depths
  are polished with a few Newton steps before the pose is assembled — no
  eigendecomposition anywhere.
- **Truncated reprojection scorer** (`scorers/reprojection.py`): fused MSAC
  scoring with the same exact per-chunk early bail-out as the Sampson
  scorers; the inlier test is division-free and points behind the camera
  count as outliers.
- **LM refiner on the pose directly** (`refiners/absolute.py`): 6 tangent
  parameters (rotation update `R exp([w]_x)` + translation) with an analytic
  reprojection jacobian, on the shared LM engine.

## Absolute pose with unknown focal (P4Pf) backend

`estimate_absolute_pose_with_focal` estimates `[R | t | f]` from 2D-3D
correspondences in pixel coordinates (square pixels, known principal point):

- **P4Pf solver** (`solvers/p4pf.py`), a port of poselib's `p4pf.cc` +
  `re3q3.cc`: the first two camera rows are parametrized by the 4-dimensional
  nullspace of the eight cross-product constraints — computed by Gaussian
  elimination + modified Gram-Schmidt like the 5-point nullspace (no LAPACK
  SVD) — the third row follows linearly, and the three rotation-orthogonality
  constraints form a 3Q3 system reduced to a degree-8 determinant polynomial
  solved with the shared Sturm machinery. Candidates are cheirality-checked
  on the sample, the solution closest to square pixels is kept, and R is
  snapped back onto SO(3) via a quaternion round-trip (as poselib does by
  storing poses as quaternions).
- **Truncated focal reprojection scorer** (`scorers/reprojection.py`): the
  `[R | t]` scorer with the focal folded into the projection, same
  division-free inlier test and per-chunk early bail-out.
- **LM refiner** (`refiners/absolute_focal.py`): 7 tangent parameters
  (rotation, translation, log-focal) with an analytic jacobian.

## Shared-focal relative pose backend

`estimate_relative_pose_with_shared_focal` estimates relative pose plus one
focal length shared by both cameras from pixel correspondences with known
principal points:

- **6-point minimal solver** (`solvers/shared_focal.py`), a port of
  poselib's `relpose_6pt_focal.cc`: 3-dimensional nullspace by Gaussian
  elimination + Gram-Schmidt, mixed by a fixed rotation because the
  pregenerated elimination template is only valid for a generic nullspace
  basis; the upstream 31x31 elimination template (solved as one augmented
  31x46 system), and the 15x15 action matrix whose real eigenvalues are
  extracted with the shared Danilevsky + Sturm machinery. Each solution
  gives F and the focal; E = K^T F K is decomposed with the shared
  closed-form essential decomposition and cheirality check into pose models
  `[R | t | f | f]` (14 flat parameters, varying-focal layout).
- **Scorer**: the varying-focal truncated Sampson scorer reused verbatim.
- **LM refiner** (`refiners/shared_focal.py`): 6 tangent parameters —
  rotation (3), translation direction (2) and one shared log-focal — with
  the Sampson jacobian built like the varying-focal refiner.

## Monodepth relative pose backends

Relative pose from correspondences plus per-image monocular depth
estimates (scale- or affine-invariant MDE outputs), following poselib's
monodepth estimators. The depth convention is poselib's
`MonoDepthTwoViewGeometry`: the 3D point of correspondence i is
`(d1_i + shift1) x1h_i` in camera 1 and `scale (d2_i + shift2) x2h_i` in
camera 2. Four 3-point minimal solvers (`solvers/monodepth.py`):

- **Calibrated without shift** (`estimate_relative_pose_with_monodepth`,
  scale-invariant depths): 3D points from the camera-1 depths, absolute
  pose of camera 2 via the shared P3P kernel, depth scale from the first
  correspondence. Models `[R | t | scale | 0 | 0]` with metric t.
- **Calibrated with shift** (`estimate_shift=True`, affine-invariant
  depths): port of poselib's `relpose_monodepth_3pt` — the problem reduces
  to a quartic in the camera-1 shift, polished with Gauss-Newton on the
  three pairwise distance constraints; also recovers one depth shift per
  image. Models `[R | t | scale | shift1 | shift2]`.
- **Shared focal** (`estimate_shared_focal_relative_pose_with_monodepth`):
  port of `relpose_monodepth_3pt_shared_focal` in centered pixel
  coordinates; the third camera-2 depth is an unknown found as a real
  eigenvalue of a 4x4 action matrix (Danilevsky + Sturm here). Models
  `[R | t | f | f | scale]`.
- **Varying focal** (`estimate_varying_focal_relative_pose_with_monodepth`):
  port of `relpose_monodepth_3pt_varying_focal` — a single 3x3 linear
  system in (1/f1^2, s^2/f2^2, s^2). Models `[R | t | f1 | f2 | scale]`.

All four assemble the pose from the two depth-induced point triplets
(`Y X^-1` like poselib, with the quaternion SO(3) snap) and share one data
layout: six columns (x1, y1, x2, y2, d1, d2) plus the two hybrid weights.

- **Scoring** is the plain truncated Sampson error (threshold `max_error`,
  poselib's `max_errors[1]`); the depth parameters do not enter the score.
- **Local optimization** (`refiners/monodepth.py`) minimizes the hybrid
  cost of poselib's `MonoDepth*RelPoseRefiner`s: truncated Sampson
  (weighted by `weight_sampson`) plus the truncated symmetric reprojection
  error through the monodepth 3D points, scaled by
  `scale_reproj = max_error^2 / max_reproj_error^2` so both terms truncate
  at the same threshold (defaults 2 px Sampson / 16 px reprojection).
  Analytic jacobians, verified against finite differences in the tests.
  Tangent parameters: rotation (3), full translation (3, the depths fix the
  scale), plus additive scale / shifts / focals per problem (7, 9, 8 and 9
  parameters).

One known gap vs poselib: their python bindings run one extra final bundle
over the inlier set with a truncated *Cauchy* loss after RANSAC, which on
a small fraction of scenes escapes shallow local minima that the hard
truncated loss cannot (zero gradient outside the threshold). On synthetic
scenes this costs a few hundredths of mAA at a 15-40x runtime advantage
(see `benchmarks/estimators/monodepth.py`).

## Install

Install from the repository root:

```
pip install .
```

Numba compiles kernels lazily on first use. To populate the local Numba cache
up front, run:

```
fastpose-warmup
```

The warmup command runs small synthetic estimations for every backend. Use
`fastpose-warmup --problem fundamental` / `essential` / `absolute` /
`absolute-focal` / `varying-focal` / `shared-focal` / `monodepth` to warm
up only one backend (`monodepth` covers all four monodepth variants).

Run with (from the repository root):

```
python -m benchmarks.estimators.fundamental            # mAA vs runtime plot
python -m benchmarks.estimators.essential
python -m benchmarks.estimators.absolute
python -m benchmarks.estimators.fundamental scaling    # runtime scaling table
python -m benchmarks.estimators.essential scaling
python -m benchmarks.estimators.absolute scaling
python -m benchmarks.estimators.absolute focal         # P4Pf mAA plot
python -m benchmarks.estimators.absolute focal-scaling
python -m benchmarks.estimators.fundamental varying-focal   # varying-focal mAA plot
python -m benchmarks.estimators.fundamental shared-focal    # shared-focal mAA plot

python -m benchmarks.estimators.monodepth               # calibrated monodepth mAA plot
python -m benchmarks.estimators.monodepth shared-focal
python -m benchmarks.estimators.monodepth varying-focal
python -m benchmarks.estimators.monodepth scaling       # runtime scaling table

python -m benchmarks.solvers.fundamental               # noise-free minimal-sample accuracy
python -m benchmarks.solvers.essential

python -m pytest tests                                 # jacobian + estimator tests
```

(First call JIT-compiles the driver for a few seconds; the benchmarks warm up
before timing.)

## Next steps

- Degeneracy handling (e.g. dominant-plane checks à la DEGENSAC).
