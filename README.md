# fastpose

A fast Python backend for robust camera pose estimation inspired by [PoseLib](https://github.com/PoseLib/PoseLib). 
This package was mostly vibe-coded using Claude Code.
Fundamental/essential
matrix, calibrated and uncalibrated relative pose (including shared- and
varying-focal variants), absolute pose (P3P/P4Pf), and monocular-depth-assisted
relative pose — all built on one [numba](https://numba.pydata.org/)-compiled
LO-RANSAC engine and benchmarked against PoseLib (C++).

See [TESTS.md](TESTS.md) for the tests and for the synthetic benchmarks that
compare against PoseLib. Real-data numbers are coming later.

## Install

From the repository root:

```
pip install .
```

Dependencies are just `numpy` and `numba`.

Numba compiles kernels lazily on first use, which makes the very first call to
each `estimate_*` function slow (a few seconds). To compile everything up
front, run:

```
fastpose-warmup
```

Use `fastpose-warmup --problem fundamental` / `essential` / `absolute` /
`absolute-focal` / `varying-focal` / `shared-focal` / `monodepth` to warm up
only one backend (`monodepth` covers all four monodepth variants). Note that the warmup may take several minutes.

### Multiprocessing

**Warm the cache before you fork.** Numba's on-disk kernel cache is not safe
against concurrent writers, and two worker processes JIT-compiling the same
kernel at once can corrupt it. Always compile in the parent first:

```python
from fastpose.estimators.warmup import warmup

warmup()                                # or run `fastpose-warmup` beforehand
with multiprocessing.Pool(n) as pool:   # fork children inherit the compiled
    ...                                 # kernels and never touch the cache
```

With the default `fork` start method the children inherit the compiled
dispatchers in memory and never read or write the cache at all, so this is
usually all you need. Under `spawn`, workers re-import and load from the
cache, which is safe as long as the warmup covered everything they use. If you
cannot warm up first, give each worker its own `NUMBA_CACHE_DIR` instead.

## Usage

Every `estimate_*` function takes correspondence arrays plus RANSAC options
(`iterations`, `max_error`, `min_iterations` for adaptive termination,
`lo_iterations` for the local-optimization budget, `seed`, and `num_threads`
for [threading](#threading)) and returns `(model, info)`:

- `model` is a dict holding the estimated quantities — `R`/`t` for pose
  problems, `F` for the fundamental matrix, plus `f`/`f1`/`f2` for the
  unknown-focal variants and `scale`/`shift1`/`shift2` for the monodepth ones.
  On total failure (`info['num_inliers'] == 0`) `model` holds a generic
  identity/placeholder result instead of a fit.
- `info` is a dict with `inliers` (boolean mask), `num_inliers`,
  `model_score` (the scorer's truncated score), `iterations` (RANSAC samples
  drawn) and `refinements` (whether the post-RANSAC polish pass ran and was
  adopted).

```python
import numpy as np
from fastpose.estimators import estimate_fundamental, estimate_relative_pose, estimate_absolute_pose

# fundamental matrix (7-point): x1, x2 are pixel coordinates
model, info = estimate_fundamental(x1, x2, iterations=1000, max_error=2.0)
F = model['F']

# calibrated relative pose (5-point): x1, x2 are normalized (x - c) / f,
# or pass camera1/camera2 (3x3 K, or a poselib.Camera) to pass pixel points
# directly and have them unprojected for you
model, info = estimate_relative_pose(x1, x2, camera1=K1, camera2=K2,
                                      iterations=1000, max_error=2.0)
R, t = model['R'], model['t']

# absolute pose (P3P): x is normalized image points, X the 3D points
model, info = estimate_absolute_pose(x, X, iterations=1000, max_error=2.0 / focal)
R, t = model['R'], model['t']
```

### API reference

All functions live under `fastpose.estimators` and are re-exported from
`fastpose`. `min pts` is the minimal-sample size — calling with fewer
correspondences raises `ValueError`.

| Function | Input | `model` keys | min pts |
|---|---|---|---|
| `estimate_fundamental` | `x1, x2` pixel coords | `F` | 7 |
| `estimate_relative_pose` | `x1, x2` calibrated (or pixel + `camera1`/`camera2`) | `R, t` | 5 |
| `estimate_relative_pose_with_varying_focals` | `x1, x2` pixel coords, optional `principal_point1`/`2` | `R, t, f1, f2` | 7 |
| `estimate_relative_pose_with_shared_focal` | `x1, x2` pixel coords, optional `principal_point1`/`2` | `R, t, f` | 6 |
| `estimate_absolute_pose` | `x` calibrated (or pixel + `camera`), `X` 3D points | `R, t` | 3 |
| `estimate_absolute_pose_with_focal` | `x` pixel coords, `X` 3D points, optional `principal_point` | `R, t, f` | 4 |
| `estimate_relative_pose_with_monodepth` | `x1, x2` calibrated, `d1, d2` per-point depths | `R, t, scale, shift1, shift2` | 3 |
| `estimate_shared_focal_relative_pose_with_monodepth` | `x1, x2` pixel coords, `d1, d2` depths, optional `principal_point1`/`2` | `R, t, f, scale` | 3 |
| `estimate_varying_focal_relative_pose_with_monodepth` | `x1, x2` pixel coords, `d1, d2` depths, optional `principal_point1`/`2` | `R, t, f1, f2, scale` | 3 |

The `# params:` comment at the top of each function in
`src/fastpose/estimators/` documents every argument and exact input units
(several take normalized/calibrated points rather than pixels — see the
docstrings if you're not passing a `camera`). Every relative/absolute-pose
function also accepts a `final_refinement_iterations` argument: an LM step
budget for a final robust-loss polish pass over the RANSAC inliers (defaults
to 100 iterations; 0 disables it), independent of `lo_iterations`. 
The uncalibrated solvers assume principals at origina if they are not provided as optional params.

### Threading

Every `estimate_*` function also takes `num_threads` and `batch_per_thread`.
The default (`num_threads=None`, or `1`) runs the single-threaded RANSAC
driver. Anything higher switches to a batched parallel driver: hypotheses are
drawn `num_threads * batch_per_thread` at a time and solved and scored across
that many numba threads. `batch_per_thread` defaults to 32.

```python
model, info = estimate_relative_pose(x1, x2, camera1=K1, camera2=K2,
                                     iterations=1000, max_error=2.0,
                                     num_threads=8)
```

Measured on 8 physical cores at a 1000-iteration budget: **1.7–2.8×** end to
end, the smaller figure at large point counts. The batched loop itself scales
about 3.5–4×; local optimization and the final refinement stay single-threaded
and cap the rest, and they cost more as the inlier count grows.

Two caveats. This buys latency on one call, not throughput — if you already
run one process per core (see [Multiprocessing](#multiprocessing)), threading
inside each call only oversubscribes, so leave it off. And the parallel result
is close to but not bit-identical to the serial one: every hypothesis in a
batch is scored against the incumbent as it stood when the batch started, so
models the serial driver would bail out of early can get scored in full. A run
*is* reproducible from `(seed, num_threads * batch_per_thread)` regardless of
how many threads actually run, and adaptive termination overshoots by at most
one batch.

`motion_from_essential` (also exported from `fastpose.estimators`) decomposes
an externally estimated essential matrix into `(R, t)` candidates, for cases
where you already have `E`/`F` from elsewhere.

## Architecture

The engine (`src/fastpose/estimators/ransac.py`) is a problem-agnostic
LO-RANSAC compiled with numba. A camera pose problem plugs in three
components, each a thin class wrapping a numba kernel with a fixed signature:

- **Solver** (`src/fastpose/solvers/`) — minimal solver:
  `solve(data, sample, models, workspace) -> num_models`
  plus metadata (`sample_size`, `num_params`, `max_models`, `workspace_size`).
- **Scorer** (`src/fastpose/scorers/`) — truncated robust score:
  `score(model, data, max_error_sq, best_score) -> (score, num_inliers)`.
- **Refiner** (`src/fastpose/refiners/`, optional) — non-minimal refit for local
  optimization: `refine(data, model, refined, max_error_sq, num_iterations) -> success`.

`data` is an opaque tuple of contiguous arrays chosen by the problem (for F:
four coordinate columns, SIMD-friendly). The engine handles sampling, model
selection, local optimization, final refinement and adaptive termination, and
is specialized per problem by `RansacEstimator(solver, scorer, refiner)`.
Adding a new problem means writing the three kernels and their wrapper — the
RANSAC loop is reused unchanged; the 5-point problem shares the Sampson
scorer verbatim with the 7-point one and only adds a solver and a refiner.

`src/fastpose/refiners/lm.py` contains a shared Levenberg-Marquardt loop
(damping schedule, damped normal equations, accept/reject, convergence),
compiled per problem via the same closure specialization. A refinement
problem only defines four small kernels over a flat state vector:
`init_state`, `state_to_model`, `accumulate` (residuals + jacobian -> normal
equations) and `apply_step` (retraction).

### Layout

```
src/fastpose/
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
    kernel_cache.py  process-independent numba cache keys for the closure-
                   specialized kernels (what makes `fastpose-warmup` stick)
    estimators/    the RANSAC engine (estimators/ransac.py) and the full pipelines
                   listed in the API reference above; shared helpers in
                   estimators/utils.py
benchmarks/    benchmarks/estimators compares against poselib (mAA + runtime
               scaling); benchmarks/solvers evaluates solver accuracy on
               synthetic noise-free minimal samples; shared metrics, data
               generation and plotting in benchmarks/utils.py
tests/         finite-difference checks of the refiner jacobians (tests/jac)
               and end-to-end estimator tests on synthetic scenes
```

All four subpackages (`solvers`, `scorers`, `refiners`, `estimators`) are only
reachable through `fastpose` (e.g. `from fastpose.solvers import p3p`) —
`import solvers` / `import estimators` etc. directly is not supported.

## Backend notes

Technical detail on each backend's solver/scorer/refiner, for anyone
extending fastpose or curious how a specific number is computed.

### `fastmath`

`fastmath=True` goes on the kernels where reassociation and FMA contraction
measurably pay — the minimal solvers, the per-point scoring loops and the LM
accumulate kernels — and stays off the O(1) work around them: the per-model
and per-LM-iteration setup in `refiners/utils.py`, the state retractions and
the RANSAC driver itself. Numba's fast-math flags are per-function and do not
propagate into a separately compiled callee, so flagging a thin wrapper only
affects its own handful of scalar operations. Kernels declared
`inline='always'` need no flag either: numba inlines them at the Numba IR
level, so they are lowered with their caller's setting.

Where it pays it pays well — the monodepth LM engine is ~40% slower without
it, the 7-point solver ~11% — and where it does not, the flag is left off
rather than applied for uniformity. Enabling it on the RANSAC driver, for
instance, measures 0% and bit-identical results, because all of that loop's
arithmetic lives in the solver, scorer and refiner it calls; it would only
add `nnan` to the score comparisons, which every kernel here is written to
avoid needing (`1e300` is the failure sentinel throughout, never `inf`).

The one deliberate removal is the shared-focal solver, below.

<details>
<summary>Fundamental matrix (7-point)</summary>

- **7-point nullspace via Gaussian elimination** with partial pivoting instead
  of a LAPACK SVD call — much faster for a 7x9 matrix.
- **Closed-form cubic solver** (Cardano/trigonometric + one Newton polish
  step) for the determinant constraint instead of `np.roots`'
  companion-matrix eigendecomposition.
- **Fused single-pass MSAC scoring** with an exact early bail-out: the
  truncated score only grows, so scoring stops as soon as the partial score
  exceeds the best score so far (checked per 512-point block). Each full
  block runs with a compile-time-constant trip count so LLVM SIMD-vectorizes
  the body; a loop bounded by a runtime `min(start + chunk, n)` does not
  vectorize and costs about 2x here.
- **LO-RANSAC with Levenberg-Marquardt**: whenever a new best model is found,
  LM minimizes the truncated Sampson error over its inliers, with F
  parametrized by its SVD factorization `F = U diag(1, sigma, 0) V^T` (7
  tangent parameters: two rotation updates + sigma), analogous to poselib's
  `FactorizedFundamentalMatrix`. Rank 2 holds by construction; a final
  refinement runs after the loop.
- **Hartley-style normalization** (single shared isotropic transform,
  threshold scaled accordingly) for numerical conditioning on pixel
  coordinates.
- Optional adaptive termination (`min_iterations < iterations`) with the
  standard inlier-ratio-based iteration bound.

</details>

<details>
<summary>Calibrated relative pose (5-point)</summary>

`src/fastpose/solvers/essential.py` + `src/fastpose/refiners/essential.py`:

- **Stewenius-style 5-point solver**: 4-dimensional nullspace by Gaussian
  elimination + Gram-Schmidt, the ten cubic constraints expanded into a
  10x20 matrix via precomputed monomial multiplication tables, Gauss-Jordan
  reduction and the 10x10 action matrix. Real eigenvalues are extracted the
  way poselib does it — characteristic polynomial via Danilevsky's method,
  Sturm-sequence bracketing with bisection + Newton polish — because
  LAPACK's nonsymmetric eigensolver is not available inside numba kernels.
- **Direct pose output with cheirality check**: each E candidate is decomposed
  inside the solver kernel into the (R, t) candidates that put *every*
  minimal-sample point in front of both cameras, so RANSAC models are poses
  `[R | t]` (12 flat parameters) and the estimator returns (R, t) directly —
  no post-hoc `motion_from_essential` on the winning model. All consistent
  candidates are emitted, as poselib's `motion_from_essential` does: the four
  share one E and so score identically under Sampson, which means a candidate
  dropped here is a hypothesis RANSAC never gets to try.
- **Pose Sampson scorer**: the same truncated Sampson form as for F, with
  `E = [t]_x R` assembled on the fly from the pose model, plus the per-point
  cheirality check (minimum depth 0.01, on unit rays) that poselib's
  `CameraPose` overloads of `compute_sampson_msac_score` and `get_inliers`
  apply. A point that passes the Sampson test but triangulates behind either
  camera is charged the truncation constant and not counted as an inlier.
  The matrix overloads poselib uses for the focal problems have no such
  check, so neither do the focal scorers here.
- **Local optimization over the relaxed inlier subset**: the LO refit runs
  over the points within 5x the squared threshold rather than the whole
  correspondence set, matching `RelativePoseEstimator::refine_model`. The
  truncated loss already zeroes the weight of everything past 1x, so the
  normal equations are unaffected — but the LM's accept/reject test is not,
  and letting far outliers contribute a constant the jacobian never models
  costs both accuracy and iterations.
- **LM refiner on the pose directly**: 5 tangent parameters — a minimal,
  gauge-free parametrization of the essential manifold — 3 for the rotation
  update `R exp([w]_x)` and 2 for the translation direction in an orthonormal
  basis of the plane orthogonal to t, retracted back to the unit sphere.
- `motion_from_essential` is kept for decomposing externally estimated E/F
  matrices (used by the fundamental matrix benchmark).

(The 5-point solver is ~10x more expensive per iteration than the 7-point
one, so its advantage at small n is smaller; scoring still dominates for
dense matches. Inlier counts still differ slightly from poselib's: the mask
returned here is recomputed after the final polish pass, where poselib
reports the one it extracted before it.)

</details>

<details>
<summary>Varying-focal relative pose</summary>

`estimate_relative_pose_with_varying_focals` estimates relative pose plus two
unknown focal lengths from pixel correspondences with known principal
points:

- **Minimal solver** (`src/fastpose/solvers/varying_focal.py`): standard
  7-point fundamental matrix hypotheses, focal lengths from Rybkin's
  closed-form formula (an SVD-free equivalent of the Bougnoux formula;
  square pixels, known principal points), then E = K2^T F K1
  decomposed with the shared closed-form essential decomposition and
  cheirality check into pose models `[R | t | f1 | f2]` (14 flat
  parameters) — every cheirality-consistent candidate, as in the calibrated
  solver above.
- **Scorer**: the truncated Sampson error of the induced fundamental matrix
  F = K2^-T E K1^-1, evaluated in the original pixel coordinates. No
  cheirality check: poselib scores an F here, through the matrix overload.
- **LM refiner** (`src/fastpose/refiners/varying_focal.py`): 7 tangent
  parameters — rotation (3), translation direction (2) and the two
  log-focals — with the Sampson jacobian built from a central-difference
  tangent basis of the induced F. Local optimization refits over the 5x
  relaxed inlier subset, as `VaryingFocalRelativePoseEstimator::refine_model`
  does.
- Hypotheses whose focal estimates are non-positive or non-finite are
  rejected inside the solver.

</details>

<details>
<summary>Shared-focal relative pose</summary>

`estimate_relative_pose_with_shared_focal` estimates relative pose plus one
focal length shared by both cameras from pixel correspondences with known
principal points:

- **6-point minimal solver** (`src/fastpose/solvers/shared_focal.py`), a port
  of poselib's `relpose_6pt_focal.cc`: 3-dimensional nullspace by Gaussian
  elimination + Gram-Schmidt, mixed by a fixed rotation because the
  pregenerated elimination template is only valid for a generic nullspace
  basis; the upstream 31x31 elimination template (solved as one augmented
  31x46 system), and the 15x15 action matrix whose real eigenvalues are
  extracted with the shared Danilevsky + Sturm machinery. Each solution
  gives F and the focal; E = K^T F K is decomposed with the shared
  closed-form essential decomposition and cheirality check into pose models
  `[R | t | f | f]` (14 flat parameters, varying-focal layout) — every
  cheirality-consistent candidate, as in the calibrated solver above.
- **Scorer**: the varying-focal truncated Sampson scorer reused verbatim (no
  cheirality check — poselib scores an F here too).
- **LM refiner** (`src/fastpose/refiners/shared_focal.py`): 6 tangent
  parameters — rotation (3), translation direction (2) and one shared
  log-focal — with the Sampson jacobian built like the varying-focal
  refiner. Local optimization refits over the 5x relaxed inlier subset, as
  `SharedFocalRelativePoseEstimator::refine_model` does.
- The solver is the one module where `fastmath` is deliberately switched
  **off** (see [`fastmath`](#fastmath) above): its
  action-matrix/characteristic-polynomial chain is ill-conditioned enough
  that allowing reassociation made the recovered focal depend on whether the
  kernels were freshly compiled or loaded from the numba cache (~1e-8
  relative, never enough to move an inlier count). Disabling it costs no
  measurable runtime and makes the solver bit-reproducible; the shared
  scorer and LM engine keep `fastmath`, where it does pay.

</details>

<details>
<summary>Absolute pose (P3P)</summary>

`estimate_absolute_pose` estimates the pose `[R | t]` (with
`lambda * (x, y, 1) = R X + t`) from calibrated 2D-3D correspondences:

- **P3P solver of Ding et al.** (`src/fastpose/solvers/p3p.py`, "Revisiting
  the P3P Problem", CVPR 2023), ported from poselib's `p3p.cc`: the three
  correspondences reduce to a single cubic; the rank-2 conic pencil splits
  into two lines, each giving a quadratic in a depth ratio, and the depths
  are polished with a few Newton steps before the pose is assembled — no
  eigendecomposition anywhere.
- **Truncated reprojection scorer** (`src/fastpose/scorers/reprojection.py`):
  fused MSAC scoring with the same exact per-block early bail-out as the
  Sampson scorers; the inlier test is division-free and points behind the
  camera count as outliers.
- **LM refiner on the pose directly** (`src/fastpose/refiners/absolute.py`):
  6 tangent parameters (rotation update `R exp([w]_x)` + translation) with an
  analytic reprojection jacobian, on the shared LM engine.

</details>

<details>
<summary>Absolute pose with unknown focal (P4Pf)</summary>

`estimate_absolute_pose_with_focal` estimates `[R | t | f]` from 2D-3D
correspondences in pixel coordinates (square pixels, known principal point):

- **P4Pf solver** (`src/fastpose/solvers/p4pf.py`), a port of poselib's
  `p4pf.cc` + `re3q3.cc`: the first two camera rows are parametrized by the
  4-dimensional nullspace of the eight cross-product constraints — computed
  by Gaussian elimination + modified Gram-Schmidt like the 5-point nullspace
  (no LAPACK SVD) — the third row follows linearly, and the three
  rotation-orthogonality constraints form a 3Q3 system reduced to a degree-8
  determinant polynomial solved with the shared Sturm machinery. Candidates
  are cheirality-checked on the sample, the solution closest to square
  pixels is kept, and R is snapped back onto SO(3) via a quaternion
  round-trip (as poselib does by storing poses as quaternions).
- **Truncated focal reprojection scorer**
  (`src/fastpose/scorers/reprojection.py`): the `[R | t]` scorer with the
  focal folded into the projection, same division-free inlier test and
  per-block early bail-out.
- **LM refiner** (`src/fastpose/refiners/absolute_focal.py`): 7 tangent
  parameters (rotation, translation, log-focal) with an analytic jacobian.

</details>

<details>
<summary>Monodepth-assisted relative pose</summary>

Relative pose from correspondences plus per-image monocular depth estimates
(scale- or affine-invariant MDE outputs), following poselib's monodepth
estimators. The depth convention is poselib's `MonoDepthTwoViewGeometry`: the
3D point of correspondence i is `(d1_i + shift1) x1h_i` in camera 1 and
`scale (d2_i + shift2) x2h_i` in camera 2. Four 3-point minimal solvers
(`src/fastpose/solvers/monodepth.py`):

- **Calibrated without shift** (`estimate_relative_pose_with_monodepth`,
  scale-invariant depths): 3D points from the camera-1 depths, absolute pose
  of camera 2 via the shared P3P kernel, depth scale from the first
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
  port of `relpose_monodepth_3pt_varying_focal` — a single 3x3 linear system
  in (1/f1^2, s^2/f2^2, s^2). Models `[R | t | f1 | f2 | scale]`.

All four assemble the pose from the two depth-induced point triplets
(`Y X^-1` like poselib, with the quaternion SO(3) snap) and share one data
layout: six columns (x1, y1, x2, y2, d1, d2) plus the two hybrid weights.

- **Scoring** is the plain truncated Sampson error (threshold `max_error`,
  poselib's `max_errors[1]`); the depth parameters do not enter the score.
- **Local optimization** (`src/fastpose/refiners/monodepth.py`) minimizes the
  hybrid cost of poselib's `MonoDepth*RelPoseRefiner`s: truncated Sampson
  (weighted by `weight_sampson`) plus the truncated symmetric reprojection
  error through the monodepth 3D points, scaled by
  `scale_reproj = max_error^2 / max_reproj_error^2` so both terms truncate
  at the same threshold (defaults 2 px Sampson / 16 px reprojection).
  Analytic jacobians, verified against finite differences in the tests.
  Tangent parameters: rotation (3), full translation (3, the depths fix the
  scale), plus additive scale / shifts / focals per problem (7, 9, 8 and 9
  parameters).

</details>

## Next steps

- Degeneracy handling (e.g. dominant-plane checks a la DEGENSAC).

## Citations

If you find this library useful please consider citing:

```bibtex
@misc{kocur2026fastpose,
  title  = {{fastpose}: a fast Python backend for robust camera pose estimation},
  author = {Kocur, Viktor},
  year   = {2026},
  url    = {https://github.com/kocurvik/fastpose}
}
```

To cite individual solvers see [REFERENCES.md](REFERENCES.md).