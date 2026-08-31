# Porting the CUDA backend to the remaining estimators

`fastpose.cuda` started as a single-problem backend (calibrated relative pose,
5-point). This document is the record of extending it to the other eight
estimators, written from what the ports actually ran into rather than from what
they looked like they would need.

**All nine are done.** The backend is no longer problem-specific: the driver,
the scorer, the LM loop and the reductions are shared, and a problem is one
module under `cuda/problems/` plus one registry entry. Read
[§1](#1-the-hard-constraints) before writing any device code and
[§2](#2-architecture-the-seam-between-the-driver-and-a-problem) before touching
a problem module; [§5](#5-per-problem-notes) has what each one specifically
needed.

---

## 0. Current state

| Estimator | Solver | CUDA solve | CUDA score | CUDA LM | `device='cuda'` |
|---|---|---|---|---|---|
| `estimate_relative_pose` | 5-point | ✅ | ✅ | ✅ | ✅ |
| `estimate_absolute_pose` | P3P | ✅ | ✅ | ✅ | ✅ |
| `estimate_absolute_pose_with_focal` | P4Pf | ✅ | ✅ | ✅ | ✅ |
| `estimate_fundamental` | 7-point | ✅ | ✅ | ✅ | ✅ |
| `estimate_relative_pose_with_varying_focals` | 7-point + Rybkin | ✅ | ✅ | ✅ | ✅ |
| `estimate_relative_pose_with_shared_focal` | 6-point | ✅ | ✅ | ✅ | ✅ |
| `estimate_*_with_monodepth` (×3 entry points, 4 solvers) | 3-point ×4 | ✅ | ✅ | ✅ | ✅ |

**All nine are done.** What remains is not porting work - see
[§7](#7-state-of-the-checklist).

### What is already there and must be reused, not rewritten

```
cuda/backend.py      the jit shim and the thread-count constants
cuda/reductions.py   reduce_sum / reduce_scalar / reduce_count,
                     get_solve_damped(n)  (damped Cholesky, memoized per n)
cuda/scoring.py      build_score_kernel(...), ScoreBuffers, NO_MODEL
cuda/lm.py           build_lm_refine_kernel(...), lm_threads_for,
                     RefineBuffers, gather_candidates
cuda/problem.py      CudaProblem - the seam; lazily builds and memoizes the
                     score kernel and one LM kernel per loss
cuda/registry.py     name -> CudaProblem, imported lazily
cuda/ransac.py       the batched driver; problem-agnostic
cuda/problems/       one module per problem, plus common.py
```

`cuda/solvers.py`, `cuda/scorers.py` and `cuda/refiners.py` are **gone** - their
problem-agnostic halves became `scoring.py` / `lm.py` / `reductions.py` and their
5-point halves became `problems/essential.py`.

---

## 1. The hard constraints

Properties of numba's CUDA target, verified on numba 0.65 and 0.67. The first
three have no workaround.

### 1.1 No `np.empty` and no `.reshape` in device code

Numba's runtime (NRT) is host-only. Both fail at link time:

```
error : Undefined reference to 'numba_attempt_nocopy_reshape' in '<cudapy-ptx>'
```

Every kernel that allocates or reshapes must be restructured so scratch is
**passed in already shaped**. This is the single most invasive change and it is
why every ported solver was split into a `_solve_*_core` plus a thin CPU
wrapper (see [§3.1](#31-step-1--refactor-the-solver-into-a-factory)).

### 1.2 A device function cannot *return* a `cuda.local.array`

The tempting shim - `alloc9()` returning `cuda.local.array(9, float64)` so one
source works for both backends - does not compile.

**But allocating one *inside* a device function and using it locally is fine**
(verified this session). That is what makes the thread-0 kernels of the
fundamental and varying-focal ports tractable: they allocate their own 3×3
scratch instead of threading it through the generic LM kernel's signature.

### 1.3 Shared memory is 48 KB per block by default

Budget it before choosing a thread count. `cuda/lm.py::lm_threads_for` now does
this automatically: the reduction array alone is `LM_THREADS × NUM_ACC` doubles
with `NUM_ACC = T(T+1)/2 + T`, so it picks the largest power of two under the
budget - 128 threads up to `T = 7`, 64 at `T = 8` and `T = 9`. Do not hard-code
a thread count in a new problem module.

### 1.4 A compiled module gets 64 KB of global constant data

Module-level numpy arrays read from device code are emitted into the module's
constant bank, and ptxas caps that bank per compiled file:

```
ptxas error : File uses too much global constant data (0x17f70 bytes, 0x10000 max)
```

The cap is 65536 bytes. The 6-point solver's index tables are 96 KB at
`int64`, so its solve kernel would not link at all. The fix is to store each
table in the narrowest dtype that holds its values - the shared-focal tables
index at most 2520 entries and mostly at most 26, so `uint8` / `int16` bring
the module to ~31.5 KB. `solvers/shared_focal.py` carries the ranges as
comments.

Two things to note before the next port. First, the budget is **per compiled
file**, and device functions are inlined into the kernel that calls them, so a
kernel's own tables share the bank with every table its callees pull in - the
5-point solver's `_T44` and the essential decomposition ride along in the
shared-focal kernel. Second, do **not** narrow float tables to buy room unless
you have checked what it does to the arithmetic; the shared-focal
coefficients are left at `float64` deliberately, because that solver is not
fastmath precisely because its conditioning makes last-bit changes visible
([§5](#5-per-problem-notes)). Narrowing only the integer tables was enough and
leaves 34 KB spare.

`solvers/monodepth.py` has no module-level tables at all, so the monodepth
port does not have to think about this.

### 1.5 `syncthreads()` must be block-uniform

Every branch containing a `cuda.syncthreads()` must be taken by all threads of
the block or none. The generic LM loop achieves this by computing all control
decisions in thread 0, publishing them to a shared `ctl` array, and syncing
before any thread reads them. Violating this deadlocks or corrupts silently -
it does not raise.

### What *does* work (verified; don't re-derive)

- **Tuples of device arrays as kernel arguments, of any width.** Probed at 4, 5
  and 8 columns. This is what lets one driver serve problems with different
  data layouts - the whole `data` tuple is now a single kernel argument and
  `data[k][i]` indexes it inside device functions.
- `cuda.local.array` with a shape from a module-level or closure constant,
  including 2-D, and including *inside a device function* ([§1.2](#12-a-device-function-cannot-return-a-cudalocalarray)).
- `cuda.shared.array` sized from a **closure** constant - which is what makes
  `build_lm_refine_kernel` able to size `red`, `B64`, `aux` per problem.
- A row view of a 2-D shared or local array (`aux[0]`, `B64[p]`) passed as a
  1-D argument to a device function.
- Slicing a shared array (`pose[9:12]` inside `essential_tangent_rows_core`).
- The same Python function object compiled by both `njit` and
  `cuda.jit(device=True)`.
- Calling an `njit` global from device code **when it is a scalar function** -
  the `loss.weight` / `loss.cost` kernels are `njit` and the CUDA target
  recompiles them for the device. This is the one exception to "every callee
  must come from the factory"; anything touching arrays still must not be.
- `cuda.jit(cache=True)` - the on-disk cache works, so `fastpose-warmup` extends.
- Module-level numpy arrays read as constants inside device functions
  (`_T44`, and the 6-point solver's large index tables) - but only up to
  64 KB per compiled module; see [§1.4](#14-a-compiled-module-gets-64-kb-of-global-constant-data).
- `math.isfinite`, `math.sqrt`, `math.copysign`, `math.log`, `math.exp`.
  **Not** `np.inf` as a literal.
- 2-D shared arrays, `float64`/`int64` reductions.

### Version-sensitive: no `min`/`max` in device code on numba 0.67

Builtin `min`/`max` compile on numba 0.65 and fail on 0.67 - in kernels as well
as in device functions, for any argument count:

```
TypeError: Signature mismatch: 2 argument types given, but function takes 1 arguments
```

0.67 retyped both builtins through an `@overload` whose implementation is
variadic (`def impl(*x)`). The CPU dispatcher folds varargs; the CUDA
device-function dispatcher hands the raw argument tuple to `compile_device`, so
a two-argument call compiles a one-parameter function with two argument types
and dies in numba's `FixupArgs` pass. Spell the comparison out instead:

```python
abs_lo = abs(lo)
lo_scale = abs_lo if abs_lo > 1.0 else 1.0    # not max(1.0, abs(lo))
```

This bites the **shared** sources hardest: the call site sits in a solver that
reads as CPU code, the CPU build lowers it fine, and the failure appears only
on a machine with the newer numba - for us the cluster (numba 0.67, python
3.12) against a dev box on 0.65. So when a GPU compile fails where nothing
reproduces locally, check the numba versions before anything else. The `fp312`
env reproduces the cluster.

**Status: the whole package is clear.** A grep of `src/fastpose` for
`(^|[^A-Za-z0-9_.])(min|max)\(` now returns only host-side Python - the driver
loops in `cuda/ransac.py`, `estimators/ransac.py`, `estimators/warmup.py`, two
numpy calls in `estimators/essential.py`, and the three comments marking the
sites that were rewritten. `solvers/shared_focal.py` and
`solvers/monodepth.py` had none to begin with, so neither port had to start
with a rewrite.

The whole suite has since been run under the `fp312` env (numba 0.67, python
3.12), so every problem's device code is known to compile there, not just to
compile locally on 0.65 ([§8](#8-what-to-do-next) has the command).

---

## 2. Architecture: the seam between the driver and a problem

`CudaProblem` (`cuda/problem.py`) is everything the driver knows about a
problem. A port writes one module under `cuda/problems/` that ends in a
`PROBLEM = CudaProblem(...)`, and adds one line to `cuda/registry.py`.

### 2.1 The three things a problem supplies

**A solve launcher** `solve_batch(data, params, samples, models, counts, stream)`.
One thread per hypothesis; the kernel allocates the solver's scratch as
`cuda.local.array` and calls the `_solve_*_core` from the shared factory.

**Two scorer device functions:**

```python
prepare(model, params, dm64) -> bool
    # float64, thread 0, once per model: the *derived* form the per-point loop
    # reads - E = [t]_x R, the induced F = K2^-T E K1^-1, the focal-scaled
    # pose. False marks a model the CPU scorer reports as (1e300, 0).

score_point(dm32, m32, data, i, max_error_sq) -> (r, ok)
    # float32, per correspondence. `dm32` is the rounded derived form, `m32`
    # the rounded model itself (the cheirality check needs the pose, not just
    # E). `r` is the normalized squared error, read only when `ok`.
```

Keep the inlier test in its **unnormalized** form (`r2 < max_error_sq * den`)
inside `score_point`: that is what makes it agree with the CPU scorer point for
point, which deliberately does not divide for outliers.

**A `build_refine_kernels(loss)` factory** returning eight device functions:

```python
init_state(model_in, state) -> bool
state_to_model(state, model)
model_derived(model, params, dm64) -> bool
jacobian_basis(model, state, params, dm64, B64, aux)
mask_point(dm32, st32, data, i, relaxed_sq) -> bool
cost_point(dm32, st32, data, i, max_error_sq) -> float32
accum_point(dm32, st32, B32, data, i, max_error_sq, red, tid, scratch) -> int
retract(state, delta, state_new, aux)
```

The float64 ones run in thread 0; the three float32 ones are the per-point hot
loop. `accum_point` adds this point's `NUM_ACC` contributions into
`red[tid, :]` and returns how many scalar residuals it contributed (1 for a
Sampson problem, 2 for a reprojection one).

### 2.2 The four shared workspaces

Split so no device function ever does offset arithmetic into one flat blob:

| | what it holds |
|---|---|
| `dm64` / `dm32` | the derived form (`derived_size` entries) |
| `model` / `st32` | the flat model and its float32 mirror, for per-point work needing the pose rather than the matrix |
| `B64` / `B32` | the tangent basis, `(num_tangent, basis_width)`; `basis_width = 1` for the reprojection problems, which have an analytic jacobian |
| `aux` | float64 `(aux_rows, aux_cols)` that must survive between thread-0 calls - the two translation basis vectors the retraction shares with the jacobian |

The LM kernel keeps `state`/`state_new` and `model`/`model_new` separately, so a
rejected trial never leaks into the returned model. It also re-derives
`dm64` at the top of every recompute branch rather than relying on what the
last trial left behind - O(1), and it removes an ordering hazard that is very
easy to reintroduce.

### 2.3 The `relaxed_scale` convention

The epipolar refiners restrict local optimization to the relaxed-threshold
inliers the way poselib's `refine_model` does. **The absolute-pose and
fundamental CPU refiners do not** - their estimators hand the RANSAC-internal
refinement the whole correspondence set, and only the final polish sees an
inlier-only one.

The generic LM kernel always applies a mask, so those problems pass
`relaxed_scale = 0.0`, which their `mask_point` reads as "keep everything":

```python
if relaxed_sq <= float32(0.0):
    return True
```

and set `min_inliers = -1` so the inlier-count gate never fires (the
`num_residuals < num_tangent` check inside the loop still does, matching
`build_lm_refine`). `final_refine` still passes `relaxed_scale = 1.0` and gets
the model's own inlier set. Getting this wrong is silent: the GPU LM converges
to a slightly different minimizer than the CPU one and only the
minimizer-quality assertion catches it.

### 2.4 Per-problem constants that are not per-point data

The varying- and shared-focal problems need the two principal points, and the
monodepth ones need `scale_reproj` and `weight_sampson`. These ride in a
**`params` float64 device vector** passed to every kernel, not in the `data`
tuple - the driver mirrors `data` to float32 per point and `params` is neither
per-point nor float32-safe. The CPU solvers keep their 8-element `data` tuple;
their `_core` takes the constants as scalars and the coordinate columns as a
4- or 6-tuple, and the CPU wrapper does the unpacking. See
`solvers/varying_focal.py` for the pattern.

The estimator entry point passes `params=` to `CudaRansacEstimator.estimate`.

**But the per-point kernels never see `params`.** Their signatures are
`score_point(dm32, m32, data, i, max_error_sq)`,
`cost_point(dm32, st32, data, i, max_error_sq)` and `accum_point(...)` - only
the thread-0 functions (`prepare`, `model_derived`, `jacobian_basis`) get it.
For the focal problems that is fine, because the principal points are needed
only to *form* the derived matrix. Monodepth is the first problem where a
constant is needed per point: both its weights gate branches inside
`cost_point` and `accum_point`.

The fix is to **carry them in the derived form**. `derived_size` is a
per-problem number and the generic kernels mirror the whole vector to float32
for free, so monodepth sets `derived_size = 11` - the epipolar matrix in 0..8,
`scale_reproj` at 9 and `weight_sampson` at 10, written from `params` by
`prepare` and `model_derived`. No driver change, no extra data column, and the
per-point kernels read them as float32 like everything else. Use this for any
future per-point constant rather than widening `data`.

---

## 3. Step-by-step for one problem

Do these in order and **verify at each step**. The parity tests are what make
this tractable; without them a precision or indexing bug surfaces only as a
slightly worse pose 2000 lines later.

### 3.1 Step 1 - refactor the solver into a factory

Purely mechanical, and CPU behaviour must not change.

1. Wrap all `@njit` kernels of `solvers/<problem>.py` in
   `def build_<problem>_kernels(jit, <callees>):`, indenting by 4.
2. Rewrite the decorators:
   `@njit(cache=True, fastmath=True)` → `@jit(fastmath=True)`,
   `@njit(cache=True, inline='always')` → `@jit(inline=True)`, etc.
3. Split the top-level `_solve_*` into:
   - `_solve_*_core(data, sample, models, <every scratch array>)` - inside the
     factory, no allocation, no reshape;
   - a thin CPU-only `_solve_*(data, sample, models, workspace)` wrapper
     outside it that slices and reshapes the flat workspace and calls the core.
     This preserves the RANSAC driver's contract exactly.
4. **Pass cross-solver callees in as factory arguments** rather than importing
   them. `p4pf` takes `real_roots_sturm`, `varying_focal` takes
   `nullspace_7pt, det3_flat, solve_cubic_real, pose_from_essential`. On the
   CPU this reuses the very kernel objects the other module already
   instantiated (a second instance duplicates its on-disk cache entry); on the
   GPU it is mandatory, because a device function cannot call an `njit` global
   that touches arrays.
5. Instantiate `_CPU = build_<problem>_kernels(cpu_jit, ...)` and rebind every
   module-level name other modules import - check with
   `grep -rn "from fastpose.solvers.<problem> import"`.

**Verify.** The indentation is done by script, so check the maths survived. The
scratchpad script `check_refactor.py` (regenerate it if lost - it is ~90 lines)
extracts each function's statements from `git show HEAD:<file>` and from the
working tree, strips indentation/comments/decorators, and compares; for a split
function it concatenates the wrapper and the core, drops the wrapper's trailing
`return <core>(...)`, and falls back to a multiset comparison because the split
reorders the data unpack past the slicing. Results this session:

| file | result |
|---|---|
| `solvers/p3p.py` | 6/6 statement-identical |
| `solvers/fundamental.py` | 6/6 statement-identical |
| `solvers/p4pf.py` | 5/8; the three differences are the two `max()` rewrites and the removed reshape/`iw` allocations, all counted and accounted for |
| `solvers/varying_focal.py` | 2/3; the one difference is the core's extra `x1_x, ... = data` unpack |

Then run the full CPU suite - it must be green before you go further.

### 3.2 Step 2 - the CUDA solve kernel

One thread per hypothesis. The solvers are scalar and branchy; there is no
useful parallelism inside one minimal sample.

```python
@cuda.jit(cache=True)
def _solve_batch_kernel(data, params, samples, models, counts):
    i = cuda.grid(1)
    if i >= samples.shape[0]:
        return
    A = cuda.local.array((5, 9), float64)
    ...                                   # one per scratch piece
    counts[i] = _solve_core(data, samples[i], models[i], A, N, ...)
```

Local memory is per-thread storage that the hardware interleaves across a warp,
so thread-uniform access coalesces for free - no manual layout needed. Keep the
solver in **float64** (see [§4](#4-precision-policy)). Pick the block size from
the per-thread local footprint: 128 for P3P (~50 doubles) and the 7-point
solvers, 64 for the 5-point and P4Pf (~6.9 KB and ~4.7 KB), 32 for the
6-point shared-focal solver (~30.5 KB).

**Verify.** Run the CPU `_solve_*` and the kernel on *identical* samples.
Model counts must match exactly (the solver's branches are on tolerances far
above codegen noise); models should agree to ~1e-6. A count mismatch means a
real bug, not rounding.

### 3.3 Step 3 - the scorer

Write `prepare` and `score_point` ([§2.1](#21-the-three-things-a-problem-supplies))
and hand them to `CudaProblem`. `build_score_kernel` does the rest: one block
per hypothesis, threads stride over correspondences, tree-reduce
`(score, inlier count)` in shared memory, loop over the several models one
sample yields keeping the running best in thread 0's registers, so the host
reads back five scalars per hypothesis rather than a `batch × max_models` table.

The per-point maths is **not** rewritten - it comes from the same factory the
CPU scorer uses (`build_sampson_point_kernels`,
`build_reprojection_point_kernels`).

Properties the shared kernel already preserves, and that you should not
re-implement: `NO_MODEL` (1e300) and index `-1` in unused slots; the derived
form built in float64 and rounded once; and the early bail-out.

**Keep the early bail-out.** It looks impossible on a block reduction and is
not: the truncated score sums non-negative terms, so any partial sum bounds the
total. Score a chunk, re-reduce the *running* total, and let every thread read
the same shared value - the decision is then block-uniform and the
`syncthreads()` stay legal. Grow the chunks geometrically: a hopeless model
almost always bails on the first one, so a small first chunk buys nearly all
the saving while doubling keeps the reduction count logarithmic in `n`. Fixed
512-point chunks cost 60% when the bail never fires; geometric ones cost ~18%
at 16k matches and ~3% at 50k, against a 1.4-1.7x gain when it does. Bail
against the best minimal score as of the round's start.

**Verify.** Against the CPU scorer with `best_score=1e300` (which disables its
bail-out). Assert inlier counts within 1, scores to `rtol=1e-5`, and - more
useful than an index match - that the model the GPU *selected* scores within
1e-5 of the true best. Two models can tie within rounding; either pick is right.

### 3.4 Step 4 - the LM refiner

Write the eight device functions ([§2.1](#21-the-three-things-a-problem-supplies))
and hand the factory to `CudaProblem`. `build_lm_refine_kernel` runs the
**entire LM loop inside one kernel launch** - launching an accumulate kernel per
LM step would pay launch latency 25 times per candidate and dominate everything.

It splits by cost:

- **O(n)** - relaxed inlier mask, cost, normal equations → across the block's
  threads, tree-reduced.
- **O(1)** - damped Cholesky, retraction, damping schedule → thread 0, published
  through shared memory.

All normal-equation accumulators reduce in **one pass**: the upper triangle of
`JtJ` plus `Jtr`, `NUM_ACC = T(T+1)/2 + T` doubles per thread. Shared budget:

```
red      LM_THREADS × NUM_ACC × 8 B      (128 × 20 × 8 = 20.5 KB for T=5)
+ state, model, dm, B, aux, JtJ/Jtr/A/delta, reduction scalars   ≈ 1.5 KB
                                        must stay under 48 KB
```

`lm_threads_for(T)` picks the thread count; `get_solve_damped(T)` gives the
memoized Cholesky. At `T=9`, `NUM_ACC = 54` and 128 threads would need 55 KB, so
it drops to 64. If that ever costs too much parallelism, the alternative is a
warp-level reduction first (`cuda.shfl_down_sync`) staging only 32 partial sets.

Match the CPU semantics exactly: select the subset **once** from the initial
model with whatever check the CPU refiner applies, then minimize over that fixed
subset - and get [§2.3](#23-the-relaxed_scale-convention) right. Carry the subset
as a per-candidate `uint8` mask; device code cannot compact into new arrays.

**Verify.** Two assertions, and the second is the one that matters:

1. Parameters against the CPU refiner, `rtol=1e-3, atol=1e-4` (float32
   jacobians move the converged pose at that level).
2. **Minimizer quality**: scored back in full float64, the GPU result must be no
   worse than the CPU refiner's by more than ~1e-4 relative. An element-wise
   tolerance alone cannot tell "rounded differently" from "converged somewhere
   worse"; this can.

### 3.5 Step 5 - the driver

Nothing to do. `cuda/ransac.py` is problem-agnostic: it reads `sample_size`,
`num_params`, `max_models`, `data_width` and the kernels off the `CudaProblem`.
Do **not** re-derive the round policy; see [§6](#6-performance-notes).

### 3.6 Step 6 - plumbing

- `device='cuda'` and `batch=None` on the estimator entry point, `check_device`
  for the `ValueError`, and `get_cuda_estimator('<name>', batch)` from
  `estimators/utils.py` (it owns the module-level cache keyed on
  `(problem, batch)` - recreating an estimator per call reallocates the device
  buffers and re-warms).
- Route the final polish through `CudaRansacEstimator.final_refine`, which is
  the same LM kernel at `relaxed_scale=1.0` with the Cauchy loss. At large `n`
  this pass is O(n) per step for 100 steps and will dominate if left on the CPU.
- Extend `_cuda_warmup_steps` in `estimators/warmup.py`. A cold GPU estimate
  compiles the solver, the scorer and **two** LM kernels (one per loss); that is
  seconds, and it is easy to mistake for steady-state cost when benchmarking.

  **Check that the warmup scene actually reaches the polish pass.** The second
  LM kernel is built for the Cauchy loss and only runs when the estimate found
  inliers, so a scene that is degenerate for a solver leaves that kernel cold
  and says nothing. It had: the absolute-pose warmup built its world points as
  `column_stack([x1 * depth, depth])`, which puts the world origin *at* the
  camera and makes the pose to recover `(I, 0)`. P3P copes; **P4Pf does not**,
  so absolute-focal found zero inliers at every iteration count and warmed
  neither of its LM kernels - on the CPU backend too, since before this port.
  `_synthetic_world_points` now maps the points through the synthetic pose.
  `tests/test_warmup.py` pins it per problem by clearing the estimator
  module's `_final_refiner` and asserting the warmup sets it.
- Add a `cuda-scaling` mode to the problem's benchmark.

---

## 4. Precision policy

Mixed, chosen per component from measured conditioning. Apply the same split:

| Component | Precision | Rationale |
|---|---|---|
| Minimal solver | **float64** | The 5-point action matrix exceeds cond 1e5 on ~15% of samples and 1e7 on 1.3%, and Danilevsky is less stable than QR on top of that. Also only O(batch) work, so float64 is free. |
| Scoring, LM residual + jacobian | **float32** | The O(iterations × matches) hot spot. |
| Score accumulator, `JtJ`/`Jtr`, Cholesky | **float64** | Sums over all correspondences must not drift; `cond(JtJ)` ≈ 3e3 is well inside float32's limit but accumulating in float64 is nearly free. |
| Pose state, model→matrix map, retraction | **float64** | O(1) per LM step, and the state is the returned answer. |

Measured cost of that split, against a full float64 reference: inlier counts
agree on ~99% of models and never differ by more than one point; scorer model
selection never flipped over 200 models; the mixed LM is a better-or-equal
minimizer than the float64 CPU refiner in 29/32 cases, worst case 4e-7 relative.

**float16 is not viable.** The Sampson numerator sums O(1) terms to an O(1e-3)
result - ~3 digits of cancellation, and fp16 has 3.3 total. Worse,
`max_error_sq` (4e-6) is *subnormal* in fp16, so the entire inlier decision
happens where fp16 has no precision left. Measured: 3.5% of the inlier set wrong
(Jaccard 0.964, up to 102 points), against 0 for float32. Rescaling overflows
fp16's 65504 ceiling. Do not revisit.

**Float literals are float64 in numba.** A single bare `1.0` in an otherwise
float32 expression silently promotes the whole chain back to double and undoes
the mixed precision. Factories that run per point take a `real=` parameter and
write `real(1.0)`. See `build_sampson_point_kernels`,
`build_reprojection_point_kernels` and `build_reprojection_primitives`.

**Verify the precision actually took effect.** A stray float64 literal is
silent. Inspect the PTX:

```python
for sig, ptx in kernel.inspect_asm().items():
    print(len(re.findall(r"\.f32", ptx)), len(re.findall(r"\.f64", ptx)))
```

Two traps in doing this. First, **numba emits `rcp.rn.f64` for `1.0/x`, not
`div`** - a regex that only looks for `div` will report zero float64 divisions
in a kernel that has plenty. Match `(sqrt|rsqrt|div|rcp)\.[\w.]*f64`. Second, a
whole-kernel count cannot separate the per-point loop from the O(1) thread-0
scaffolding, and the LM kernel's f64 sqrt/div is dominated by the scaffolding
(the retraction's normalize, the damped Cholesky, `svd3`) - so the whole-kernel
number is uninformative for the LM. Isolate the hot loop instead: compile a
probe kernel that calls **only** the per-point device functions -
`score_point`, `mask_point`, `cost_point`, `accum_point` - once each, and
inspect that. Everything in its PTX is per-point by construction.

Measured this way for all ten problems:

| | score kernel | per-point probe |
|---|---|---|
| essential | 2 f32 sqrt, 1 f32 div, 2 f32 rcp; **0 f64** | 5 f32 sqrt, 4 f32 div, 4 f32 rcp; **0 f64** |
| absolute | 1 f32 div; **0 f64** | 3 f32 div; **0 f64** |
| absolute-focal | 1 f32 div; **0 f64** | 3 f32 div; **0 f64** |
| fundamental | 1 f32 div; **0 f64** | 1 f32 sqrt, 4 f32 div; **0 f64** |
| varying-focal | 1 f32 div; 2 f64 rcp | 1 f32 sqrt, 4 f32 div; **0 f64** |
| shared-focal | 1 f32 div; 2 f64 rcp | 1 f32 sqrt, 4 f32 div; **0 f64** |
| monodepth | 1 f32 div; **0 f64** | 1 f32 sqrt, 8 f32 div; **0 f64** |
| monodepth-shift | 1 f32 div; **0 f64** | 1 f32 sqrt, 8 f32 div; **0 f64** |
| monodepth-shared-focal | 1 f32 div; **0 f64** | 1 f32 sqrt, 12 f32 div; **0 f64** |
| monodepth-varying-focal | 1 f32 div; **0 f64** | 1 f32 sqrt, 12 f32 div; **0 f64** |

Every per-point `sqrt` and division is `.f32` in every problem. The two f64
`rcp` in the varying- and shared-focal scorers are `1/f1` and `1/f2` inside
`prepare`, which is O(1) per model and is deliberately float64
([§3.3](#33-step-3--the-scorer)). The monodepth probes carry more f32
divisions because they exercise four reprojection kernels (the residual-only
pair and the residual-plus-jacobian pair) on top of the Sampson ones. The f64
instructions the probes do retain (84-586) are the float64 accumulation into
`red` and the score, which is the policy in the table above.

### Cache keys

`kernel_cache.stabilize` hashes closure cells to give `cache=True` a
process-independent key. It understands kernels, literals, and (since the
mixed-precision work) numba type objects. If you add a closure cell of some
other type, `_tag` raises `_Unstable`, which is swallowed - and the effect is
silent: every kernel that reaches it loses its on-disk cache and recompiles
every run. If `fastpose-warmup` stops helping, look here first.

---

## 5. Per-problem notes

### Ported

**Absolute pose (P3P)** - `sample_size=3, num_params=12, max_models=4,
workspace=50`, `NUM_TANGENT=6`. Needed a new reprojection point-kernel factory
(`build_reprojection_point_kernels` in `scorers/reprojection.py`) and an
analytic jacobian factory (`build_reprojection_primitives` in
`refiners/absolute.py`). Both CPU scorers now go through them, so there is one
copy of the maths. Uses `relaxed_scale = 0.0` ([§2.3](#23-the-relaxed_scale-convention)).

**Absolute pose with focal (P4Pf)** - `sample_size=4, num_params=13,
max_models=1, workspace=454 + 137 int64`, `NUM_TANGENT=7`. Reuses the Sturm
machinery from the 5-point solver, passed into `build_p4pf_kernels`. The focal
is folded into the pose once per model by `focal_scale_pose`, after which the
residual is the plain calibrated one and is literally the same device function;
`build_reprojection_primitives(focal=True)` adds the `dr/dlog f` column. Two
`max()` sites cleared. Uses `relaxed_scale = 0.0`.

**Fundamental (7-point)** - `workspace=93, max_models=3`, `NUM_TANGENT=7`,
`state_size=19`. The LM was the blocked piece: `svd_init_state` called
`np.linalg.svd`, which is host-only. Resolved by **option 1** - a one-sided
Jacobi `svd3` added to `build_refiner_primitives`, used by *both* backends, so
`np.linalg.svd` is gone from the package. Measured over 3000 random, rank-2 and
badly-scaled 3×3 inputs: reconstruction 2.0e-15, orthogonality 1.6e-15,
singular values 9.7e-16 relative. Notes:
- The third left singular vector is taken from the cross product of the first
  two only when `s[2] <= 1e-15 * s[0]`; taking it always breaks reconstruction
  for full-rank inputs whose true `U` has det -1.
- `u3`'s sign and `V`'s orientation are gauge: `F = U diag(1,σ,0) Vt` does not
  depend on them, and `U·SO(3)` covers the same set of orthonormal 2-frames
  either way, so a different SVD convention than LAPACK's only reparametrizes
  the tangent space. The CPU tests confirm this.
- `state_size` (19) ≠ `num_params` (9) here, which is why the generic LM kernel
  keeps both and mirrors the *model* for the per-point loops.
- The state's `U`/`Vt` are 3×3 but a shared array is flat and `.reshape` does
  not compile, so the thread-0 kernels copy 18 doubles into per-thread local
  3×3 arrays before calling the shared cores. O(1) against an O(n) accumulate.
- Uses `relaxed_scale = 0.0`.

**Varying focal (7-point + Rybkin)** - `num_params=14, max_models=12,
workspace=122`, `NUM_TANGENT=7`. First problem needing the `params` vector
([§2.4](#24-per-problem-constants-that-are-not-per-point-data)) for the two
principal points. `calibrate_epipolar` and `model_to_fundamental` moved into
`build_sampson_point_kernels` as allocation-free `*_core` variants;
`log_focal_tangent_rows` moved into `build_refiner_primitives`. The scorer and
the relaxed mask have **no cheirality check** (poselib scores an F here) - that
difference is deliberate and is preserved.

**Shared focal (6-point)** - `sample_size=6, num_params=14, max_models=60,
workspace=3662`, `NUM_TANGENT=6`, `state_size=13`, `MIN_INLIERS=6`,
`relaxed_scale=LO_INLIER_SCALE`. The refiner is the varying-focal one with the
two focals tied: one log-focal tangent row, formed as the **sum** of the two
rows `log_focal_tangent_rows` produces. Scorer, derived form and per-point
jacobian are literally varying-focal's device functions, cheirality check
absent in both, as there.

Three things this port hit that the notes had not anticipated:

- **The 64 KB constant-data cap** ([§1.4](#14-a-compiled-module-gets-64-kb-of-global-constant-data)).
  The solve kernel would not link until the index tables were narrowed. This
  was the only genuine blocker.
- **Occupancy was the smaller problem.** The kernel does carry ~30.5 KB of
  per-thread local memory (3665 doubles + 144 int64) and `SOLVE_THREADS` is
  32 rather than 64 or 128 because of it, but the feared collapse did not
  happen. Measured solve kernel: 4.24 ms for a single hypothesis, then
  10.40 / 10.45 / 10.84 / 12.10 / 18.17 ms at 32 / 128 / 512 / 1024 / 4096 -
  4.4 us per hypothesis at 4096, against the 5-point solver's 1.4 us. So the
  per-round floor is ~3.2x the 5-point's and the shape of the curve is the
  same. One CPU solve is 62.6 us.
- **The solver cannot be held to an elementwise CPU/GPU tolerance.** This is
  the one solver built without `fastmath`, because its conditioning amplifies
  last-bit differences into the 8th significant digit; NVVM associates
  differently from LLVM regardless of that flag. Measured over 512 samples,
  the per-model relative difference has median 2.5e-12 and p90 2.1e-10 but a
  **maximum of 5.5e-2** on near-degenerate samples. Model *counts* agree
  exactly on every sample of both a clean and a noisy scene, which is the
  check that catches a structural bug; the parity test asserts the
  distribution (median < 1e-9, >95% within 1e-6) instead of an elementwise
  bound, which a heavy tail would only let you set uselessly loose. See
  `_solver_models_shared_focal` in tests/test_cuda.py.

The refactor was verified bit-exact rather than statement-identical: the
statement checker reports 5/8 (the `_EIG_IND` conversion, the factory-parameter
renames and the inlined workspace carving), so the check that matters ran the
pre-refactor module out of `git show HEAD:` alongside the new one on 400
identical samples - 0 count mismatches, max absolute difference **0.0**.
Regenerate that script if it is lost; it is ~60 lines and worth more than the
statement diff on a solver this branchy.

**Monodepth (×4 solvers, ×3 entry points)** - `sample_size=3`,
`num_params=15`, `state_size=15`, `max_models` 4/4/4/1, `NUM_TANGENT` 7/9/8/9
(at 8 and 9 `lm_threads_for` drops to 64 threads on its own). One module, four
`PROBLEM_*` objects, `data_width=6`, `relaxed_scale=0.0` and `min_inliers=-1`
(the CPU refiners have no relaxed-inlier wrapper), and **no cheirality check**
anywhere - the translation is metric rather than unit, so `MIN_DEPTH` would
mean something different than it does for calibrated relative pose.

Three things worth knowing:

- **The two hybrid weights are per-point constants**, which the seam did not
  previously support - they now ride in the derived form
  ([§2.4](#24-per-problem-constants-that-are-not-per-point-data)).
- **The hybrid cost is the only genuinely new device code.** `accum_point`
  contributes up to three residual families per correspondence (2 + 2 scalar
  reprojection residuals and 1 Sampson), so `scratch_shape` is `(4, 9)` -
  dsdF, J, J0, J1 - as predicted.
- **The refiner needed a factory extraction first.** Unlike the other six
  problems, monodepth's per-point maths lived inline in
  `_make_accumulate_*`, so there was nothing for the GPU to instantiate.
  `refiners/monodepth.py` now exposes `build_monodepth_primitives` (the E->F
  map, the tangent basis, the focal rows) and two point-kernel factories:
  `build_monodepth_reproj_kernels` (residual **and** jacobian rows, like
  `reprojection_point_jacobian`) and `build_monodepth_residual_kernels`
  (residual only, for the O(n) cost evaluation, like `sampson_residual`
  against `sampson_point_jacobian`). Both backends go through them.

That extraction also removed two duplicates: monodepth had its own copies of
the Sampson residual and jacobian, byte-identical to `sampson_residual` and
`sampson_point_jacobian` - they are gone, and the calibrated hybrid cost came
out bit-identical afterwards, which is what proved they really were the same
maths. The focal hybrid cost moved by 2e-16 relative, because the shared
residual kernel groups the projection the way the *accumulate* did rather
than the way the old cost did; that makes cost and accumulate consistent with
each other, which is the pairing the LM's accept/reject test cares about.

Verification of the two refactors, against the pre-refactor modules loaded out
of `git show HEAD:`:

| | result |
|---|---|
| solvers, all four | counts identical on 400 samples each; p3p, shared-focal and varying-focal **bit-exact**, shift median 2e-13 with a tail to 7e-9 |
| accumulate, all four | residual counts identical; `JtJ` to 4e-16 (calibrated) and 7e-12 (focal, on pixel-scale values) |
| hybrid costs | calibrated **bit-identical**; focal 2e-16 relative |

The shift solver is the one that is not bit-exact: its Gauss-Newton polish of
the quartic amplifies the reassociation that splitting a `fastmath` kernel
into a wrapper plus a core produces. Note that comparing against `HEAD` needs
a **fresh `NUMBA_CACHE_DIR`** - a kernel loaded from the on-disk cache as
object code cannot be cross-inlined, and that alone moved p3p by 1e-12 and the
shift solver by an order of magnitude.

### Remaining

Nothing. All nine estimators run on `device='cuda'`.

---

## 6. Performance notes

Do not re-derive these; they cost a day of measurement.

**The per-round floor is the minimal solver's latency, not launch or readback
overhead.** The 5-point solve kernel, hypotheses in flight against wall clock:

| hypotheses | 1 | 32 | 128 | 512 | 1024 | 4096 |
|---|---|---|---|---|---|---|
| solve (ms) | 1.15 | 4.54 | 4.98 | 4.90 | 4.65 | 5.62 |
| µs per hypothesis | 1151 | 142 | 39 | 9.6 | 4.5 | **1.4** |

One solve on a single thread costs ~1.15 ms and 4096 of them cost 5.62 ms, so
a large round is simply how that latency gets amortized - and nothing about it
depends on the match count. Sampling and scoring are 0.1-2 ms by comparison.

**Round size tracks the iteration budget, not the match count.** Scaling `batch`
by `num_points` sounds right and is wrong: 4096 beat 128 by ~20× equally at 2k,
16k and 50k matches. What does change the answer is adaptive termination - a
round overshoots by up to its own size, so `estimate` ramps from `FIRST_ROUND`
only when `min_iterations < iterations`, and runs at full `batch` otherwise
(the default, since `min_iterations=None` means no adaptive termination).

**Local optimization: one candidate per round.** The gate - improve the best
*minimal* score or inlier count - is what limits the budget. Refining the
round's top-k was measured as pure waste: k = 1, 4, 32 and 128 returned the same
inlier count, the same pose to 1e-4°, and the same wall clock. The gate lands
at **6-37 refinements over a 20000-iteration run**, the same order as the
serial driver, and reaches the same inlier count.

**Reducing host syncs is worth ~10%, not more.** Packing the score reduction
into one array and making the gather/LO readbacks conditional took a round from
~13 syncs to 1 and bought about a tenth. Worth keeping, not worth contorting for.

**Everything above was measured on an RTX A4000 Laptop, which runs float64 at
1/64 of float32.** An A100 is 1/2. Correctness parity transfers (it is a
numerical property); wall-clock does not. In particular the ~5 ms solve latency
is a float64 cost and should fall sharply, which flattens the batch curve and
makes `DEFAULT_BATCH` matter much less. Re-run
`python -m benchmarks.estimators.essential cuda-scaling` on the target hardware
before trusting any tuning constant in this repo.

**How much of the mixed-precision win transfers is the open question, and it
cuts both ways.** Moving the per-point loop to float32 bought **~5x** over an
all-float64 version on this card, and that figure is inflated by its 1/64
float64 rate; an A100 at 1/2 will see less of it. But the inner loop is
`sqrt`- and division-heavy - the cheirality test alone is two square roots and
two divisions per point - and float32 `sqrt` is a single hardware instruction
where float64 `sqrt` is a software sequence on **every** NVIDIA part, not just
this one. So the gain should still exceed the 2:1 FLOPS ratio. Only a run on
the target hardware settles it.

**Shared-focal is the one problem whose round floor is materially higher.**
Its solve kernel is 4.4 us per hypothesis at `batch=4096` against the 5-point's
1.4 us ([§5](#5-per-problem-notes) has the full curve), and that shows up
end to end: 2000 iterations over 3000 matches took 198 / 57 / 34 / 22 ms at
`batch` 128 / 512 / 1024 / 4096, against 57 ms for the CPU driver. Small rounds
are much worse for it than for the other problems, which is the same rule as
everywhere else, only steeper.

**The monodepth problems are the one place the GPU does not clearly win.**
Their solvers are 3-point and cheap, so the CPU driver is already fast: at
3000 matches and 2000 iterations, CPU vs CUDA measured 34/40 ms (calibrated),
33/24 (shift), 32/17 (shared-focal) and 45/22 (varying-focal). The calibrated
P3P variant is *slower* on the GPU. That is the expected shape - 2000
iterations is a single round at `batch=4096`, so nothing amortizes - and it
is a reason to keep `device='cpu'` the default, not a defect.

**No timing was taken for absolute, absolute-focal, fundamental or
varying-focal.** Their solve kernels are all cheaper than the 5-point's, so
the round floor should be lower, but that is a prediction, not a measurement.

---

## 7. State of the checklist

Per problem: solver refactor / solve kernel / scorer / LM / plumbing.

| | refactor | solve | score | LM | `device='cuda'` |
|---|---|---|---|---|---|
| essential | (pre-existing) | ✅ | ✅ | ✅ | ✅ |
| absolute | ✅ 6/6 verified | ✅ | ✅ | ✅ | ✅ |
| absolute-focal | ✅ 5/8 accounted | ✅ | ✅ | ✅ | ✅ |
| fundamental | ✅ 6/6 verified | ✅ | ✅ | ✅ | ✅ |
| varying-focal | ✅ 2/3 accounted | ✅ | ✅ | ✅ | ✅ |
| shared-focal | ✅ bit-exact | ✅ | ✅ | ✅ | ✅ |
| monodepth ×4 | ✅ 3/4 bit-exact | ✅ | ✅ | ✅ | ✅ |

Cross-cutting:

- [x] **`tests/test_cuda.py` rewritten** against the new API and parameterized
      over all ten problems - 71 tests, green. See
      [§7.1](#71-the-shape-of-teststest_cudapy).
- [x] PTX inspected for **all ten** problems - every per-point sqrt and
      division is `.f32`, zero f64 `sqrt`/`div`/`rcp` in every probe
      ([§4](#4-precision-policy) has the table and the two traps).
- [x] Compile check under numba 0.67 (`fp312` env): the whole suite, 170
      passed / 5 skipped. That env had neither pytest nor tqdm; both were
      `pip install`ed (numba 0.67.0 / numpy 2.5.2 pins survived, verified
      after).
- [x] `pyproject.toml` `[tool.setuptools] packages` was missing
      `fastpose.cuda.problems`, so a non-editable install would have shipped a
      backend with no problem modules. Fixed.
- [x] `_cuda_warmup_steps` covers all ten problems, and the degenerate-scene
      bug it turned up is in [§3.6](#36-step-6--plumbing).
- [ ] `cuda-scaling` benchmark modes for the new problems
- [ ] README backend section and module layout (it still describes
      `cuda/solvers.py` / `cuda/scorers.py` / `cuda/refiners.py`)
- [ ] Timing / speedup numbers for absolute, absolute-focal, fundamental and
      varying-focal (shared-focal's and monodepth's are in
      [§6](#6-performance-notes) and [§8](#8-what-to-do-next))

### 7.1 The shape of `tests/test_cuda.py`

Worth knowing before touching any problem, because adding one is a single
function.

A `Case` carries everything that differs between problems: the scene, the
`data` tuple the CPU references take, the `cols` the device kernels take
(these differ only for the two focal problems, whose principal points ride in
`params` on the GPU and inside `data` on the CPU), `max_error_sq`, the CPU
solver / scorer / refiner, and an `error(model)` for the clean-scene
assertion. One builder function per problem returns a `Case`; `BUILDERS` maps
name to builder and `PROBLEMS` drives the `@pytest.mark.parametrize` on all
five kernel-level tests:

    test_cuda_solver_matches_cpu_solver        counts exact, then
                                               case.check_solver_models
    test_cuda_solver_is_exact_on_a_clean_scene best model error < 1e-6
    test_cuda_scorer_matches_cpu_scorer        counts +-1, score rtol 1e-5,
                                               selection as good as the best
    test_cuda_lm_matches_cpu_lm                basin agreement, then params
                                               rtol 1e-3/atol 1e-4 there
    test_cuda_lm_improves_the_cost_it_minimizes

plus `test_cuda_estimator_matches_cpu_estimator`, parameterized the same way,
which is the end-to-end smoke for all nine entry points, and two warmup tests
that assert `_cuda_warmup_steps` names every problem and that running them
leaves both LM kernels built. The driver tests (reproducibility, batch, the
one-candidate-per-round gate, the failure path) stay on
`estimate_relative_pose` - they test `cuda/ransac.py`, which is
problem-agnostic.

Two per-case hooks, both of which exist because a blanket tolerance would
have asserted nothing. Override either only with a measurement in hand.

`check_solver_models(cpu, gpu)` defaults to `assert_allclose` at 1e-6.
Shared-focal overrides it, because its CPU/GPU difference is too heavy-tailed
for any elementwise bound to be meaningful ([§5](#5-per-problem-notes)); the
exact model-count comparison stays either way.

`lm_cost` defaults to `cpu_score` and is **what the LM actually minimizes**.
For eight of the ten problems those are the same functional; for monodepth
the LM minimizes the hybrid cost while the scorer computes the Sampson error
alone, so asking the refiner to improve the score was asking for the wrong
thing - and monodepth-shift failed on exactly that before the hook existed.

### 7.2 Why `test_cuda_lm_matches_cpu_lm` does not assert "never worse"

It used to, per candidate. That is not a true property of these costs and the
monodepth port is where it broke: under numba 0.65 the GPU converged 0.7%
*below* the CPU on monodepth-shared-focal candidate 5, and under 0.67 0.8%
*above*.

Before loosening anything, the question worth answering is whether that is
the GPU's jacobian or the cost surface. It is the surface: perturbing that
candidate's **initial model** by 1e-7 relative and re-running the **CPU**
refiner alone lands in ~25 distinct minima spanning 3.2% in cost, while
candidates 3 and 8 hold a single basin to 4e-9. A minimal sample can put the
LM on a start from which the descent is chaotic, and there float32-versus-
float64 arithmetic simply picks a different minimum.

So the test asserts the three things that *are* true, each of which a real
jacobian or precision bug would break:

- at least 75% of candidates reach the same basin (costs within 1e-6
  relative) - a wrong jacobian moves nearly all of them;
- the parameters match to rtol 1e-3 / atol 1e-4 **on those candidates**;
- the median relative cost difference is <= 1e-6 and the worst is <= 5%, the
  latter chosen outside the 3.2% spread the CPU alone shows on a chaotic
  start and far inside anything a broken jacobian produces.

To add a problem: write `_<name>_case(seed, n, noise, outliers)` and add one
entry to `BUILDERS`. Nothing else.

Three things the rewrite fixed rather than transcribed, all of which had made
the old file weaker than it looked:

- it passed the **float64** columns to the scorer and the LM. Those kernels
  are built with `real=float32`, so float64 inputs promoted the whole
  per-point chain back to double and the mixed-precision path under test was
  never exercised. `device_columns` now returns both and the tests pass the
  float32 ones.
- `test_cuda_driver_refines_at_most_one_candidate_per_round` monkeypatches
  `ransac_module.refine_prepared`, whose signature gained a leading `problem`.
- `RefineBuffers` now takes `(max_candidates, num_points, num_params)`, and
  `refine_prepared` reads `buffers.init_models` rather than gathering, so the
  LM tests `copy_to_device` the candidates directly.

---

## 8. What to do next

**Every estimator is ported.** What is left is not porting work, and neither
item is blocked on anything:

**1. `cuda-scaling` benchmark modes** for the eight problems that lack them,
and timing numbers for absolute, absolute-focal, fundamental and
varying-focal. Follow `benchmarks/estimators/essential.py`, and read
[§6](#6-performance-notes) first rather than re-deriving the round policy.

**2. README** - its backend section still describes `cuda/solvers.py`,
`cuda/scorers.py` and `cuda/refiners.py`, which have not existed for three
sessions, and it still says the CUDA backend covers calibrated relative pose
only.

Both are worth doing on the **A100**, not here: every number in this document
came off an RTX A4000 Laptop, which runs float64 at 1/64 of float32 against
the A100's 1/2 ([§6](#6-performance-notes)). Correctness parity transfers;
wall-clock does not, and the float64-heavy solve kernels are exactly what
should improve most.

### Smoke results to reproduce

Each of these was CPU vs CUDA on the same seed, and is the quickest way to tell
whether a change broke something. Same-seed agreement is *not* expected in
general (the drivers draw different samples), but on these scenes it held:

| problem | CPU | CUDA |
|---|---|---|
| essential, 4000 matches, 2000 it | 2800 inliers, rot 0.0159°, score 0.005137 | identical |
| absolute, 3000 pts, 2000 it | 2100 inliers, rot 0.00342°, score 0.00913599 | identical |
| absolute-focal, 3000 pts, 2000 it | 2100 inliers, rot 0.00339°, f 1199.909 | identical |
| fundamental, 4000 matches, 2000 it | 2801 inliers, score 5135.07 | 2801 inliers, score 5137.59; F agrees to 4.7e-5 |
| varying-focal, 3000 matches, 2000 it | 2107 inliers, score 8369.54 | 2106 inliers, score 8366.94 |
| shared-focal, 3000 matches, 2000 it | 2106 inliers, rot 0.0159°, f 1000.675, score 3834.78, 57 ms | 2106 inliers, rot 0.0160°, f 1000.822, score 3834.95, 22 ms at batch 4096 |
| monodepth, 3000 matches, 2000 it | 2106 inliers, rot 0.0054°, scale 1.2502, score 0.00385, 34 ms | 2107 inliers, rot 0.0191°, scale 1.2505, score 0.00386, 40 ms |
| monodepth-shift | 2106 inliers, rot 0.0189°, scale 1.2502, 33 ms | 2107 inliers, rot 0.0206°, scale 1.2509, 24 ms |
| monodepth-shared-focal | 2106 inliers, rot 0.0025°, f 1000.72, score 3853.39, 32 ms | 2107 inliers, rot 0.0063°, f 1002.20, score 3857.80, 17 ms |
| monodepth-varying-focal | 2106 inliers, rot 0.0071°, f1 1000.97, f2 1000.34, 45 ms | 2107 inliers, rot 0.0062°, f1 1002.17, f2 1002.23, 22 ms |

The monodepth scenes carry 1% multiplicative depth noise, under which the two
*shifts* are only weakly identifiable - both backends recover them to ~0.05
against a ground truth of (-0.05, +0.03), and they differ from each other by
about as much. That is the scene, not the port:
`test_cuda_solver_is_exact_on_a_clean_scene[monodepth-shift]` recovers both
shifts to better than 1e-6 when the depths are exact.

Exact-pose recovery on clean, noiseless scenes was also checked for essential,
absolute, absolute-focal and fundamental (all inliers, rotation error < 1e-5°).
`test_cuda_solver_is_exact_on_a_clean_scene` now covers this for all ten as a
test - fundamental through the max algebraic residual rather than a pose
error, and the monodepth problems through the scale and shifts or focals too.

`pytest tests/` is green at **175 passed** (104 CPU + 71 CUDA) on numba 0.65 /
python 3.10, and at **170 passed, 5 skipped** on the `fp312` env (numba 0.67 /
python 3.12) - the 5 skips are the poselib comparison tests, and poselib is
not installed there. Run it with

    $env:PYTHONPATH="D:\Research\code\fastpose\src;D:\Research\code\fastpose"
    conda run -n fp312 python -m pytest tests/ -q

fastpose is not installed into `fp312`, hence the `PYTHONPATH`. `pytest` and
`tqdm` were `pip install`ed there; do **not** `conda install` into that env,
which exists to pin numba 0.67 / numpy 2.5.2 and where a resolver run can
quietly move them. Verify the pins after touching it.
