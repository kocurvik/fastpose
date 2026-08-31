# Porting the CUDA backend to the remaining estimators

`fastpose.cuda` currently covers **calibrated relative pose (5-point)** only.
This document is the recipe for extending it to the other eight estimators,
written from what the first port actually ran into rather than from what it
looked like it would need.

Read [§1](#1-the-four-hard-constraints) before writing any device code — three
of those four constraints have no workaround and they shape every design
decision downstream.

---

## 0. Current state

| Estimator | Solver | CUDA solve | CUDA score | CUDA LM | Notes |
|---|---|---|---|---|---|
| `estimate_relative_pose` | 5-point | ✅ | ✅ | ✅ | reference implementation |
| `estimate_fundamental` | 7-point | — | — | ⚠️ | LM blocked, see [§5](#5-per-problem-notes) |
| `estimate_absolute_pose` | P3P | — | — | — | easiest next |
| `estimate_absolute_pose_with_focal` | P4Pf | — | — | — | |
| `estimate_relative_pose_with_varying_focals` | 7-point + Rybkin | — | — | — | |
| `estimate_relative_pose_with_shared_focal` | 6-point | — | — | — | ⚠️ 29 KB/thread scratch |
| `estimate_*_with_monodepth` (×3, 4 solvers) | 3-point ×4 | — | — | — | hybrid cost |

The pieces that are already problem-agnostic and should be reused, not
rewritten: `cuda/backend.py`, the reduction helpers and damped-Cholesky solve
in `cuda/refiners.py`, and most of `cuda/ransac.py`.

---

## 1. The four hard constraints

These are properties of numba's CUDA target, verified on numba 0.65. The first
three have no workaround.

### 1.1 No `np.empty` and no `.reshape` in device code

Numba's runtime (NRT) is host-only. Both fail at link time:

```
error : Undefined reference to 'numba_attempt_nocopy_reshape' in '<cudapy-ptx>'
```

Every kernel that allocates or reshapes must be restructured so scratch is
**passed in already shaped**. This is the single most invasive change and it is
why `_solve_5pt` was split (see [§3.1](#31-step-1--refactor-the-solver-into-a-factory)).

### 1.2 A device function cannot return a `cuda.local.array`

The tempting shim — `alloc9()` returning `cuda.local.array(9, float64)` so one
source works for both backends — does **not** compile. Scratch must be
allocated in the *kernel* and passed down as arguments.

### 1.3 Shared memory is 48 KB per block by default

Budget it before choosing a thread count. The LM kernel's reduction array alone
is `LM_THREADS × NUM_ACC` doubles; at 128 threads and 20 accumulators that is
20.5 KB. See [§3.4](#34-step-4--the-lm-refiner) for the arithmetic.

### 1.4 `syncthreads()` must be block-uniform

Every branch containing a `cuda.syncthreads()` must be taken by all threads of
the block or none. The LM loop achieves this by computing all control decisions
in thread 0, publishing them to a shared `ctl` array, and syncing before any
thread reads them. Violating this deadlocks or corrupts silently — it does not
raise.

### What *does* work (verified, don't re-derive)

- Tuples of device arrays as kernel arguments, and unpacked inside device functions.
- `cuda.local.array` with a shape from a module-level constant.
- The same Python function object compiled by both `njit` and `cuda.jit(device=True)`.
- `cuda.jit(cache=True)` — the on-disk cache works, so `fastpose-warmup` extends.
- Module-level numpy arrays read as constants inside device functions (e.g. `_T44`).
- `math.isfinite`, `math.sqrt`, `math.copysign`. **Not** `np.inf` as a literal.
- 2-D shared arrays, `float64`/`int64` reductions.

---

## 2. The pattern: one source, two backends

Do not write a second copy of any solver's maths. The repo's existing idiom —
closure-specialized factories (`build_ransac`, `build_lm_refine`) — extends to
the backend choice.

A kernel is written against the shim in `src/fastpose/jit_backend.py`:

```python
def build_five_point_kernels(jit):          # jit(fastmath=..., inline=...)
    @jit(fastmath=True, inline=True)
    def _poly1_mul_acc(a, b, factor, out):
        ...
    return {'solve_5pt_core': _solve_5pt_core, ...}

_CPU = build_five_point_kernels(cpu_jit)    # njit(cache=True, ...)
```

and `fastpose/cuda/solvers.py` builds the same factory with `cuda_jit`
(`cuda.jit(device=True, ...)`). Two rules:

- **Every callee must also come from the factory.** A kernel that calls an
  `njit` global will not compile for CUDA. This is why
  `fill_epipolar_matrix` moved into `build_fill_epipolar_matrix(jit)`.
- **Float literals are float64 in numba.** A single bare `1.0` in an otherwise
  float32 expression silently promotes the whole chain back to double and
  undoes the mixed precision. Factories that run per point take a `real=`
  parameter and write `real(1.0)`. See `build_sampson_point_kernels`.

### Cache keys

`kernel_cache.stabilize` hashes closure cells to give `cache=True` a
process-independent key. It understands kernels, literals, and (since the
mixed-precision work) numba type objects. If you add a closure cell of some
other type, `_tag` raises `_Unstable`, which is swallowed — and the effect is
silent: every kernel that reaches it loses its on-disk cache and recompiles
every run. If `fastpose-warmup` stops helping, look here first.

---

## 3. Step-by-step for one problem

Do these in order and **verify at each step**. The parity tests are what make
this tractable; without them a precision or indexing bug surfaces only as a
slightly worse pose 2000 lines later.

### 3.1 Step 1 — refactor the solver into a factory

Purely mechanical, and CPU behaviour must not change.

1. Wrap all `@njit` kernels of `solvers/<problem>.py` in
   `def build_<problem>_kernels(jit):`, indenting by 4.
2. Rewrite the decorators:
   `@njit(cache=True, fastmath=True)` → `@jit(fastmath=True)`,
   `@njit(cache=True, inline='always')` → `@jit(inline=True)`, etc.
3. Split the top-level `_solve_*` into:
   - `_solve_*_core(data, sample, models, <every scratch array>)` — inside the
     factory, no allocation, no reshape;
   - a thin CPU-only `_solve_*(data, sample, models, workspace)` wrapper
     outside it that slices and reshapes the flat workspace and calls the core.
     This preserves the RANSAC driver's contract exactly.
4. Instantiate `_CPU = build_<problem>_kernels(cpu_jit)` and rebind every
   module-level name other modules import (`_pose_from_essential`,
   `_real_roots_sturm`, … — check with `grep -rn "from fastpose.solvers.<problem> import"`).

**Verify.** The indentation is done by script, so check the maths survived:
extract each function's statements (stripping indentation, comments and
decorators) from `git show HEAD:<file>` and from the new file, and assert they
are equal. For the split `_solve_*`, concatenate the wrapper preamble and the
core body and compare against the old body. The 5-point port came out at 14/15
functions byte-identical and 71/71 statements for the split one. Then run the
full CPU suite — it must be green before you go further.

### 3.2 Step 2 — the CUDA solve kernel

One thread per hypothesis. The solvers are scalar and branchy; there is no
useful parallelism inside one minimal sample.

```python
@cuda.jit(cache=True)
def _solve_batch_kernel(x1_x, x1_y, x2_x, x2_y, samples, models, counts):
    i = cuda.grid(1)
    if i >= samples.shape[0]:
        return
    A = cuda.local.array((5, 9), float64)
    ...                                   # one per scratch piece
    counts[i] = _solve_core((x1_x, x1_y, x2_x, x2_y), samples[i], models[i],
                            A, N, ...)
```

Local memory is per-thread storage that the hardware interleaves across a warp,
so thread-uniform access coalesces for free — no manual layout needed. Keep the
solver in **float64** (see [§4](#4-precision-policy)).

**Verify.** Run the CPU `_solve_*` and the kernel on *identical* samples.
Model counts must match exactly (the solver's branches are on tolerances far
above codegen noise); models should agree to ~1e-6. A count mismatch means a
real bug, not rounding.

### 3.3 Step 3 — the scorer

One block per hypothesis; threads stride over correspondences and tree-reduce
`(score, inlier count)` in shared memory. The block loops over the several
models one sample yields, keeping its running best in thread 0's registers, so
the host reads back five scalars per hypothesis rather than a
`batch × max_models` table.

The per-point maths is **not** rewritten — it comes from the same factory the
CPU scorer uses (`build_sampson_point_kernels`, or the reprojection equivalent
you will need to add for the absolute-pose problems). Only the reduction is new.

Two properties to preserve, both already true of `cuda/scorers.py`:

- Write `NO_MODEL` (1e300) and index `-1` into unused model slots.
- Form the model-derived matrix (E, F) in **float64** from the float64 model and
  round it once, then run the per-point loop in float32.

There is no early bail-out on the GPU — a block reduction cannot have one. This
is the same trade `build_parallel_ransac` documents: a model the CPU driver
abandons mid-scan comes back with a *full* inlier count, which can change which
candidate local optimization sees.

**Verify.** Against the CPU scorer with `best_score=1e300` (which disables its
bail-out). Assert inlier counts within 1, scores to `rtol=1e-5`, and — more
useful than an index match — that the model the GPU *selected* scores within
1e-5 of the true best. Two models can tie within rounding; either pick is right.

### 3.4 Step 4 — the LM refiner

One block per candidate, and the **entire LM loop inside one kernel launch**.
Launching an accumulate kernel per LM step would pay launch latency 25 times per
candidate and dominate everything.

Split by cost:

- **O(n)** — relaxed inlier mask, cost, normal equations → across the block's
  threads, tree-reduced.
- **O(1)** — damped Cholesky, retraction, damping schedule → thread 0, published
  through shared memory.

Reduce all normal-equation accumulators in **one pass**: pack the upper triangle
of `JtJ` plus `Jtr` into `NUM_ACC = T(T+1)/2 + T` doubles per thread, where
`T = NUM_TANGENT`. Shared budget:

```
red      LM_THREADS × NUM_ACC × 8 B      (128 × 20 × 8 = 20.5 KB for T=5)
+ state, jacobian basis, JtJ/Jtr/A/delta, reduction scalars   ≈ 3 KB
                                        must stay under 48 KB
```

`T` grows for other problems (6 for absolute, 7 for absolute-focal / varying
focal / fundamental, up to 9 for monodepth). At `T=9`, `NUM_ACC = 54` and 128
threads needs 55 KB — **over budget**. Either drop to 64 threads (27 KB) or do a
warp-level reduction first (`cuda.shfl_down_sync`) and stage only 32 partial
sets in shared. Compute this before choosing `LM_THREADS`.

Reuse `_reduce_sum`, `_reduce_scalar`, `_reduce_count` and `_solve_damped5`
(generalize the latter's hard-coded `NUM_TANGENT`).

Match the CPU semantics exactly: select the relaxed inlier subset **once** from
the initial model at `LO_INLIER_SCALE × max_error_sq` with whatever check the
CPU refiner applies, then minimize over that fixed subset. Carry the subset as a
per-candidate `uint8` mask — device code cannot compact into new arrays.

Build the kernel through `build_lm_refine_kernel(loss)` so the RANSAC-internal
(truncated) and final-polish (Cauchy) passes share one source.

**Verify.** Two assertions, and the second is the one that matters:

1. Parameters against the CPU refiner, `rtol=1e-3, atol=1e-4` (float32
   jacobians move the converged pose at that level).
2. **Minimizer quality**: scored back in full float64, the GPU result must be no
   worse than the CPU refiner's by more than ~1e-4 relative. An element-wise
   tolerance alone cannot tell "rounded differently" from "converged somewhere
   worse"; this can.

### 3.5 Step 5 — the driver

`cuda/ransac.py` is nearly problem-agnostic. To generalize it, parameterize:

- the `data` tuple width (4 columns for epipolar, 6+2 for monodepth, 2D+3D for
  absolute pose) — the kernels take columns positionally today;
- `SAMPLE_SIZE`, `NUM_PARAMS`, `MAX_MODELS`;
- the number of float32 mirror columns to maintain.

Everything else — the round loop, the gate, adaptive termination, the ramp, the
packed readback — is shared. Do **not** re-derive the round policy; see
[§6](#6-performance-notes).

### 3.6 Step 6 — plumbing

- `device='cuda'` on the estimator entry point, validating with a clear
  `ValueError`, plus a module-level estimator cache keyed on `batch` (it owns
  the device buffers; recreating it per call reallocates and re-warms).
- Route the final polish through `CudaRansacEstimator.final_refine`, which is
  the same LM kernel at `relaxed_scale=1.0` with the Cauchy loss. At large `n`
  this pass is O(n) per step for 100 steps and will dominate if left on the CPU.
- Extend `_cuda_warmup_steps` in `estimators/warmup.py`. A cold GPU estimate
  compiles the solver, the scorer and **two** LM kernels (one per loss); that is
  seconds, and it is easy to mistake for steady-state cost when benchmarking.
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
result — ~3 digits of cancellation, and fp16 has 3.3 total. Worse,
`max_error_sq` (4e-6) is *subnormal* in fp16, so the entire inlier decision
happens where fp16 has no precision left. Measured: 3.5% of the inlier set wrong
(Jaccard 0.964, up to 102 points), against 0 for float32. Rescaling overflows
fp16's 65504 ceiling. Do not revisit.

**Verify the precision actually took effect.** A stray float64 literal is
silent. Inspect the PTX:

```python
for sig, ptx in kernel.inspect_asm().items():
    print(len(re.findall(r"\.f32", ptx)), len(re.findall(r"\.f64", ptx)))
```

What matters is not the overall ratio but that every **per-point** `sqrt` and
division is `.f32`. In the shipped kernels: score 2 f32 sqrt / 0 f64; LM 3 f32
sqrt (two from cheirality, one from the jacobian) / 0 f64 in the inner loop.
Remaining f64 instructions are the O(1) scaffolding, which is correct.

---

## 5. Per-problem notes

Ordered easiest first.

### Absolute pose (P3P) — start here
`sample_size=3, num_params=12, max_models=4, workspace=50`. Small scratch, no
eigen-decomposition, `NUM_TANGENT=6`. Needs a new **reprojection** scorer
(`scorers/reprojection.py` factored the same way as `build_sampson_point_kernels`)
and its analytic jacobian. The cleanest second port and the one that will prove
the driver generalization.

### Absolute pose with focal (P4Pf)
`sample_size=4, num_params=13, max_models=1, workspace=454`. Reuses the Sturm
machinery you will already have ported from the 5-point solver. `NUM_TANGENT=7`.

### Fundamental (7-point) — ⚠️ LM blocked
Solver and scorer are the easiest of all (`workspace=93`, `max_models=3`,
`SampsonScorer` with no cheirality). **But** `refiners/fundamental.py` uses
`svd_init_state`, which calls `np.linalg.svd` — unavailable in device code. It
is the only refiner that does. Options, in order of preference:

1. Write a device 3×3 SVD (one-sided Jacobi, ~50 lines, well conditioned at this
   size) and add it to the factory.
2. Reparametrize F away from the `U diag(1,σ,0) Vᵀ` factorization.
3. Ship solve+score on GPU and leave the LM on the CPU — acceptable only at
   small `n`, since the LM is O(n) per step.

### Varying focal (7-point + Rybkin)
`num_params=14, max_models=12, workspace=122, NUM_TANGENT=7`. Needs
`model_to_fundamental` and `calibrate_epipolar` moved into the shared point-kernel
factory. The scorer has **no** cheirality check (poselib scores an F here) —
keep that difference.

### Shared focal (6-point) — ⚠️ occupancy
`workspace_size=3662` doubles = **29 KB of local memory per thread**. That is
far beyond the 6.9 KB the 5-point solver uses and will wreck occupancy; expect
the solve kernel to be latency-bound to a degree the 5-point one is not.
Measure a single-hypothesis solve before committing to the port. Also note this
is the one solver where `fastmath` is deliberately **off** for conditioning —
preserve that, and be aware NVVM may reassociate differently than LLVM anyway,
so check the recovered focal against the CPU path specifically.

### Monodepth (×4)
Four solvers sharing one 8-column data layout (x1, y1, x2, y2, d1, d2, plus two
hybrid weights) — the driver's data handling needs generalizing first. The
refiner minimizes a **hybrid** cost (truncated Sampson + truncated symmetric
reprojection), so the LM kernel needs both residual families in its accumulate.
`NUM_TANGENT` is 7/9/8/9 — at 9 the reduction exceeds the shared-memory budget
at 128 threads (see [§3.4](#34-step-4--the-lm-refiner)).

---

## 6. Performance notes

Do not re-derive these; they cost a day of measurement.

**The per-round floor is the minimal solver's latency, not launch or readback
overhead.** One 5-point solve on a single GPU thread costs ~1 ms, and the whole
kernel costs ~5 ms with 4096 hypotheses in flight — 4.54, 4.98, 4.90, 4.65 and
5.62 ms at 32, 128, 512, 1024 and 4096. That is 142 µs per hypothesis at 32 and
1.4 µs at 4096. Large rounds are how that latency is amortized.

**Round size tracks the iteration budget, not the match count.** Scaling `batch`
by `num_points` sounds right and is wrong: 4096 beat 128 by ~20× equally at 2k,
16k and 50k matches. What does change the answer is adaptive termination — a
round overshoots by up to its own size, so `estimate` ramps from `FIRST_ROUND`
only when `min_iterations < iterations`, and runs at full `batch` otherwise
(the default, since `min_iterations=None` means no adaptive termination).

**Local optimization: one candidate per round.** The gate — improve the best
*minimal* score or inlier count — is what limits the budget. Refining the
round's top-k was measured as pure waste: k = 1, 4, 32 and 128 returned the same
inlier count, the same pose to 1e-4°, and the same wall clock.

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

---

## 7. Checklist per problem

- [ ] Solver refactored into a factory; CPU bodies verified unchanged; full CPU suite green
- [ ] CUDA solve kernel; **model counts match the CPU solver exactly**
- [ ] Scorer reduction; inlier counts within 1, selection quality within 1e-5
- [ ] LM kernel; shared-memory budget computed for this `NUM_TANGENT`; parity **and** minimizer-quality asserted
- [ ] PTX inspected: per-point `sqrt`/division are `.f32`
- [ ] Driver generalized for the data layout; no new round-policy invention
- [ ] `device='cuda'` plumbed, estimator cached on `batch`, final polish on device
- [ ] Warmup step added (compiles solver + scorer + **two** LM kernels)
- [ ] `cuda-scaling` benchmark mode added
- [ ] End-to-end tests: exact-pose, agrees-with-CPU, info fields, reproducibility from a seed, total-failure path
- [ ] README backend section and this table updated
