# Tests and Benchmarks

## Running tests

```
python -m pytest tests
```

Covers finite-difference checks of the refiner jacobians (`tests/jac`) and
end-to-end estimator tests on synthetic scenes.

`tests/test_cuda.py` checks the CUDA backend against the CPU one kernel by
kernel — the batched 5-point solver, the block-reduction scorer, the
block-parallel LM, and the end-to-end estimator. The whole module skips when
no usable CUDA device is present, so the suite still runs on CPU-only
machines.

The parity tolerances there are not arbitrary. Solver and refiner kernels are
compiled from the same source for both backends, but NVVM and LLVM contract
and reassociate `fastmath` expressions differently, so agreement is to a few
ulps rather than bit-exact; the scorer additionally compares a GPU tree
reduction against a sequential CPU sum. Inlier counts are integers and are
asserted to match exactly.

## Running synthetic benchmarks

```
python -m benchmarks.estimators.fundamental            # mAA vs runtime plot
python -m benchmarks.estimators.essential
python -m benchmarks.estimators.absolute
python -m benchmarks.estimators.fundamental scaling    # runtime scaling table
python -m benchmarks.estimators.essential scaling
python -m benchmarks.estimators.essential cuda-scaling  # CPU vs GPU, matches x iterations
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
```

`benchmarks/estimators` compares against poselib (mAA + runtime scaling) on
synthetic scenes; `benchmarks/solvers` evaluates solver accuracy on synthetic
noise-free minimal samples. Shared metrics, data generation and plotting live
in `benchmarks/utils.py`. The `scaling` / `*-scaling` variants print a runtime
table instead of producing a plot.

(First call JIT-compiles the driver for a few seconds; the benchmarks warm up
before timing.)
