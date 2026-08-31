"""CUDA backend: batch-parallel LO-RANSAC on the GPU.

The CPU engine draws one hypothesis at a time (or a small batch across numba
threads) and is bounded by how fast a single core can solve and score. This
backend instead keeps thousands of hypotheses in flight: a round of `batch`
minimal samples is solved one-thread-per-hypothesis, scored one-block-per-
hypothesis with the block reducing over all correspondences, and locally
optimized one-block-per-candidate. Only a handful of scalars cross the PCIe
bus per round.

That makes it a win in exactly the regime the CPU driver is worst at - many
iterations and many matches - and a loss on small problems, where a round of
kernel launches (~30-60us) costs more than the few hundred microseconds the
CPU driver would have needed in total.

What is shared with the CPU path and what is not
------------------------------------------------
The minimal-solver math is *the same source*: `solvers/essential.py` builds
its kernels through `build_five_point_kernels(jit)` and this package
instantiates that factory with `cuda.jit(device=True)` instead of `njit`.
There is no second copy of the Sturm/Danilevsky chain to keep in sync.

Two things could not be shared, both because numba's runtime is host-only, so
neither `np.empty` nor `.reshape` compiles in device code:

- scratch is passed in pre-shaped (see the note in `solvers/essential.py`);
  the CUDA kernel allocates it with `cuda.local.array`, which the hardware
  interleaves across threads, so the accesses coalesce for free.
- the scorer and the LM accumulate are *reductions* here, not serial loops,
  so they are written fresh in `cuda/scorers.py` and `cuda/refiners.py`
  rather than re-decorated. They are checked against the CPU kernels
  point-for-point in `tests/test_cuda.py`.

Availability
------------
`is_available()` reports whether a usable CUDA device is present; `require()`
raises with the reason if not. Import of this package never fails and never
initializes a context, so `import fastpose` stays cheap on machines with no
GPU.
"""

_UNAVAILABLE_REASON = None
_AVAILABLE = None


def _probe():
    global _AVAILABLE, _UNAVAILABLE_REASON
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        from numba import cuda
    except ImportError as exc:  # pragma: no cover - numba is a hard dependency
        _AVAILABLE, _UNAVAILABLE_REASON = False, f"numba.cuda unimportable: {exc}"
        return False
    try:
        if not cuda.is_available():
            _AVAILABLE = False
            _UNAVAILABLE_REASON = (
                "numba.cuda.is_available() is False - no CUDA driver or no "
                "visible device")
            return False
        # is_available() only checks the driver; touching a device is what
        # catches a driver/toolkit mismatch or an already-exhausted GPU
        cuda.current_context()
    except Exception as exc:
        _AVAILABLE, _UNAVAILABLE_REASON = False, f"{type(exc).__name__}: {exc}"
        return False
    _AVAILABLE, _UNAVAILABLE_REASON = True, None
    return True


def is_available():
    # True when a CUDA device can actually be used, not merely imported
    return _probe()


def unavailable_reason():
    # human-readable reason is_available() returned False, or None
    _probe()
    return _UNAVAILABLE_REASON


def require():
    # raises RuntimeError with the probe's reason; used by the device='cuda'
    # entry points so a missing GPU fails with something actionable
    if not _probe():
        raise RuntimeError(
            "device='cuda' requires a working CUDA device: "
            f"{unavailable_reason()}")
