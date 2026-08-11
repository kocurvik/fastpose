"""Process-independent numba cache keys for closure-specialized kernels.

Both engines in this package specialize a driver by closing over the problem
kernels as compile-time constants: `build_ransac` (estimators/ransac.py) and
`build_lm_refine` (refiners/lm.py), plus the loss-parametrized factories in
scorers/ and refiners/. That is what lets numba bind the calls statically -
but it interacts badly with `cache=True`.

Numba's on-disk cache key for a function with a closure is a hash of the
*pickled closure cell contents* (`Cache._index_key` in numba/core/caching.py).
Cell contents here are other Dispatchers, and a Dispatcher pickles through
`_MemoMixin._uuid`, which is lazily generated as a fresh `uuid4()` per
process. So the key of every closure-specialized kernel differed on every
run: the cache never hit, `fastpose-warmup` bought nothing, and each run
appended one more dead entry to the on-disk index (which is how the
__pycache__ directories accumulated dozens of .nbc files for a single
kernel).

`stabilize` fixes that by assigning each kernel a deterministic uuid derived
from its module, qualified name and - recursively - the identity of every
kernel it closes over. Two specializations built from the same factory (say
the truncated-loss and Cauchy-loss variants of an accumulate kernel) differ
in their cell contents and therefore still get different uuids, which is what
keeps them from colliding in the cache; identical specializations rebuilt in
a later process get the same uuid and hit.

This leans on two numba internals (`Dispatcher._set_uuid` and the name-mangled
`_MemoMixin__uuid`), so everything here is best-effort: if numba changes shape
the kernels are simply left with their per-process uuids and we fall back to
recompiling every run - slow, like before, but never wrong. A stale or
mismatched key can only ever cause a cache *miss*, because numba still
verifies the signature and the function bytecode before reusing an entry.
"""

import hashlib

# name-mangled attribute of numba.core.dispatcher._MemoMixin; read directly so
# we can tell "not yet assigned" from "already lazily assigned a random uuid"
# without going through the property (which would assign one)
_UUID_ATTR = '_MemoMixin__uuid'
_PREFIX = 'fastpose:'

# cell types whose repr is a faithful, stable identity
_LITERALS = (bool, int, float, complex, str, bytes, type(None))


class _Unstable(Exception):
    # a closure cell we cannot identify deterministically
    pass


def _is_kernel(obj):
    return hasattr(obj, 'py_func') and hasattr(obj, '_set_uuid')


def _digest(parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode('utf-8'))
        h.update(b'\x00')
    return h.hexdigest()


def _tag(value, seen):
    if _is_kernel(value):
        return _pin(value, seen)
    if isinstance(value, _LITERALS):
        return repr(value)
    raise _Unstable(type(value).__name__)


def _pin_referenced_globals(py_func, seen):
    # the pickled form of a kernel embeds the kernels it calls as globals, so
    # those need deterministic uuids too or the enclosing hash stays unstable
    globals_ = py_func.__globals__
    for name in py_func.__code__.co_names:
        value = globals_.get(name)
        if _is_kernel(value):
            _pin(value, seen)


def _pin(kernel, seen):
    existing = getattr(kernel, _UUID_ATTR, None)
    if existing is not None:
        return existing
    if id(kernel) in seen:
        # recursive kernel: break the cycle, the qualified name identifies it
        py_func = kernel.py_func
        return '%s.%s' % (py_func.__module__, py_func.__qualname__)
    seen.add(id(kernel))

    py_func = kernel.py_func
    parts = ['%s.%s' % (py_func.__module__, py_func.__qualname__)]
    # closure cells are what distinguish two specializations built by the same
    # factory, so they carry the identity; globals are resolved by name and
    # only need to be pinned, not hashed
    cells = py_func.__closure__ or ()
    for name, cell in zip(py_func.__code__.co_freevars, cells):
        parts.append('%s=%s' % (name, _tag(cell.cell_contents, seen)))

    uuid = _PREFIX + _digest(parts)
    kernel._set_uuid(uuid)
    _pin_referenced_globals(py_func, seen)
    return uuid


def stabilize(*kernels):
    # give each kernel (and everything it closes over or calls) a uuid derived
    # from what it is rather than from when it was built, so that cache=True on
    # an enclosing closure produces the same key in every process
    try:
        seen = set()
        for kernel in kernels:
            if _is_kernel(kernel):
                _pin(kernel, seen)
    except Exception:
        # best-effort only: any kernel left unpinned just keeps its per-process
        # uuid, which costs a recompile and never returns wrong code
        pass
