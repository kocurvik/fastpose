"""Lazy name -> `CudaProblem` lookup.

Importing a problem module builds its device-function factories, which is
what makes numba compile. Keeping the mapping lazy is what lets
`import fastpose` stay cheap and lets a machine without a GPU import
`fastpose.cuda` at all.
"""

import importlib

# problem name -> module under fastpose.cuda.problems. The names are the ones
# the estimator entry points and `fastpose-warmup --problem` already use.
_MODULES = {
    'essential': 'essential',
    'absolute': 'absolute',
    'absolute-focal': 'absolute_focal',
    'fundamental': 'fundamental',
    'homography': 'homography',
    'varying-focal': 'varying_focal',
    'shared-focal': 'shared_focal',
    'monodepth': 'monodepth',
    'monodepth-shift': 'monodepth',
    'monodepth-shared-focal': 'monodepth',
    'monodepth-varying-focal': 'monodepth',
}

# problems whose module exposes several PROBLEM objects keyed by name
_ATTRS = {
    'monodepth': 'PROBLEM_CALIBRATED',
    'monodepth-shift': 'PROBLEM_SHIFT',
    'monodepth-shared-focal': 'PROBLEM_SHARED_FOCAL',
    'monodepth-varying-focal': 'PROBLEM_VARYING_FOCAL',
}

_CACHE = {}


def problem_names():
    return tuple(_MODULES)


def get_problem(name):
    if name not in _CACHE:
        try:
            module_name = _MODULES[name]
        except KeyError:
            raise ValueError(
                "unknown CUDA problem %r; expected one of %s"
                % (name, ", ".join(_MODULES))) from None
        module = importlib.import_module(
            'fastpose.cuda.problems.' + module_name)
        _CACHE[name] = getattr(module, _ATTRS.get(name, 'PROBLEM'))
    return _CACHE[name]
