"""Per-problem device code for the CUDA backend.

Each module here supplies the three things `cuda/problem.py` needs and
nothing else: a launcher for the batched minimal solver, the scorer's two
device functions, and a factory that builds the LM device functions for a
given loss. The driver, the reductions, the damped Cholesky and the score
bookkeeping are shared and live one directory up.

Importing this package must not compile anything - `fastpose.cuda` is
imported on machines with no GPU. The modules are therefore imported lazily
by `fastpose.cuda.registry.get_problem`.
"""
