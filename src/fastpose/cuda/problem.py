"""What the generic CUDA driver needs to know about one estimation problem.

`CudaProblem` is the seam between `cuda/ransac.py` - which owns the round
loop, the local-optimization gate, adaptive termination and the packed
readback, none of which is problem-specific - and the per-problem device code
in `cuda/problems/`. A port adds one module there and one entry in the
registry; it does not touch the driver.

The kernels are built lazily and memoized, because building them is what
triggers numba's compilation: importing `fastpose.cuda` on a machine with a
GPU must stay as cheap as importing it without one.

Sizes, and where each comes from:

    sample_size, num_params, max_models   the CPU solver class
    data_width                            columns of the `data` tuple
    num_tangent, state_size               the CPU refiner module
    derived_size                          the derived form the per-point loops
                                          read (9 for an epipolar matrix, 12
                                          for a focal-scaled pose)
    basis_width                           columns of the tangent basis; 9 for
                                          the Sampson problems, 1 (unused) for
                                          the reprojection ones
    aux_shape, scratch_shape              float64 thread-0 scratch that must
                                          survive between calls, and the
                                          per-thread float32 jacobian scratch
    min_inliers                           poselib's `if (num_inl <= N) return`
                                          gate in refine_model
"""

import numpy as np
from numba import cuda

from fastpose.cuda.lm import build_lm_refine_kernel, lm_threads_for
from fastpose.cuda.scoring import build_score_kernel


class CudaProblem():

    def __init__(self, name, sample_size, num_params, max_models, data_width,
                 num_tangent, state_size, derived_size, basis_width,
                 aux_shape, scratch_shape, min_inliers, relaxed_scale,
                 solve_batch, score_kernels, refine_factory,
                 default_params=None):
        self.name = name
        self.sample_size = sample_size
        self.num_params = num_params
        self.max_models = max_models
        self.data_width = data_width
        self.num_tangent = num_tangent
        self.state_size = state_size
        self.derived_size = derived_size
        self.basis_width = basis_width
        self.aux_shape = aux_shape
        self.scratch_shape = scratch_shape
        self.min_inliers = min_inliers
        self.relaxed_scale = relaxed_scale
        self.lm_threads = lm_threads_for(num_tangent)
        self.default_params = (np.zeros(1, dtype=np.float64)
                               if default_params is None else
                               np.ascontiguousarray(default_params,
                                                    dtype=np.float64))

        self._solve_batch = solve_batch
        self._score_kernels = score_kernels
        self._refine_factory = refine_factory
        self._score_kernel = None
        self._lm_kernels = {}

    # -- kernels ------------------------------------------------------------

    def solve_batch(self, data, params, samples, models, counts, stream=0):
        self._solve_batch(data, params, samples, models, counts, stream)

    def score_kernel(self):
        if self._score_kernel is None:
            prepare, score_point = self._score_kernels
            self._score_kernel = build_score_kernel(
                prepare, score_point, self.derived_size, self.num_params)
        return self._score_kernel

    def lm_kernel(self, loss):
        # memoized per loss *type*, so rebuilding an estimator does not
        # recompile; the RANSAC-internal truncated pass and the final Cauchy
        # polish are the two that exist in practice
        key = type(loss)
        if key not in self._lm_kernels:
            self._lm_kernels[key] = build_lm_refine_kernel(
                self._refine_factory(loss), self.num_tangent, self.state_size,
                self.num_params, self.derived_size, self.basis_width,
                self.aux_shape, self.scratch_shape, self.min_inliers,
                self.lm_threads)
        return self._lm_kernels[key]

    # -- device buffers -----------------------------------------------------

    def allocate_models(self, batch):
        # per-round model and count buffers, reused across rounds
        models = cuda.device_array((batch, self.max_models, self.num_params),
                                   dtype=np.float64)
        counts = cuda.device_array(batch, dtype=np.int64)
        return models, counts
