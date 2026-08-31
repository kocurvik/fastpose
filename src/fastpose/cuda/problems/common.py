"""GPU instantiations of the shared kernel factories, built exactly once.

Several problems need the same device functions - the Sampson primitives, the
retraction and tangent basis, the Sturm root isolation the 5-point and P4Pf
solvers share. Building a factory twice would produce two distinct dispatcher
objects for the same source, which means two compilations and two on-disk
cache entries, so every problem module imports its pieces from here instead.

Nothing in this module compiles: `cuda.jit(device=True)` only records the
function, and numba compiles it when the kernel that calls it is launched.
"""

from numba import float32, float64

from fastpose.cuda.backend import cuda_jit
from fastpose.refiners.absolute import build_reprojection_primitives
from fastpose.refiners.monodepth import (build_monodepth_primitives,
                                         build_monodepth_reproj_kernels,
                                         build_monodepth_residual_kernels)
from fastpose.refiners.utils import build_refiner_primitives
from fastpose.scorers.reprojection import build_reprojection_point_kernels
from fastpose.scorers.sampson import build_sampson_point_kernels
from fastpose.solvers.essential import build_five_point_kernels
from fastpose.solvers.fundamental import build_seven_point_kernels
from fastpose.solvers.p3p import build_p3p_kernels
from fastpose.solvers.p4pf import build_p4pf_kernels
from fastpose.solvers.shared_focal import build_shared_focal_kernels

# --- epipolar -------------------------------------------------------------
# float32 per-point kernels; the scores and normal equations accumulate in
# float64 in the callers
SAMPSON32 = build_sampson_point_kernels(cuda_jit, real=float32)
# float64, used once per model to form E (or F) before it is rounded
SAMPSON64 = build_sampson_point_kernels(cuda_jit, real=float64)

# float64: the retractions, tangent bases and pose state. All O(1) per LM
# step, and the state is what the caller ultimately gets back.
PRIM64 = build_refiner_primitives(cuda_jit, real=float64)
# float32: the per-point Sampson jacobian, which is the O(n) work
PRIM32 = build_refiner_primitives(cuda_jit, real=float32)

# --- reprojection ---------------------------------------------------------
REPROJ32 = build_reprojection_point_kernels(cuda_jit, real=float32)
REPROJ64 = build_reprojection_point_kernels(cuda_jit, real=float64)
REPROJ_JAC32 = build_reprojection_primitives(cuda_jit, real=float32)
REPROJ_FOCAL_JAC32 = build_reprojection_primitives(cuda_jit, real=float32,
                                                   focal=True)

# --- solvers shared between problems --------------------------------------
# the 5-point chain also carries the Sturm root isolation P4Pf needs and the
# essential decomposition the focal solvers need; the 7-point chain carries
# the nullspace, determinant and cubic the focal solvers build on
FIVE_POINT = build_five_point_kernels(cuda_jit)
REAL_ROOTS_STURM = FIVE_POINT['real_roots_sturm']
SEVEN_POINT = build_seven_point_kernels(cuda_jit)

# The monodepth solvers borrow from three more chains: P3P (the calibrated
# variant *is* a P3P on depth-induced 3D points), P4Pf (its 3xn null-space
# solve and rotation projection) and the 6-point shared-focal solver (its
# Danilevsky characteristic polynomial). Built here rather than in each
# problem module so `absolute`, `absolute-focal`, `shared-focal` and the four
# monodepth problems all share one dispatcher per kernel.
P3P = build_p3p_kernels(cuda_jit)
P4PF = build_p4pf_kernels(cuda_jit, REAL_ROOTS_STURM)
SHARED_FOCAL = build_shared_focal_kernels(cuda_jit, REAL_ROOTS_STURM,
                                          FIVE_POINT['pose_from_essential'])

# --- monodepth ------------------------------------------------------------
# float64: the E -> F map and the tangent basis, both O(1) per LM step
MONO64 = build_monodepth_primitives(cuda_jit, real=float64)
# float32 per-point reprojection residuals and jacobian rows, one build per
# (num_tangent, focal) combination the four monodepth problems use
MONO_REPROJ32 = {
    key: build_monodepth_reproj_kernels(cuda_jit, real=float32,
                                        num_tangent=key[0], focal=key[1])
    for key in ((7, False), (9, False), (8, True), (9, True))
}
# float32 residual-only variants for the O(n) cost evaluation, which does not
# pay for the jacobian rows
MONO_RESID32 = {
    focal: build_monodepth_residual_kernels(cuda_jit, real=float32,
                                            focal=focal)
    for focal in (False, True)
}
