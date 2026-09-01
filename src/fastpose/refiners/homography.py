"""Homography refiner: LM on the truncated symmetric transfer error with H
carried as a unit-norm 9-vector on the sphere S^8.

A homography has 8 degrees of freedom in a 9-entry matrix, and the missing
one is pure gauge: scaling H changes nothing a residual can see. Fixing an
entry (H33 = 1, say) removes it but breaks wherever that entry passes through
zero. Instead the state here *is* the flat H, normalized to unit Frobenius
norm, and the 8 tangent parameters are the orthogonal complement of h inside
R^9:

    state[0:9]   H (row-major 3x3), ||H||_F = 1
    tangent      an orthonormal basis of h^perp, from one Householder
                 reflector; retraction h <- (h + B^T delta) / ||.||

The basis is a deterministic function of h - the same requirement
`tangent_basis` in refiners/utils.py carries - so the jacobian and the
retraction always agree, and rebuilding it inside the retraction costs 72
entries against an O(n) accumulate.

The residuals are the four components of the symmetric transfer error, two
forward and two backward, each scaled by 1/sqrt(2) so that their squares sum
to the *mean* of the two transfer terms - the e2 the scorer thresholds.

The forward pair differentiates the way any projective transfer does; the
backward pair needs d(H^-1) = -H^-1 dH H^-1, which collapses to a rank-one
outer product per residual:

    d r_back / dH = -(H^-T b) (H^-1 x')^T

for the row b = d pi / d q of the projection at q = H^-1 x'. That identity is
what makes the backward term as cheap as the forward one - but it holds for
the *true* inverse only, which is why `homography_derived` in
scorers/transfer.py fixes the scale of H^-1 rather than renormalizing it.

Only the jacobian accumulation and the retraction are defined here - the LM
loop itself lives in refiners/lm.py.
"""

import math

import numpy as np
from numba import float64, njit

from fastpose.jit_backend import cpu_jit
from fastpose.refiners.lm import build_lm_refine
from fastpose.refiners.losses import TruncatedLoss
from fastpose.scorers.transfer import (DERIVED_SIZE, build_symmetric_transfer_cost,
                                       homography_derived,
                                       symmetric_transfer_score)

STATE_SIZE = 9
MODEL_SIZE = 9
NUM_TANGENT = 8     # the tangent space of the unit sphere in R^9

# residual components per correspondence: forward (2) then backward (2)
NUM_RESIDUALS = 4


# ---------------------------------------------------------------------------
# primitives shared with the CUDA refiner
#
# `fastpose/cuda/problems/homography.py` runs the same LM on the GPU, but as a
# block reduction rather than a serial loop, so it cannot reuse the accumulate
# kernel below. It must agree with it term for term, which is what building
# the state map, the tangent basis, the retraction and the per-point jacobian
# from one factory guarantees - the retraction in particular has to rebuild
# *exactly* the basis the jacobian used. See fastpose/jit_backend.py for the
# `jit` shim.
#
# The two kernels that need scratch take it pre-shaped because `np.empty` does
# not compile in device code; the CPU wrappers below restore the shorter
# signatures.
# ---------------------------------------------------------------------------

def build_homography_primitives(jit, real=float64):
    # `real` types the float literals in `transfer_point_jacobian`, the only
    # kernel here that runs per point and so the only one the CUDA refiner
    # builds in float32; see build_sampson_point_kernels for why a bare
    # literal would silently promote the expression back to float64. The
    # tangent basis and the retraction are O(1) per LM step and stay float64.
    INV_SQRT2 = real(0.7071067811865476)

    @jit()
    def sphere_init_state(model, state):
        # the state is the model, normalized onto the unit sphere
        norm_sq = 0.0
        for j in range(9):
            v = model[j]
            # a non-finite entry is the only way this can fail, and an
            # exception cannot propagate out of the parallel RANSAC driver
            if not math.isfinite(v):
                return False
            state[j] = v
            norm_sq += v * v
        if not (norm_sq > 0.0):
            return False
        inv = 1.0 / math.sqrt(norm_sq)
        for j in range(9):
            state[j] *= inv
        return True

    @jit()
    def sphere_state_to_model(state, h):
        for j in range(9):
            h[j] = state[j]

    @jit()
    def sphere_tangent_basis_core(h, B, w):
        # orthonormal basis of the tangent space of the unit sphere at h,
        # written as the 8 rows of B (flat 9-vectors).
        #
        # Take the Householder reflector P = I - 2 w w^T that maps h to
        # alpha e_0. P is orthogonal and symmetric, so {P e_j} is an
        # orthonormal basis of R^9 with P e_0 = h / ||h||; the remaining
        # eight columns therefore span h^perp exactly. `w` is scratch (9).
        norm_sq = 0.0
        for j in range(9):
            norm_sq += h[j] * h[j]
        norm = math.sqrt(norm_sq)
        # the far reflection, so w never comes out of a cancelling difference
        alpha = -math.copysign(norm, h[0])
        w[0] = h[0] - alpha
        for j in range(1, 9):
            w[j] = h[j]
        u_sq = 0.0
        for j in range(9):
            u_sq += w[j] * w[j]
        inv = 1.0 / math.sqrt(u_sq)
        for j in range(9):
            w[j] *= inv

        for p in range(8):
            k = p + 1
            two_wk = 2.0 * w[k]
            for j in range(9):
                B[p, j] = -two_wk * w[j]
            B[p, k] += 1.0

    @jit()
    def sphere_retract_core(h, delta, h_new, B, w):
        # h <- (h + sum_p delta_p B[p]) / ||.||; the basis is rebuilt from h
        # rather than passed in, so this and the jacobian cannot drift apart.
        # The step is orthogonal to h, so the pre-normalization norm is
        # sqrt(1 + |delta|^2) and can never vanish.
        sphere_tangent_basis_core(h, B, w)
        norm_sq = 0.0
        for j in range(9):
            s = h[j]
            for p in range(8):
                s += delta[p] * B[p, j]
            h_new[j] = s
            norm_sq += s * s
        if not (norm_sq > 0.0):
            for j in range(9):
                h_new[j] = h[j]
            return
        inv = 1.0 / math.sqrt(norm_sq)
        for j in range(9):
            h_new[j] *= inv

    @jit(fastmath=True, inline=True)
    def transfer_point_jacobian(d, x, y, xp, yp, dr, res):
        # the four symmetric transfer residuals of one correspondence under
        # the derived form d = [H | H^-1], written into `res`, and their
        # derivatives d r_k / dH as the four flat 9-vectors of `dr`.
        # Returns (e2, ok) with e2 the squared symmetric transfer error;
        # ok is False when either transfer sends the point to infinity, where
        # dr and res are left untouched and the caller drops the point.
        #
        # Every component carries the 1/sqrt(2) of the scorer's *mean* of the
        # two transfer terms, so that sum_k r_k^2 is exactly the e2 the cost
        # kernel computes. That identity is what makes JtJ/Jtr the Gauss-
        # Newton system of the cost the LM's accept/reject test evaluates;
        # dropping the factor would leave the two off by a constant 2. It
        # rides in `inv_p`/`inv_q`, so it costs no arithmetic in the rows -
        # only the four residuals pay a multiply.
        p0 = d[0] * x + d[1] * y + d[2]
        p1 = d[3] * x + d[4] * y + d[5]
        p2 = d[6] * x + d[7] * y + d[8]
        q0 = d[9] * xp + d[10] * yp + d[11]
        q1 = d[12] * xp + d[13] * yp + d[14]
        q2 = d[15] * xp + d[16] * yp + d[17]
        if p2 == real(0.0) or q2 == real(0.0):
            return real(0.0), False

        # forward: r = pi(H x) - x', so d r_k / dH = a_k x_h^T with
        # a_0 = (1, 0, -u) / p2 and a_1 = (0, 1, -v) / p2
        raw_p = real(1.0) / p2
        u = p0 * raw_p
        v = p1 * raw_p
        res[0] = INV_SQRT2 * (u - xp)
        res[1] = INV_SQRT2 * (v - yp)
        inv_p = INV_SQRT2 * raw_p
        au = -u * inv_p
        av = -v * inv_p
        dr[0, 0] = inv_p * x
        dr[0, 1] = inv_p * y
        dr[0, 2] = inv_p
        dr[0, 3] = real(0.0)
        dr[0, 4] = real(0.0)
        dr[0, 5] = real(0.0)
        dr[0, 6] = au * x
        dr[0, 7] = au * y
        dr[0, 8] = au
        dr[1, 0] = real(0.0)
        dr[1, 1] = real(0.0)
        dr[1, 2] = real(0.0)
        dr[1, 3] = inv_p * x
        dr[1, 4] = inv_p * y
        dr[1, 5] = inv_p
        dr[1, 6] = av * x
        dr[1, 7] = av * y
        dr[1, 8] = av

        # backward: r = pi(H^-1 x') - x, and with G = H^-1, q = G x'_h,
        # dG = -G dH G gives d r_k / dH = -(G^T b_k) q^T for the same
        # projection rows b_0 = (1, 0, -ub) / q2 and b_1 = (0, 1, -vb) / q2
        raw_q = real(1.0) / q2
        ub = q0 * raw_q
        vb = q1 * raw_q
        res[2] = INV_SQRT2 * (ub - x)
        res[3] = INV_SQRT2 * (vb - y)
        inv_q = INV_SQRT2 * raw_q
        bu = -ub * inv_q
        bv = -vb * inv_q
        for i in range(3):
            # (G^T b)_i, with the zero component of each b skipped
            c0 = d[9 + i] * inv_q + d[15 + i] * bu
            c1 = d[12 + i] * inv_q + d[15 + i] * bv
            dr[2, 3 * i] = -c0 * q0
            dr[2, 3 * i + 1] = -c0 * q1
            dr[2, 3 * i + 2] = -c0 * q2
            dr[3, 3 * i] = -c1 * q0
            dr[3, 3 * i + 1] = -c1 * q1
            dr[3, 3 * i + 2] = -c1 * q2

        e2 = (res[0] * res[0] + res[1] * res[1]
              + res[2] * res[2] + res[3] * res[3])
        return e2, True

    return {
        'sphere_init_state': sphere_init_state,
        'sphere_state_to_model': sphere_state_to_model,
        'sphere_tangent_basis_core': sphere_tangent_basis_core,
        'sphere_retract_core': sphere_retract_core,
        'transfer_point_jacobian': transfer_point_jacobian,
    }


_CPU_PRIM = build_homography_primitives(cpu_jit)
_init_state = _CPU_PRIM['sphere_init_state']
_state_to_model = _CPU_PRIM['sphere_state_to_model']
_sphere_tangent_basis_core = _CPU_PRIM['sphere_tangent_basis_core']
_sphere_retract_core = _CPU_PRIM['sphere_retract_core']
transfer_point_jacobian = _CPU_PRIM['transfer_point_jacobian']


@njit(cache=True)
def sphere_tangent_basis(h, B):
    # CPU wrapper: allocates the scratch the shared core wants pre-shaped.
    # The GPU refiner calls the core with per-thread local arrays instead, so
    # both backends build the same basis.
    w = np.empty(9)
    _sphere_tangent_basis_core(h, B, w)


@njit(cache=True)
def _apply_step(state, delta, state_new):
    B = np.empty((NUM_TANGENT, 9))
    w = np.empty(9)
    _sphere_retract_core(state, delta, state_new, B, w)


def build_transfer_accumulate(loss):
    # builds the normal equations of the four symmetric transfer residuals
    # r_k for the tangent basis B: JtJ += w_i sum_k J_k J_k^T,
    # Jtr += w_i sum_k J_k r_k with J_k[p] = dr_k/dH . B[p] and
    # w_i = loss.weight(e2_i, max_error_sq); a zero weight drops the point
    # entirely (for TruncatedLoss this is exactly the hard truncation the
    # scorer applies). `loss` is bound as a compile-time constant, the same
    # closure pattern build_lm_refine uses for its own kernels.
    #
    # Deliberately serial, like `build_sampson_accumulate` - see the measured
    # note there before trying to parallelize an LM inner loop.
    weight_fn = loss.weight

    @njit(cache=True, fastmath=True)
    def _accumulate(data, f, state, JtJ, Jtr, max_error_sq):
        # returns the number of contributing scalar residuals (4 per point)
        x1_x, x1_y, x2_x, x2_y = data
        n = x1_x.shape[0]

        for p in range(NUM_TANGENT):
            Jtr[p] = 0.0
            for q in range(NUM_TANGENT):
                JtJ[p, q] = 0.0

        d = np.empty(DERIVED_SIZE)
        if not homography_derived(f, d):
            return 0
        B = np.empty((NUM_TANGENT, 9))
        sphere_tangent_basis(state, B)

        dr = np.empty((NUM_RESIDUALS, 9))
        res = np.empty(NUM_RESIDUALS)
        J = np.empty((NUM_RESIDUALS, NUM_TANGENT))

        num_residuals = 0
        for i in range(n):
            e2, ok = transfer_point_jacobian(d, x1_x[i], x1_y[i], x2_x[i],
                                             x2_y[i], dr, res)
            if not ok:
                continue
            w = weight_fn(e2, max_error_sq)
            if w <= 0.0:
                continue
            num_residuals += NUM_RESIDUALS

            # all four tangent-space rows first, then one pass over the upper
            # triangle summing their outer products - 36 updates to JtJ per
            # point rather than 4 x 36. The GPU accumulate is structured the
            # same way, where each of those updates lands in shared memory.
            for k in range(NUM_RESIDUALS):
                for p in range(NUM_TANGENT):
                    acc = 0.0
                    for j in range(9):
                        acc += dr[k, j] * B[p, j]
                    J[k, p] = acc
            for p in range(NUM_TANGENT):
                jr = 0.0
                for k in range(NUM_RESIDUALS):
                    jr += J[k, p] * res[k]
                Jtr[p] += w * jr
                for q in range(p, NUM_TANGENT):
                    jj = 0.0
                    for k in range(NUM_RESIDUALS):
                        jj += J[k, p] * J[k, q]
                    JtJ[p, q] += w * jj

        for p in range(NUM_TANGENT):
            for q in range(p):
                JtJ[p, q] = JtJ[q, p]
        return num_residuals

    return _accumulate


_accumulate_truncated = build_transfer_accumulate(TruncatedLoss())
_accumulate = _accumulate_truncated  # back-compat alias for the default kernel

_refine_homography_lm = build_lm_refine(_init_state, _state_to_model,
                                        _accumulate_truncated, _apply_step,
                                        symmetric_transfer_score, STATE_SIZE,
                                        NUM_TANGENT, MODEL_SIZE)

_final_kernels = {}


def _get_final_refine(loss):
    # lazily compiles (and caches) the LM kernel for a non-default loss; see
    # essential.py's _get_final_refine for the general rationale
    key = type(loss)
    if key not in _final_kernels:
        accumulate_final = build_transfer_accumulate(loss)
        cost_final = build_symmetric_transfer_cost(loss)
        _final_kernels[key] = build_lm_refine(
            _init_state, _state_to_model, accumulate_final, _apply_step,
            cost_final, STATE_SIZE, NUM_TANGENT, MODEL_SIZE)
    return _final_kernels[key]


class LMHomographyRefiner():
    # local optimization: Levenberg-Marquardt on the symmetric transfer error
    # with H carried as a unit-norm 9-vector and 8 tangent parameters on the
    # sphere. num_iterations is the LM step budget; converged runs stop
    # earlier. `loss` selects the robust cost/weighting (TruncatedLoss by
    # default, matching every RANSAC-internal use; pass e.g. CauchyLoss() or
    # TruncatedCauchyLoss() for a final polish pass on an inlier-only subset).
    def __init__(self, num_iterations=25, loss=TruncatedLoss()):
        self.num_iterations = num_iterations
        self.loss = loss
        if not isinstance(loss, TruncatedLoss):
            self.refine = _get_final_refine(loss)

    refine = staticmethod(_refine_homography_lm)
