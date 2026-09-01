"""Truncated symmetric transfer error (MSAC) scorer for homographies.

The symmetric transfer error of a correspondence (x, x') under a homography
H is the Hartley-Zisserman pair of transfer distances, *averaged*:

    e^2 = ( d(x', H x)^2 + d(x, H^-1 x')^2 ) / 2.

The averaging is what keeps `max_error` meaning the same thing it means
everywhere else. Poselib's `compute_homography_msac_score` (and OpenCV's
`findHomography`) threshold the forward transfer distance alone, so a
threshold on the plain *sum* of the two terms would be a sqrt(2)-tighter gate
at the same number. With the mean, a correspondence whose two transfer
distances agree - which is the ordinary case, they differ only where H is
locally far from an isometry - has e equal to that one-way distance, so a
threshold ports across unchanged and the MSAC score stays on poselib's scale.
What the symmetry buys is unchanged: a point badly transferred in *either*
direction is charged for it, and the refinement below minimizes both.

Both directions are needed per point, so the *derived form* this scorer works
on is 18 doubles rather than 9: the flat row-major H in 0..8 and its inverse
in 9..17. `homography_derived` builds it, and every kernel here and in
`refiners/homography.py` takes that 18-vector rather than H alone - which is
also exactly what the CUDA `prepare` / `model_derived` contract wants (see
cuda/scoring.py), so both backends read the same layout.

The inverse is the true H^-1 = adj(H) / det(H), not a renormalized one. The
residuals themselves are invariant to the scale of either matrix, but the
*jacobian* of the backward term is not (see refiners/homography.py), so
fixing the scale here is what lets the refiner use the plain
d(H^-1) = -H^-1 dH H^-1 form. A near-singular H is rejected instead, on the
scale-invariant test |det H| > DEGENERATE_TOL * ||H||_F * ||adj H||_F, which
for a unit-norm H is essentially its smallest singular value. The tolerance
is loose enough that it only fires on genuinely degenerate models and tight
enough to keep ||H^-1|| - and therefore the float32 per-point arithmetic on
the GPU - in range.
"""

import math

import numpy as np
from numba import float64, njit

from fastpose.jit_backend import cpu_jit
from fastpose.scorers.sampson import SCORE_CHUNK

# length of the derived form: H (9) followed by H^-1 (9)
DERIVED_SIZE = 18

# smallest |det H| / (||H||_F ||adj H||_F) a model may have and still be
# scored; below this H is treated as unusable, the way a zero Sampson
# denominator is for the epipolar problems.
#
# Loose on purpose. A homography written in *pixel* coordinates is the K H K^-1
# conjugate of a well-behaved one, which alone costs several orders of
# magnitude of conditioning - at focal 1000 the ratio above is already ~1e-5 for
# a perfectly ordinary scene, and it falls with the focal. The estimator only
# ever runs RANSAC in the normalized frame, but `score_numpy` evaluates the
# returned pixel H, so a tolerance tuned to normalized coordinates would reject
# valid models there. This one still rejects every rank-deficient H (their ratio
# is exactly 0) and keeps ||H^-1|| / ||H|| under 1e12, which bounds the largest
# float32 quantity the GPU per-point loop forms at ~1e24 - well inside range.
DEGENERATE_TOL = 1e-12


# ---------------------------------------------------------------------------
# per-point primitives, shared by the CPU scorers below and by the CUDA
# reduction scorer in fastpose/cuda/problems/homography.py
#
# Same arrangement as `build_sampson_point_kernels`: these are the only place
# the symmetric transfer residual and the inverse are written down, so the GPU
# block reduction - which cannot reuse the blocked loops below - still agrees
# with the CPU scorer point for point. See that factory for why `real` exists.
# ---------------------------------------------------------------------------

def build_transfer_point_kernels(jit, real=float64):
    @jit()
    def homography_derived(model, d):
        # d[0:9] = H, d[9:18] = H^-1, for the flat row-major 3x3 `model`.
        # False for a singular or non-finite H, which the CPU scorers report
        # as (1e300, 0) and the CUDA scorer as NO_MODEL.
        norm_sq = real(0.0)
        for j in range(9):
            v = model[j]
            if not math.isfinite(v):
                return False
            d[j] = v
            norm_sq += v * v
        if not (norm_sq > real(0.0)):
            return False

        # adjugate: adj(H)_ij is the cofactor C_ji, so adj = det * H^-1
        d[9] = model[4] * model[8] - model[5] * model[7]
        d[10] = model[2] * model[7] - model[1] * model[8]
        d[11] = model[1] * model[5] - model[2] * model[4]
        d[12] = model[5] * model[6] - model[3] * model[8]
        d[13] = model[0] * model[8] - model[2] * model[6]
        d[14] = model[2] * model[3] - model[0] * model[5]
        d[15] = model[3] * model[7] - model[4] * model[6]
        d[16] = model[1] * model[6] - model[0] * model[7]
        d[17] = model[0] * model[4] - model[1] * model[3]

        det = model[0] * d[9] + model[1] * d[12] + model[2] * d[15]
        adj_sq = real(0.0)
        for j in range(9, 18):
            adj_sq += d[j] * d[j]
        # |det| > tol ||H||_F ||adj||_F, squared to avoid the two square roots
        tol_sq = real(DEGENERATE_TOL * DEGENERATE_TOL)
        if not (det * det > tol_sq * norm_sq * adj_sq):
            return False
        inv = real(1.0) / det
        for j in range(9, 18):
            d[j] *= inv
        return True

    @jit(fastmath=True, inline=True)
    def symmetric_transfer_residual(d, x, y, xp, yp):
        # squared symmetric transfer numerator and denominator of one
        # correspondence under the derived form d = [H | H^-1]. Returned
        # unreduced so callers can apply the inlier test without dividing for
        # outliers, exactly as the Sampson residual is; a denominator of zero
        # means a point transferred to infinity in one of the two directions
        # and is the caller's cue to take the outlier branch.
        #
        #   e^2 = ( |pi(H x) - x'|^2 + |pi(H^-1 x') - x|^2 ) / 2
        #       = (Nf q2^2 + Nb p2^2) / (2 p2^2 q2^2)
        #
        # with p = H x, q = H^-1 x', Nf = |p_01 - x' p2|^2 and
        # Nb = |q_01 - x q2|^2. The factor 2 rides in the denominator, which
        # costs one multiply and keeps the numerator free of it.
        p0 = d[0] * x + d[1] * y + d[2]
        p1 = d[3] * x + d[4] * y + d[5]
        p2 = d[6] * x + d[7] * y + d[8]
        q0 = d[9] * xp + d[10] * yp + d[11]
        q1 = d[12] * xp + d[13] * yp + d[14]
        q2 = d[15] * xp + d[16] * yp + d[17]

        df0 = p0 - xp * p2
        df1 = p1 - yp * p2
        db0 = q0 - x * q2
        db1 = q1 - y * q2
        p2_sq = p2 * p2
        q2_sq = q2 * q2
        numerator = ((df0 * df0 + df1 * df1) * q2_sq
                     + (db0 * db0 + db1 * db1) * p2_sq)
        return numerator, real(2.0) * p2_sq * q2_sq

    return {
        'homography_derived': homography_derived,
        'symmetric_transfer_residual': symmetric_transfer_residual,
    }


_CPU_POINT = build_transfer_point_kernels(cpu_jit)
homography_derived = _CPU_POINT['homography_derived']
symmetric_transfer_residual = _CPU_POINT['symmetric_transfer_residual']


@njit(cache=True, fastmath=True, inline='always')
def _transfer_block(d, x1_x, x1_y, x2_x, x2_y, start, count, max_error_sq):
    # truncated (MSAC) symmetric transfer score + inlier count over count
    # points starting at start. Called with a literal count for every full
    # block so the loop vectorizes; see SCORE_CHUNK in scorers/sampson.py
    score = 0.0
    num_inliers = 0
    for j in range(count):
        i = start + j
        numerator, denominator = symmetric_transfer_residual(
            d, x1_x[i], x1_y[i], x2_x[i], x2_y[i])
        if denominator > 0.0 and numerator < max_error_sq * denominator:
            score += numerator / denominator
            num_inliers += 1
        else:
            score += max_error_sq
    return score, num_inliers


@njit(cache=True, fastmath=True)
def symmetric_transfer_score(model, data, max_error_sq, best_score):
    # truncated (MSAC) symmetric transfer score + inlier count in one fused
    # pass; the truncated score only grows with each point, so once the
    # partial score exceeds the best score so far the model cannot win and
    # scoring can stop early (exact, loss-free bail-out). Checked per
    # SCORE_CHUNK points, as in sampson_score.
    x1_x, x1_y, x2_x, x2_y = data
    d = np.empty(DERIVED_SIZE)
    if not homography_derived(model, d):
        return 1e300, 0
    n = x1_x.shape[0]
    score = 0.0
    num_inliers = 0
    num_full = n // SCORE_CHUNK
    for c in range(num_full):
        block_score, block_inliers = _transfer_block(
            d, x1_x, x1_y, x2_x, x2_y, c * SCORE_CHUNK, SCORE_CHUNK,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
        if score >= best_score:
            return score, num_inliers
    tail = n - num_full * SCORE_CHUNK
    if tail > 0:
        block_score, block_inliers = _transfer_block(
            d, x1_x, x1_y, x2_x, x2_y, num_full * SCORE_CHUNK, tail,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
    return score, num_inliers


def symmetric_transfer_errors_numpy(H, x1, x2):
    # vectorized squared symmetric transfer error of every correspondence;
    # inf everywhere if H is unusable by the same test homography_derived
    # applies, and at any single point that transfers to infinity, so those
    # points land in the outlier branch
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    errors = np.full(len(x1), np.inf)
    if not np.all(np.isfinite(H)):
        return errors

    adj = np.array([
        [H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1],
         H[0, 2] * H[2, 1] - H[0, 1] * H[2, 2],
         H[0, 1] * H[1, 2] - H[0, 2] * H[1, 1]],
        [H[1, 2] * H[2, 0] - H[1, 0] * H[2, 2],
         H[0, 0] * H[2, 2] - H[0, 2] * H[2, 0],
         H[0, 2] * H[1, 0] - H[0, 0] * H[1, 2]],
        [H[1, 0] * H[2, 1] - H[1, 1] * H[2, 0],
         H[0, 1] * H[2, 0] - H[0, 0] * H[2, 1],
         H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]]])
    det = H[0, 0] * adj[0, 0] + H[0, 1] * adj[1, 0] + H[0, 2] * adj[2, 0]
    norm_sq = float(np.sum(H ** 2))
    adj_sq = float(np.sum(adj ** 2))
    if not (det * det > DEGENERATE_TOL ** 2 * norm_sq * adj_sq):
        return errors
    G = adj / det

    x1h = np.column_stack([x1, np.ones(len(x1))])
    x2h = np.column_stack([x2, np.ones(len(x2))])
    p = x1h @ H.T
    q = x2h @ G.T

    forward = np.full(len(x1), np.inf)
    ok = p[:, 2] != 0.0
    forward[ok] = np.sum((p[ok, :2] / p[ok, 2:3] - x2[ok]) ** 2, axis=1)
    backward = np.full(len(x1), np.inf)
    ok = q[:, 2] != 0.0
    backward[ok] = np.sum((q[ok, :2] / q[ok, 2:3] - x1[ok]) ** 2, axis=1)
    return 0.5 * (forward + backward)


def _msac(errors, inliers, max_error_sq):
    # truncated score for a given inlier decision: inliers contribute their
    # own residual, everything else the truncation constant
    return float(np.sum(np.where(inliers, errors, max_error_sq)))


class SymmetricTransferScorer():
    # truncated symmetric transfer error (MSAC) scorer for homography models
    # (a flat row-major 3x3). `max_error` is a one-way pixel distance, as in
    # poselib: the two transfer terms are averaged, not summed.
    score = staticmethod(symmetric_transfer_score)

    @staticmethod
    def score_numpy(H, x1, x2, max_error):
        # vectorized reference implementation; also used to extract the
        # final inlier mask
        max_error_sq = max_error ** 2
        errors = symmetric_transfer_errors_numpy(H, x1, x2)
        inliers = errors <= max_error_sq
        return (_msac(errors, inliers, max_error_sq), inliers,
                int(np.count_nonzero(inliers)))


# ---------------------------------------------------------------------------
# loss-selectable cost kernel for the final refinement pass (LM accept/reject
# only; RANSAC model selection above always stays truncated MSAC via
# symmetric_transfer_score, so its early per-block bail-out stays valid).
# ---------------------------------------------------------------------------

def build_symmetric_transfer_cost(loss):
    # generalizes symmetric_transfer_score's truncated capped-quadratic into
    # an arbitrary loss.cost(r2, max_error_sq); always divides (no need for
    # the truncated fast path's skip-the-division trick since this is only
    # used inside the LM loop, not the O(ransac_iterations x n) scorer)
    cost_fn = loss.cost

    @njit(cache=True, fastmath=True)
    def _symmetric_transfer_cost(model, data, max_error_sq, best_score):
        x1_x, x1_y, x2_x, x2_y = data
        d = np.empty(DERIVED_SIZE)
        if not homography_derived(model, d):
            return 1e300, 0
        n = x1_x.shape[0]
        score = 0.0
        num_inliers = 0
        for i in range(n):
            numerator, denominator = symmetric_transfer_residual(
                d, x1_x[i], x1_y[i], x2_x[i], x2_y[i])
            if denominator > 0.0:
                r2 = numerator / denominator
                score += cost_fn(r2, max_error_sq)
                if r2 < max_error_sq:
                    num_inliers += 1
            else:
                score += cost_fn(1e18, max_error_sq)
        return score, num_inliers

    return _symmetric_transfer_cost
