"""Truncated Sampson error (MSAC) scorer for epipolar geometry.

The same Sampson form works for fundamental and essential matrices;
`SampsonScorer` scores flat 3x3 models directly, `PoseSampsonScorer` scores
relative pose models [R | t] by assembling E = [t]_x R on the fly.

Poselib has two overloads of `compute_sampson_msac_score` (robust/utils.cc)
and they are not equivalent. The one taking a `CameraPose` - used by the
calibrated `RelativePoseEstimator` - additionally requires every inlier to
triangulate in front of both cameras (`check_cheirality` with a minimum
depth of `MIN_DEPTH`); a point that passes the Sampson test but fails
cheirality is charged the truncation constant and not counted. The one
taking a matrix - used by the shared- and varying-focal estimators, which
score an F rather than a pose - does not. `get_inliers` splits the same way.
`PoseSampsonScorer` therefore applies the check and the focal scorers below
deliberately do not.

The check belongs to model *scoring* only: poselib's bundles minimize the
plain Sampson cost, so the cheirality-free `pose_sampson_score` (and the
kernels from `build_pose_sampson_cost`) are what the LM refiners use.
"""

import math

import numpy as np
from numba import float64, njit

from fastpose.jit_backend import cpu_jit


# points per early-bail-out check. The block helpers below are called with
# this as a literal for every full block, so after inlining LLVM sees a
# constant trip count and vectorizes the body; a `for i in range(start, end)`
# with a runtime `end` does not vectorize (it compiles to a mostly scalar
# loop, which cost ~2x on the O(ransac_iterations x n) scorers).
SCORE_CHUNK = 512

# minimum depth poselib's robust/utils.cc passes to check_cheirality, in
# units of the pose translation (unit norm for the models this package
# produces, so effectively a fraction of the baseline)
MIN_DEPTH = 0.01


# ---------------------------------------------------------------------------
# per-point primitives, shared by the CPU scorers below and by the CUDA
# reduction scorer in fastpose/cuda/scorers.py
#
# These are the only place the Sampson residual, the pose-to-E map and the
# cheirality test are written down. The GPU scorer is a block reduction rather
# than a serial blocked loop, so it cannot reuse the *_score kernels below -
# but it must agree with them point for point, which is what sharing these
# three primitives guarantees. `inline=True` everywhere: on the CPU they are
# inlined at the Numba IR level and lowered with their caller's fastmath, so
# the blocked loops still vectorize exactly as before.
# ---------------------------------------------------------------------------

def build_sampson_point_kernels(jit, real=float64):
    # `real` is the numba type every float literal in these kernels is cast
    # to. It exists because a bare Python literal is float64 in numba, so a
    # single `1.0` in an otherwise float32 expression promotes the whole chain
    # back to float64 - silently undoing the mixed precision. The CUDA scorer
    # and LM build these with float32; everything else uses the default.
    @jit(fastmath=True, inline=True)
    def sampson_residual(f, x, y, xp, yp):
        # squared Sampson numerator and denominator of one correspondence
        # under the flat row-major 3x3 epipolar matrix f. Returned unreduced
        # so callers can apply the inlier test without dividing for outliers;
        # a denominator of zero marks a degenerate model and is the caller's
        # cue to take the outlier branch rather than divide.
        fx1_0 = f[0] * x + f[1] * y + f[2]
        fx1_1 = f[3] * x + f[4] * y + f[5]
        fx1_2 = f[6] * x + f[7] * y + f[8]
        ftx2_0 = f[0] * xp + f[3] * yp + f[6]
        ftx2_1 = f[1] * xp + f[4] * yp + f[7]
        residual = xp * fx1_0 + yp * fx1_1 + fx1_2
        denominator = (fx1_0 * fx1_0 + fx1_1 * fx1_1
                       + ftx2_0 * ftx2_0 + ftx2_1 * ftx2_1)
        return residual * residual, denominator

    @jit(inline=True)
    def essential_from_pose(pose, e):
        # e = flat E = [t]_x R for a pose model [R (row-major 3x3) | t (3)];
        # any t scale (the Sampson error is invariant to it)
        tx = pose[9]
        ty = pose[10]
        tz = pose[11]
        for j in range(3):
            r0 = pose[j]
            r1 = pose[3 + j]
            r2 = pose[6 + j]
            e[j] = -tz * r1 + ty * r2
            e[3 + j] = tz * r0 - tx * r2
            e[6 + j] = -ty * r0 + tx * r1

    @jit(inline=True)
    def cheirality_ok(pose, x, y, xp, yp, min_depth):
        # poselib's check_cheirality (misc/essential.cc): the two depths from
        # the least-squares intersection of the unit rays through the
        # calibrated points (x, y) and (xp, yp) must both exceed min_depth.
        # The common 1 / (1 - a^2) factor is positive, so upstream drops it
        # and scales min_depth by (1 - a^2) instead; kept identical here.
        one = real(1.0)
        inv1 = one / math.sqrt(x * x + y * y + one)
        inv2 = one / math.sqrt(xp * xp + yp * yp + one)
        u0 = x * inv1
        u1 = y * inv1
        u2 = inv1
        v0 = xp * inv2
        v1 = yp * inv2
        v2 = inv2
        ru0 = pose[0] * u0 + pose[1] * u1 + pose[2] * u2
        ru1 = pose[3] * u0 + pose[4] * u1 + pose[5] * u2
        ru2 = pose[6] * u0 + pose[7] * u1 + pose[8] * u2
        tx = pose[9]
        ty = pose[10]
        tz = pose[11]
        a = -(ru0 * v0 + ru1 * v1 + ru2 * v2)
        b1 = -(ru0 * tx + ru1 * ty + ru2 * tz)
        b2 = v0 * tx + v1 * ty + v2 * tz
        depth1 = b1 - a * b2
        depth2 = b2 - a * b1
        thr = min_depth * (one - a * a)
        return depth1 > thr and depth2 > thr

    @jit()
    def calibrate_epipolar_core(e, pp1x, pp1y, pp2x, pp2y, inv1, inv2, f, a):
        # f = flat K2^-T e K1^-1 for a flat row-major 3x3 e, with inv1 = 1 / f1,
        # inv2 = 1 / f2 and K = [[f, 0, ppx], [0, f, ppy], [0, 0, 1]]. The map is
        # linear in e, so the focal refiners reuse it to push a tangent direction
        # dE/dtheta through to dF/dtheta with the same code that builds F itself.
        # `a` is caller-provided scratch (9), because np.empty is host-only.
        # rows of A = K2^-T e, then columns of F = A K1^-1
        for j in range(3):
            a[j] = inv2 * e[j]
            a[3 + j] = inv2 * e[3 + j]
            a[6 + j] = e[6 + j] - inv2 * (pp2x * e[j] + pp2y * e[3 + j])
        for i in range(3):
            f[3 * i] = inv1 * a[3 * i]
            f[3 * i + 1] = inv1 * a[3 * i + 1]
            f[3 * i + 2] = (a[3 * i + 2]
                            - inv1 * (pp1x * a[3 * i] + pp1y * a[3 * i + 1]))

    @jit()
    def model_to_fundamental_core(model, pp1x, pp1y, pp2x, pp2y, f, e, a):
        # f = flat F = K2^-T E K1^-1 for a pose model [R | t | f1 | f2] with
        # K = [[f, 0, ppx], [0, f, ppy], [0, 0, 1]]; False for invalid focals.
        # `e` (9) and `a` (9) are caller-provided scratch.
        f1 = model[12]
        f2 = model[13]
        if f1 <= real(0.0) or f2 <= real(0.0):
            return False
        essential_from_pose(model, e)
        calibrate_epipolar_core(e, pp1x, pp1y, pp2x, pp2y, real(1.0) / f1,
                                real(1.0) / f2, f, a)
        return True

    return {
        'sampson_residual': sampson_residual,
        'essential_from_pose': essential_from_pose,
        'cheirality_ok': cheirality_ok,
        'calibrate_epipolar_core': calibrate_epipolar_core,
        'model_to_fundamental_core': model_to_fundamental_core,
    }


_CPU_POINT = build_sampson_point_kernels(cpu_jit)
sampson_residual = _CPU_POINT['sampson_residual']
essential_from_pose = _CPU_POINT['essential_from_pose']
cheirality_ok = _CPU_POINT['cheirality_ok']
_calibrate_epipolar_core = _CPU_POINT['calibrate_epipolar_core']
_model_to_fundamental_core = _CPU_POINT['model_to_fundamental_core']


@njit(cache=True, fastmath=True, inline='always')
def _sampson_block(f, x1_x, x1_y, x2_x, x2_y, start, count, max_error_sq):
    # truncated (MSAC) Sampson score + inlier count over count points
    # starting at start
    score = 0.0
    num_inliers = 0
    for j in range(count):
        i = start + j
        r2, denominator = sampson_residual(f, x1_x[i], x1_y[i], x2_x[i], x2_y[i])
        # r2 / denominator < max_error_sq, without dividing for outliers.
        # Degenerate models can make the Sampson denominator exactly zero;
        # keep those points in the outlier branch instead of dividing.
        if denominator > 0.0 and r2 < max_error_sq * denominator:
            score += r2 / denominator
            num_inliers += 1
        else:
            score += max_error_sq
    return score, num_inliers


@njit(cache=True, fastmath=True)
def sampson_score(f, data, max_error_sq, best_score):
    # truncated (MSAC) Sampson score + inlier count in one fused pass;
    # the truncated score only grows with each point, so once the partial
    # score exceeds the best score so far the model cannot win and scoring
    # can stop early (exact, loss-free bail-out). Checked per SCORE_CHUNK
    # points.
    x1_x, x1_y, x2_x, x2_y = data
    n = x1_x.shape[0]
    score = 0.0
    num_inliers = 0
    num_full = n // SCORE_CHUNK
    for c in range(num_full):
        block_score, block_inliers = _sampson_block(
            f, x1_x, x1_y, x2_x, x2_y, c * SCORE_CHUNK, SCORE_CHUNK,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
        if score >= best_score:
            return score, num_inliers
    tail = n - num_full * SCORE_CHUNK
    if tail > 0:
        block_score, block_inliers = _sampson_block(
            f, x1_x, x1_y, x2_x, x2_y, num_full * SCORE_CHUNK, tail,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
    return score, num_inliers


def sampson_errors_numpy(F, x1, x2):
    # vectorized squared Sampson error of every correspondence; inf where the
    # denominator degenerates, so those points land in the outlier branch
    x1_x = x1[:, 0]
    x1_y = x1[:, 1]
    x2_x = x2[:, 0]
    x2_y = x2[:, 1]

    fx1_0 = F[0, 0] * x1_x + F[0, 1] * x1_y + F[0, 2]
    fx1_1 = F[1, 0] * x1_x + F[1, 1] * x1_y + F[1, 2]
    fx1_2 = F[2, 0] * x1_x + F[2, 1] * x1_y + F[2, 2]
    ftx2_0 = F[0, 0] * x2_x + F[1, 0] * x2_y + F[2, 0]
    ftx2_1 = F[0, 1] * x2_x + F[1, 1] * x2_y + F[2, 1]

    residual = x2_x * fx1_0 + x2_y * fx1_1 + fx1_2
    denominator = fx1_0 * fx1_0 + fx1_1 * fx1_1 + ftx2_0 * ftx2_0 + ftx2_1 * ftx2_1
    errors = np.full(len(x1), np.inf)
    valid = denominator > 0.0
    errors[valid] = residual[valid] * residual[valid] / denominator[valid]
    return errors


def _msac(errors, inliers, max_error_sq):
    # truncated score for a given inlier decision: inliers contribute their
    # own residual, everything else the truncation constant
    return float(np.sum(np.where(inliers, errors, max_error_sq)))


class SampsonScorer():
    # truncated Sampson error (MSAC) scorer for epipolar geometry; the same
    # scorer works for fundamental and essential matrix models
    score = staticmethod(sampson_score)

    @staticmethod
    def score_numpy(F, x1, x2, max_error):
        # vectorized reference implementation; also used to extract the
        # final inlier mask
        max_error_sq = max_error ** 2
        errors = sampson_errors_numpy(F, x1, x2)
        inliers = errors <= max_error_sq
        return _msac(errors, inliers, max_error_sq), inliers, int(np.count_nonzero(inliers))


@njit(cache=True, fastmath=True)
def pose_sampson_score(model, data, max_error_sq, best_score):
    # scorer for pose models: assemble E = [t]_x R, then the shared
    # truncated Sampson score. No cheirality check - this is the cost the LM
    # refiners minimize, matching poselib's bundles; model selection goes
    # through pose_sampson_cheirality_score instead
    e = np.empty(9)
    essential_from_pose(model, e)
    return sampson_score(e, data, max_error_sq, best_score)


@njit(cache=True, fastmath=True, inline='always')
def _cheirality_block(model, e, x1_x, x1_y, x2_x, x2_y, start, count,
                      max_error_sq):
    score = 0.0
    num_inliers = 0
    for j in range(count):
        i = start + j
        x = x1_x[i]
        y = x1_y[i]
        xp = x2_x[i]
        yp = x2_y[i]
        r2, denominator = sampson_residual(e, x, y, xp, yp)
        if (denominator > 0.0 and r2 < max_error_sq * denominator
                and cheirality_ok(model, x, y, xp, yp, MIN_DEPTH)):
            score += r2 / denominator
            num_inliers += 1
        else:
            score += max_error_sq
    return score, num_inliers


@njit(cache=True, fastmath=True)
def pose_sampson_cheirality_score(model, data, max_error_sq, best_score):
    # poselib's compute_sampson_msac_score(CameraPose, ...): the truncated
    # Sampson score of E = [t]_x R, with a point counted as an inlier (and
    # charged its own residual rather than the truncation constant) only if
    # it also triangulates in front of both cameras. Same blocked early
    # bail-out as sampson_score - the score still only grows.
    x1_x, x1_y, x2_x, x2_y = data
    e = np.empty(9)
    essential_from_pose(model, e)
    n = x1_x.shape[0]
    score = 0.0
    num_inliers = 0
    num_full = n // SCORE_CHUNK
    for c in range(num_full):
        block_score, block_inliers = _cheirality_block(
            model, e, x1_x, x1_y, x2_x, x2_y, c * SCORE_CHUNK, SCORE_CHUNK,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
        if score >= best_score:
            return score, num_inliers
    tail = n - num_full * SCORE_CHUNK
    if tail > 0:
        block_score, block_inliers = _cheirality_block(
            model, e, x1_x, x1_y, x2_x, x2_y, num_full * SCORE_CHUNK, tail,
            max_error_sq)
        score += block_score
        num_inliers += block_inliers
    return score, num_inliers


def cheirality_mask_numpy(R, t, x1, x2, min_depth=MIN_DEPTH):
    # vectorized cheirality_ok over all correspondences
    x1h = np.column_stack([x1, np.ones(len(x1))])
    x2h = np.column_stack([x2, np.ones(len(x2))])
    u = x1h / np.linalg.norm(x1h, axis=1)[:, np.newaxis]
    v = x2h / np.linalg.norm(x2h, axis=1)[:, np.newaxis]
    ru = u @ np.asarray(R).T
    a = -np.sum(ru * v, axis=1)
    b1 = -(ru @ t)
    b2 = v @ t
    thr = min_depth * (1.0 - a * a)
    return (b1 - a * b2 > thr) & (b2 - a * b1 > thr)


class PoseSampsonScorer():
    # truncated Sampson error (MSAC) scorer for relative pose models
    # [R | t] (12 flat parameters, R row-major), cheirality-checked per point
    # exactly as poselib's CameraPose overloads are
    score = staticmethod(pose_sampson_cheirality_score)

    @staticmethod
    def score_numpy(R, t, x1, x2, max_error):
        E = np.array([[0.0, -t[2], t[1]],
                      [t[2], 0.0, -t[0]],
                      [-t[1], t[0], 0.0]]) @ R
        max_error_sq = max_error ** 2
        errors = sampson_errors_numpy(E, x1, x2)
        inliers = (errors <= max_error_sq) & cheirality_mask_numpy(R, t, x1, x2)
        return _msac(errors, inliers, max_error_sq), inliers, int(np.count_nonzero(inliers))


@njit(cache=True)
def calibrate_epipolar(e, pp1x, pp1y, pp2x, pp2y, inv1, inv2, f):
    # CPU wrapper: allocates the scratch the shared core wants passed in
    a = np.empty(9)
    _calibrate_epipolar_core(e, pp1x, pp1y, pp2x, pp2y, inv1, inv2, f, a)


@njit(cache=True)
def model_to_fundamental(model, pp1x, pp1y, pp2x, pp2y, f):
    # CPU wrapper: allocates the scratch the shared core wants passed in
    e = np.empty(9)
    a = np.empty(9)
    return _model_to_fundamental_core(model, pp1x, pp1y, pp2x, pp2y, f, e, a)


@njit(cache=True, fastmath=True)
def varying_focal_pose_sampson_score(model, data, max_error_sq, best_score):
    x1_x, x1_y, x2_x, x2_y, pp1x, pp1y, pp2x, pp2y = data
    f = np.empty(9)
    if not model_to_fundamental(model, pp1x, pp1y, pp2x, pp2y, f):
        return 1e300, 0
    return sampson_score(f, (x1_x, x1_y, x2_x, x2_y), max_error_sq, best_score)


class VaryingFocalPoseSampsonScorer():
    # truncated Sampson error for pose models [R | t | f1 | f2], evaluated
    # in the original image coordinate system with fixed principal points.
    score = staticmethod(varying_focal_pose_sampson_score)

    @staticmethod
    def score_numpy(R, t, f1, f2, pp1, pp2, x1, x2, max_error):
        E = np.array([[0.0, -t[2], t[1]],
                      [t[2], 0.0, -t[0]],
                      [-t[1], t[0], 0.0]]) @ R
        K1i = np.array([[1.0 / f1, 0.0, -pp1[0] / f1],
                        [0.0, 1.0 / f1, -pp1[1] / f1],
                        [0.0, 0.0, 1.0]])
        K2i = np.array([[1.0 / f2, 0.0, -pp2[0] / f2],
                        [0.0, 1.0 / f2, -pp2[1] / f2],
                        [0.0, 0.0, 1.0]])
        F = K2i.T @ E @ K1i
        return SampsonScorer.score_numpy(F, x1, x2, max_error)


@njit(cache=True, fastmath=True)
def shared_focal_pose_sampson_score(model, data, max_error_sq, best_score):
    # Same model layout as varying-focal, but the shared-focal solver/refiner
    # keeps model[12] == model[13].
    return varying_focal_pose_sampson_score(model, data, max_error_sq,
                                            best_score)


@njit(cache=True, fastmath=True)
def monodepth_pose_sampson_score(model, data, max_error_sq, best_score):
    # scorer for calibrated monodepth models [R | t | scale | shift1 |
    # shift2]: the truncated Sampson error of E = [t]_x R (the depth
    # parameters do not enter the scoring).
    #
    # Poselib's RelativePoseMonoDepthEstimator scores through the CameraPose
    # overload and so does apply the cheirality check that PoseSampsonScorer
    # now mirrors. It is deliberately not applied here: these models carry a
    # metric translation rather than a unit one, which changes what
    # MIN_DEPTH means, so aligning the monodepth estimators needs its own
    # look rather than a copy of the calibrated scorer.
    x1_x, x1_y, x2_x, x2_y = data[0], data[1], data[2], data[3]
    e = np.empty(9)
    essential_from_pose(model, e)
    return sampson_score(e, (x1_x, x1_y, x2_x, x2_y), max_error_sq, best_score)


class MonoDepthPoseSampsonScorer():
    # truncated Sampson error for calibrated monodepth models
    # [R | t | scale | shift1 | shift2]
    score = staticmethod(monodepth_pose_sampson_score)

    @staticmethod
    def score_numpy(R, t, x1, x2, max_error):
        # no cheirality check, matching monodepth_pose_sampson_score above
        E = np.array([[0.0, -t[2], t[1]],
                      [t[2], 0.0, -t[0]],
                      [-t[1], t[0], 0.0]]) @ R
        return SampsonScorer.score_numpy(E, x1, x2, max_error)


@njit(cache=True, fastmath=True)
def monodepth_focal_pose_sampson_score(model, data, max_error_sq, best_score):
    # scorer for monodepth focal models [R | t | f1 | f2 | scale] in
    # centered pixel coordinates: truncated Sampson error of the induced
    # F = K2^-T E K1^-1 (principal points already subtracted)
    x1_x, x1_y, x2_x, x2_y = data[0], data[1], data[2], data[3]
    f = np.empty(9)
    if not model_to_fundamental(model, 0.0, 0.0, 0.0, 0.0, f):
        return 1e300, 0
    return sampson_score(f, (x1_x, x1_y, x2_x, x2_y), max_error_sq, best_score)


class MonoDepthFocalPoseSampsonScorer():
    # truncated Sampson error for monodepth focal models
    # [R | t | f1 | f2 | scale] in centered pixel coordinates
    score = staticmethod(monodepth_focal_pose_sampson_score)

    @staticmethod
    def score_numpy(R, t, f1, f2, x1, x2, max_error):
        zero = np.zeros(2)
        return VaryingFocalPoseSampsonScorer.score_numpy(
            R, t, f1, f2, zero, zero, x1, x2, max_error)


class SharedFocalPoseSampsonScorer():
    # truncated Sampson error for pose models [R | t | f | f], evaluated in
    # image coordinates with fixed principal points.
    score = staticmethod(shared_focal_pose_sampson_score)

    @staticmethod
    def score_numpy(R, t, f, pp1, pp2, x1, x2, max_error):
        return VaryingFocalPoseSampsonScorer.score_numpy(
            R, t, f, f, pp1, pp2, x1, x2, max_error)


# ---------------------------------------------------------------------------
# loss-selectable cost kernels for the final refinement pass (LM accept/
# reject only; RANSAC model selection above always stays truncated MSAC via
# the *_score functions, so their early per-block bail-out stays valid).
# ---------------------------------------------------------------------------

def build_sampson_cost(loss):
    # generalizes sampson_score's truncated capped-quadratic into an
    # arbitrary loss.cost(r2, max_error_sq); always divides (no need for the
    # truncated fast path's skip-the-division trick since this is only used
    # inside the LM loop, not the O(ransac_iterations x n) scorer)
    cost_fn = loss.cost

    @njit(cache=True, fastmath=True)
    def _sampson_cost(f, data, max_error_sq, best_score):
        x1_x, x1_y, x2_x, x2_y = data
        n = x1_x.shape[0]
        score = 0.0
        num_inliers = 0
        for i in range(n):
            x = x1_x[i]
            y = x1_y[i]
            xp = x2_x[i]
            yp = x2_y[i]
            fx1_0 = f[0] * x + f[1] * y + f[2]
            fx1_1 = f[3] * x + f[4] * y + f[5]
            fx1_2 = f[6] * x + f[7] * y + f[8]
            ftx2_0 = f[0] * xp + f[3] * yp + f[6]
            ftx2_1 = f[1] * xp + f[4] * yp + f[7]
            residual = xp * fx1_0 + yp * fx1_1 + fx1_2
            denominator = (fx1_0 * fx1_0 + fx1_1 * fx1_1
                           + ftx2_0 * ftx2_0 + ftx2_1 * ftx2_1)
            if denominator > 0.0:
                r2 = residual * residual / denominator
                score += cost_fn(r2, max_error_sq)
                if r2 < max_error_sq:
                    num_inliers += 1
            else:
                score += cost_fn(1e18, max_error_sq)
        return score, num_inliers

    return _sampson_cost


def build_pose_sampson_cost(loss):
    sampson_cost = build_sampson_cost(loss)

    @njit(cache=True, fastmath=True)
    def _pose_sampson_cost(model, data, max_error_sq, best_score):
        e = np.empty(9)
        essential_from_pose(model, e)
        return sampson_cost(e, data, max_error_sq, best_score)

    return _pose_sampson_cost


def build_varying_focal_pose_sampson_cost(loss):
    # shared by the shared-focal refiner too (same model layout, mirroring
    # shared_focal_pose_sampson_score reusing varying_focal_pose_sampson_score)
    sampson_cost = build_sampson_cost(loss)

    @njit(cache=True, fastmath=True)
    def _varying_focal_pose_sampson_cost(model, data, max_error_sq, best_score):
        x1_x, x1_y, x2_x, x2_y, pp1x, pp1y, pp2x, pp2y = data
        f = np.empty(9)
        if not model_to_fundamental(model, pp1x, pp1y, pp2x, pp2y, f):
            return 1e300, 0
        return sampson_cost(f, (x1_x, x1_y, x2_x, x2_y), max_error_sq, best_score)

    return _varying_focal_pose_sampson_cost
