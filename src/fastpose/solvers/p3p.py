"""Minimal P3P solver for calibrated absolute pose.

Port of the P3P solver of Ding et al. from PoseLib (p3p.cc, authors
Yaqing Ding and Mark Shachkov, BSD-3-Clause, Copyright (c) 2020 Viktor
Larsson): "Revisiting the P3P Problem", CVPR 2023. The three 2D-3D
correspondences are reduced to a single cubic; the rank-2 conic pencil is
split into two lines, each yielding a quadratic in the depth ratio, and the
depths are polished with a few Newton steps before the pose is assembled.

Pose models are `[R | t]` (12 flat parameters, R row-major) with
lambda * (x, y, 1) = R X + t for calibrated image points (x, y).

`data` layout for this problem: five contiguous float64 columns
(x_x, x_y, X_x, X_y, X_z) - calibrated 2D points and 3D world points.

The kernels are built by `build_p3p_kernels(jit)` so the same source compiles
for the CPU (`njit`) and for CUDA (`cuda.jit(device=True)`); see
fastpose/jit_backend.py. `_solve_p3p_core` takes every scratch array
pre-shaped because numba's runtime is host-only, so neither `np.empty` nor
`.reshape` compiles in device code - the CPU wrapper `_solve_p3p` below
carves them out of the driver's flat workspace, the CUDA kernel allocates
them with `cuda.local.array`.
"""

import math

from numba import njit

from fastpose.jit_backend import cpu_jit

MODEL_SIZE = 12


def build_p3p_kernels(jit):
    @jit(inline=True)
    def _cubic_single_real(c2, c1, c0):
        # one real root of x^3 + c2*x^2 + c1*x + c0 (Cardano / trigonometric);
        # second return value is True iff the root is the only real one
        a = c1 - c2 * c2 / 3.0
        b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0
        c = b * b / 4.0 + a * a * a / 27.0
        if c > 0.0:
            c = math.sqrt(c)
            b *= -0.5
            u = b + c
            v = b - c
            root = (math.copysign(abs(u) ** (1.0 / 3.0), u)
                    + math.copysign(abs(v) ** (1.0 / 3.0), v) - c2 / 3.0)
            return root, True
        elif c < 0.0:
            arg = 3.0 * b / (2.0 * a) * math.sqrt(-3.0 / a)
            if arg > 1.0:
                arg = 1.0
            elif arg < -1.0:
                arg = -1.0
            root = 2.0 * math.sqrt(-a / 3.0) * math.cos(math.acos(arg) / 3.0) - c2 / 3.0
        else:
            root = -c2 / 3.0
            if a != 0.0:
                root += 3.0 * b / a
        return root, False

    @jit(inline=True)
    def _root2real(b, c, r):
        # roots of x^2 + b*x + c with a tolerance for marginally negative
        # discriminants; r2 = -2 marks a rejected (negative) placeholder root
        v = b * b - 4.0 * c
        if v < -1e-12:
            r[0] = -0.5 * b
            r[1] = -0.5 * b
            return False
        if v < 0.0:
            r[0] = -0.5 * b
            r[1] = -2.0
            return True
        y = math.sqrt(v)
        if b < 0.0:
            r[0] = 0.5 * (-b + y)
            r[1] = 0.5 * (-b - y)
        else:
            r[0] = 2.0 * c / (-b + y)
            r[1] = 2.0 * c / (-b - y)
        return True

    @jit(inline=True)
    def _refine_lambda(l1, l2, l3, a01, a02, a12, b01, b02, b12):
        # Newton polish of the three depths on the pairwise distance equations
        # |l_i x_i - l_j x_j|^2 = a_ij, using the closed-form jacobian inverse
        for _ in range(5):
            r1 = l1 * l1 - 2.0 * l1 * l2 * b01 + l2 * l2 - a01
            r2 = l1 * l1 - 2.0 * l1 * l3 * b02 + l3 * l3 - a02
            r3 = l2 * l2 - 2.0 * l2 * l3 * b12 + l3 * l3 - a12
            if abs(r1) + abs(r2) + abs(r3) < 1e-10:
                return l1, l2, l3
            x11 = l1 - l2 * b01
            x12 = l2 - l1 * b01
            x21 = l1 - l3 * b02
            x23 = l3 - l1 * b02
            x32 = l2 - l3 * b12
            x33 = l3 - l2 * b12
            det = x11 * x23 * x32 + x12 * x21 * x33
            if det == 0.0:
                break
            detJ = 0.5 / det
            l1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ
            l2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ
            l3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ
        return l1, l2, l3

    @jit(inline=True)
    def _write_pose(xs, Xs, X01, X02, XXinv, d0, d1, d2, pose):
        # R maps [X01 | X02 | X01 x X02] to the same triplet built from the
        # camera-frame vectors v1 = d0*x0 - d1*x1, v2 = d0*x0 - d2*x2
        v10 = d0 * xs[0, 0] - d1 * xs[1, 0]
        v11 = d0 * xs[0, 1] - d1 * xs[1, 1]
        v12 = d0 * xs[0, 2] - d1 * xs[1, 2]
        v20 = d0 * xs[0, 0] - d2 * xs[2, 0]
        v21 = d0 * xs[0, 1] - d2 * xs[2, 1]
        v22 = d0 * xs[0, 2] - d2 * xs[2, 2]
        c0 = v11 * v22 - v12 * v21
        c1 = v12 * v20 - v10 * v22
        c2 = v10 * v21 - v11 * v20
        # R = [v1 | v2 | v1 x v2] @ XXinv (all as column-stacked matrices)
        for j in range(3):
            pose[j] = v10 * XXinv[0, j] + v20 * XXinv[1, j] + c0 * XXinv[2, j]
            pose[3 + j] = v11 * XXinv[0, j] + v21 * XXinv[1, j] + c1 * XXinv[2, j]
            pose[6 + j] = v12 * XXinv[0, j] + v22 * XXinv[1, j] + c2 * XXinv[2, j]
        for i in range(3):
            pose[9 + i] = (d0 * xs[0, i]
                           - pose[3 * i] * Xs[0, 0]
                           - pose[3 * i + 1] * Xs[0, 1]
                           - pose[3 * i + 2] * Xs[0, 2])

    @jit(fastmath=True)
    def _p3p_impl(xs, Xs, X01, X02, XXinv, C, pq, taus, models):
        # P3P core on prepared buffers: xs (3, 3) unit bearing vectors, Xs (3, 3)
        # 3D points (both mutated - the points are reordered internally), the
        # rest scratch. Writes up to 4 poses into models[count, 0:12] and returns
        # their count. Kept separate from the data/sample plumbing so other
        # problems (e.g. monodepth relative pose) can reuse the solver.
        a01 = 0.0
        a02 = 0.0
        a12 = 0.0
        for j in range(3):
            d01 = Xs[0, j] - Xs[1, j]
            d02 = Xs[0, j] - Xs[2, j]
            d12 = Xs[1, j] - Xs[2, j]
            a01 += d01 * d01
            a02 += d02 * d02
            a12 += d12 * d12

        # reorder so that the pair (1, 2) has the largest 3D distance
        swap_with = -1
        if a01 > a02:
            if a01 > a12:
                swap_with = 2
        elif a02 > a12:
            swap_with = 1
        if swap_with >= 0:
            for j in range(3):
                tmp = xs[0, j]
                xs[0, j] = xs[swap_with, j]
                xs[swap_with, j] = tmp
                tmp = Xs[0, j]
                Xs[0, j] = Xs[swap_with, j]
                Xs[swap_with, j] = tmp
        a01 = 0.0
        a02 = 0.0
        a12 = 0.0
        for j in range(3):
            X01[j] = Xs[0, j] - Xs[1, j]
            X02[j] = Xs[0, j] - Xs[2, j]
            d12 = Xs[1, j] - Xs[2, j]
            a01 += X01[j] * X01[j]
            a02 += X02[j] * X02[j]
            a12 += d12 * d12
        if a12 <= 0.0:
            return 0  # coincident 3D points

        # inverse of [X01 | X02 | X01 x X02]; the determinant is |X01 x X02|^2
        cx0 = X01[1] * X02[2] - X01[2] * X02[1]
        cx1 = X01[2] * X02[0] - X01[0] * X02[2]
        cx2 = X01[0] * X02[1] - X01[1] * X02[0]
        det = cx0 * cx0 + cx1 * cx1 + cx2 * cx2
        if det <= 0.0:
            return 0  # collinear 3D points
        inv_det = 1.0 / det
        # rows of the inverse are the cross products of the other two columns
        XXinv[0, 0] = (X02[1] * cx2 - X02[2] * cx1) * inv_det
        XXinv[0, 1] = (X02[2] * cx0 - X02[0] * cx2) * inv_det
        XXinv[0, 2] = (X02[0] * cx1 - X02[1] * cx0) * inv_det
        XXinv[1, 0] = (cx1 * X01[2] - cx2 * X01[1]) * inv_det
        XXinv[1, 1] = (cx2 * X01[0] - cx0 * X01[2]) * inv_det
        XXinv[1, 2] = (cx0 * X01[1] - cx1 * X01[0]) * inv_det
        XXinv[2, 0] = cx0 * inv_det
        XXinv[2, 1] = cx1 * inv_det
        XXinv[2, 2] = cx2 * inv_det

        a12d = 1.0 / a12
        a = a01 * a12d
        b = a02 * a12d

        m01 = xs[0, 0] * xs[1, 0] + xs[0, 1] * xs[1, 1] + xs[0, 2] * xs[1, 2]
        m02 = xs[0, 0] * xs[2, 0] + xs[0, 1] * xs[2, 1] + xs[0, 2] * xs[2, 2]
        m12 = xs[1, 0] * xs[2, 0] + xs[1, 1] * xs[2, 1] + xs[1, 2] * xs[2, 2]

        m12sq = 1.0 - m12 * m12
        m02sq = m02 * m02 - 1.0
        m01sq = m01 * m01 - 1.0
        ab = a * b
        bsq = b * b
        asq = a * a
        m013 = -2.0 + 2.0 * m01 * m02 * m12
        bsqm12sq = bsq * m12sq
        asqm12sq = asq * m12sq
        abm12sq = 2.0 * ab * m12sq

        k3 = bsqm12sq + b * m02sq
        if k3 == 0.0:
            return 0
        k3_inv = 1.0 / k3
        k2 = k3_inv * ((a - 1.0) * m02sq + abm12sq + bsqm12sq + b * m013)
        k1 = k3_inv * (asqm12sq + abm12sq + a * m013 + (b - 1.0) * m01sq)
        k0 = k3_inv * (asqm12sq + a * m01sq)

        s, G = _cubic_single_real(k2, k1, k0)

        C[0, 0] = -a + s * (1.0 - b)
        C[0, 1] = -m02 * s
        C[0, 2] = a * m12 + b * m12 * s
        C[1, 0] = C[0, 1]
        C[1, 1] = s + 1.0
        C[1, 2] = -m01
        C[2, 0] = C[0, 2]
        C[2, 1] = C[1, 2]
        C[2, 2] = -a - b * s + 1.0

        # split the rank-2 conic C into two lines p, q via C +/- skew(v) with
        # v from the best-conditioned column of the (negated) adjugate
        adj00 = C[1, 2] * C[2, 1] - C[1, 1] * C[2, 2]
        adj11 = C[0, 2] * C[2, 0] - C[0, 0] * C[2, 2]
        adj22 = C[0, 1] * C[1, 0] - C[0, 0] * C[1, 1]
        adj01 = C[0, 1] * C[2, 2] - C[0, 2] * C[2, 1]
        adj02 = C[0, 2] * C[1, 1] - C[0, 1] * C[1, 2]
        adj12 = C[0, 0] * C[1, 2] - C[0, 2] * C[1, 0]

        if adj00 > adj11:
            if adj00 > adj22:
                diag = adj00
                v0 = adj00
                v1 = adj01
                v2 = adj02
            else:
                diag = adj22
                v0 = adj02
                v1 = adj12
                v2 = adj22
        elif adj11 > adj22:
            diag = adj11
            v0 = adj01
            v1 = adj11
            v2 = adj12
        else:
            diag = adj22
            v0 = adj02
            v1 = adj12
            v2 = adj22
        if diag <= 0.0:
            return 0  # degenerate conic
        inv = 1.0 / math.sqrt(diag)
        v0 *= inv
        v1 *= inv
        v2 *= inv

        pq[0, 0] = C[0, 0]
        pq[0, 1] = C[1, 0] + v2
        pq[0, 2] = C[2, 0] - v1
        pq[1, 0] = C[0, 0]
        pq[1, 1] = C[0, 1] - v2
        pq[1, 2] = C[0, 2] + v1

        count = 0
        for i in range(2):
            p0 = pq[i, 0]
            p1 = pq[i, 1]
            p2 = pq[i, 2]
            # [p0 p1 p2] * [d2; d0; d1] = 0; eliminate d0 or d1 depending on
            # which of p0, p1 is better conditioned
            if abs(p0) <= abs(p1):
                w0 = -p0 / p1
                w1 = -p2 / p1
                den = w1 * w1 - b
                if den == 0.0:
                    continue
                ca = 1.0 / den
                cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca
                cc = (w0 * w0 - 2.0 * m02 * w0 - b + 1.0) * ca
                if not _root2real(cb, cc, taus):
                    continue
                for ti in range(2):
                    tau = taus[ti]
                    if tau <= 0.0:
                        continue
                    den = tau * (tau - 2.0 * m12) + 1.0
                    if den <= 0.0:
                        continue
                    d2 = math.sqrt(a12 / den)
                    d1 = tau * d2
                    d0 = w0 * d2 + w1 * d1
                    if d0 < 0.0:
                        continue
                    d0, d1, d2 = _refine_lambda(d0, d1, d2, a01, a02, a12,
                                                m01, m02, m12)
                    _write_pose(xs, Xs, X01, X02, XXinv, d0, d1, d2, models[count])
                    count += 1
            else:
                w0 = -p1 / p0
                w1 = -p2 / p0
                den = -a * w1 * w1 + 2.0 * a * m12 * w1 - a + 1.0
                if den == 0.0:
                    continue
                ca = 1.0 / den
                cb = 2.0 * (a * m12 * w0 - m01 - a * w0 * w1) * ca
                cc = (1.0 - a * w0 * w0) * ca
                if not _root2real(cb, cc, taus):
                    continue
                for ti in range(2):
                    tau = taus[ti]
                    if tau <= 0.0:
                        continue
                    den = tau * (tau - 2.0 * m01) + 1.0
                    if den <= 0.0:
                        continue
                    d0 = math.sqrt(a01 / den)
                    d1 = tau * d0
                    d2 = w0 * d0 + w1 * d1
                    if d2 < 0.0:
                        continue
                    d0, d1, d2 = _refine_lambda(d0, d1, d2, a01, a02, a12,
                                                m01, m02, m12)
                    _write_pose(xs, Xs, X01, X02, XXinv, d0, d1, d2, models[count])
                    count += 1

            # a unique real cubic root means the second line adds no solutions
            if count > 0 and G:
                break

        return count

    @jit(fastmath=True)
    def _solve_p3p_core(data, sample, models, xs, Xs, X01, X02, XXinv, C, pq,
                        taus):
        # scratch is passed in pre-shaped: see the note above the factory
        x_x, x_y, X_x, X_y, X_z = data

        for k in range(3):
            i = sample[k]
            nx = x_x[i]
            ny = x_y[i]
            inv = 1.0 / math.sqrt(nx * nx + ny * ny + 1.0)
            xs[k, 0] = nx * inv
            xs[k, 1] = ny * inv
            xs[k, 2] = inv
            Xs[k, 0] = X_x[i]
            Xs[k, 1] = X_y[i]
            Xs[k, 2] = X_z[i]

        return _p3p_impl(xs, Xs, X01, X02, XXinv, C, pq, taus, models)

    return {
        'cubic_single_real': _cubic_single_real,
        'root2real': _root2real,
        'refine_lambda': _refine_lambda,
        'write_pose': _write_pose,
        'p3p_impl': _p3p_impl,
        'solve_p3p_core': _solve_p3p_core,
    }


_CPU = build_p3p_kernels(cpu_jit)

# module-level names the other solvers and the tests import; these are the CPU
# kernels, byte-for-byte the same source as before the factory
_cubic_single_real = _CPU['cubic_single_real']
_root2real = _CPU['root2real']
_refine_lambda = _CPU['refine_lambda']
_write_pose = _CPU['write_pose']
_p3p_impl = _CPU['p3p_impl']
_solve_p3p_core = _CPU['solve_p3p_core']


@njit(cache=True, fastmath=True)
def _solve_p3p(data, sample, models, workspace):
    # RANSAC-driver entry point: carves the flat workspace into the pre-shaped
    # scratch `_solve_p3p_core` wants. CPU-only - `.reshape` does not compile
    # for CUDA, where the kernel allocates the same pieces with
    # `cuda.local.array` instead.
    xs = workspace[0:9].reshape(3, 3)
    Xs = workspace[9:18].reshape(3, 3)
    X01 = workspace[18:21]
    X02 = workspace[21:24]
    XXinv = workspace[24:33].reshape(3, 3)
    C = workspace[33:42].reshape(3, 3)
    pq = workspace[42:48].reshape(2, 3)
    taus = workspace[48:50]
    return _solve_p3p_core(data, sample, models, xs, Xs, X01, X02, XXinv, C,
                           pq, taus)


class P3PSolver():
    # minimal solver for calibrated absolute pose from 3 correspondences
    sample_size = 3
    num_params = MODEL_SIZE
    max_models = 4
    workspace_size = 50
    solve = staticmethod(_solve_p3p)
