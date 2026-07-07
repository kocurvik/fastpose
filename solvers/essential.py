"""Minimal 5-point solver for calibrated relative pose.

The minimal solver follows Stewenius' formulation: a 4-dimensional nullspace
E = x*E1 + y*E2 + z*E3 + w*E4 (w=1), the ten cubic constraints det(E) = 0 and
2*E*E^T*E - trace(E*E^T)*E = 0 expanded into a 10x20 coefficient matrix,
Gauss-Jordan elimination, and the 10x10 action matrix for multiplication by
x. Because numba's np.linalg.eig cannot return complex eigenvalues, the real
eigenvalues are extracted like poselib does it: characteristic polynomial via
Danilevsky's method and Sturm-sequence root bracketing; the eigenvectors then
follow from a nullspace computation per real root.

Each essential matrix candidate is decomposed directly into a pose: of the
four (R, t) candidates of E, the one with the best cheirality count on the
minimal sample is kept, so the solver outputs pose models [R | t]
(12 flat parameters, R row-major, t unit norm) with x2 ~ R x1 + t.

`data` layout: identical to the fundamental solver, a tuple of four
contiguous float64 columns of *calibrated* (normalized) image points.
"""

import math

import numpy as np
from numba import njit

from solvers.utils import fill_epipolar_matrix


# ---------------------------------------------------------------------------
# monomial multiplication tables (built once in plain Python)
#
# variables ordered (x, y, z, 1); degree-2 monomial basis (10):
#   x2, xy, xz, y2, yz, z2, x, y, z, 1
# degree-3 column basis (20): the ten degree-3 monomials first,
#   x3, x2y, x2z, xy2, xyz, xz2, y3, y2z, yz2, z3,
# followed by the ten degree<=2 monomials above (the quotient ring basis)
# ---------------------------------------------------------------------------

def _build_mul_tables():
    var_exps = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
    deg2 = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2),
            (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
    deg3 = [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1), (1, 0, 2),
            (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)]
    idx2 = {m: i for i, m in enumerate(deg2)}
    idx3 = {m: i for i, m in enumerate(deg3 + deg2)}

    t44 = np.empty((4, 4), dtype=np.int64)
    for i, a in enumerate(var_exps):
        for j, b in enumerate(var_exps):
            t44[i, j] = idx2[(a[0] + b[0], a[1] + b[1], a[2] + b[2])]
    t104 = np.empty((10, 4), dtype=np.int64)
    for i, a in enumerate(deg2):
        for j, b in enumerate(var_exps):
            t104[i, j] = idx3[(a[0] + b[0], a[1] + b[1], a[2] + b[2])]
    return t44, t104


_T44, _T104 = _build_mul_tables()


@njit(cache=True, inline='always', fastmath=True)
def _poly1_mul_acc(a, b, factor, out):
    # out (10,) += factor * (a * b) for degree-1 polys a, b (4 coeffs each)
    for i in range(4):
        fa = factor * a[i]
        if fa != 0.0:
            for j in range(4):
                out[_T44[i, j]] += fa * b[j]


@njit(cache=True, inline='always', fastmath=True)
def _poly2_mul1_acc(a, b, factor, out):
    # out (20,) += factor * (a * b) for degree-2 a (10 coeffs), degree-1 b (4)
    for i in range(10):
        fa = factor * a[i]
        if fa != 0.0:
            for j in range(4):
                out[_T104[i, j]] += fa * b[j]


@njit(cache=True, inline='always')
def _load_entry(N, i, j, out):
    # coefficient vector (x, y, z, 1) of entry E[i, j] of the nullspace pencil
    k = 3 * i + j
    out[0] = N[0, k]
    out[1] = N[1, k]
    out[2] = N[2, k]
    out[3] = N[3, k]


# ---------------------------------------------------------------------------
# solver kernels
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _nullspace_5pt(A, N):
    # 4-dimensional nullspace of the 5x9 epipolar constraint matrix via
    # Gaussian elimination (pivots on columns 0..4, free variables 5..8),
    # then modified Gram-Schmidt for a well-conditioned basis
    for col in range(5):
        piv = col
        max_val = abs(A[col, col])
        for r in range(col + 1, 5):
            v = abs(A[r, col])
            if v > max_val:
                max_val = v
                piv = r
        if max_val < 1e-12:
            return False
        if piv != col:
            for c in range(col, 9):
                t = A[col, c]
                A[col, c] = A[piv, c]
                A[piv, c] = t
        inv = 1.0 / A[col, col]
        for r in range(col + 1, 5):
            factor = A[r, col] * inv
            if factor != 0.0:
                A[r, col] = 0.0
                for c in range(col + 1, 9):
                    A[r, c] -= factor * A[col, c]

    for b in range(4):
        free_col = 5 + b
        for j in range(9):
            N[b, j] = 0.0
        N[b, free_col] = 1.0
        for r in range(4, -1, -1):
            s = A[r, free_col]
            for c in range(r + 1, 5):
                s += A[r, c] * N[b, c]
            N[b, r] = -s / A[r, r]

    for b in range(4):
        for k in range(b):
            dot = 0.0
            for j in range(9):
                dot += N[b, j] * N[k, j]
            for j in range(9):
                N[b, j] -= dot * N[k, j]
        norm = 0.0
        for j in range(9):
            norm += N[b, j] * N[b, j]
        if norm < 1e-24:
            return False
        inv = 1.0 / math.sqrt(norm)
        for j in range(9):
            N[b, j] *= inv
    return True


@njit(cache=True, fastmath=True)
def _build_constraints(N, M, EEt, tr, ct, av, bv):
    # expand the ten cubic constraints on E = x*N0 + y*N1 + z*N2 + N3 into
    # the 10x20 coefficient matrix M
    for r in range(10):
        for c in range(20):
            M[r, c] = 0.0

    # row 0: det(E) via cofactor expansion along the first row
    for r in range(3):
        for c in range(10):
            ct[r, c] = 0.0
    _load_entry(N, 1, 1, av)
    _load_entry(N, 2, 2, bv)
    _poly1_mul_acc(av, bv, 1.0, ct[0])
    _load_entry(N, 1, 2, av)
    _load_entry(N, 2, 1, bv)
    _poly1_mul_acc(av, bv, -1.0, ct[0])
    _load_entry(N, 1, 2, av)
    _load_entry(N, 2, 0, bv)
    _poly1_mul_acc(av, bv, 1.0, ct[1])
    _load_entry(N, 1, 0, av)
    _load_entry(N, 2, 2, bv)
    _poly1_mul_acc(av, bv, -1.0, ct[1])
    _load_entry(N, 1, 0, av)
    _load_entry(N, 2, 1, bv)
    _poly1_mul_acc(av, bv, 1.0, ct[2])
    _load_entry(N, 1, 1, av)
    _load_entry(N, 2, 0, bv)
    _poly1_mul_acc(av, bv, -1.0, ct[2])
    for k in range(3):
        _load_entry(N, 0, k, av)
        _poly2_mul1_acc(ct[k], av, 1.0, M[0])

    # rows 1..9: 2*E*E^T*E - trace(E*E^T)*E
    for i in range(3):
        for j in range(3):
            for c in range(10):
                EEt[i, j, c] = 0.0
            for k in range(3):
                _load_entry(N, i, k, av)
                _load_entry(N, j, k, bv)
                _poly1_mul_acc(av, bv, 1.0, EEt[i, j])
    for c in range(10):
        tr[c] = EEt[0, 0, c] + EEt[1, 1, c] + EEt[2, 2, c]

    for i in range(3):
        for j in range(3):
            row = M[1 + 3 * i + j]
            for k in range(3):
                _load_entry(N, k, j, av)
                _poly2_mul1_acc(EEt[i, k], av, 2.0, row)
            _load_entry(N, i, j, av)
            _poly2_mul1_acc(tr, av, -1.0, row)


@njit(cache=True, fastmath=True)
def _gauss_jordan_10x20(M):
    # reduce M to [I | B] with partial pivoting
    scale = 0.0
    for r in range(10):
        for c in range(20):
            v = abs(M[r, c])
            if v > scale:
                scale = v
    if scale == 0.0:
        return False
    tol = 1e-12 * scale
    for col in range(10):
        piv = col
        max_val = abs(M[col, col])
        for r in range(col + 1, 10):
            v = abs(M[r, col])
            if v > max_val:
                max_val = v
                piv = r
        if max_val < tol:
            return False
        if piv != col:
            for c in range(col, 20):
                t = M[col, c]
                M[col, c] = M[piv, c]
                M[piv, c] = t
        inv = 1.0 / M[col, col]
        for c in range(col, 20):
            M[col, c] *= inv
        for r in range(10):
            if r != col:
                factor = M[r, col]
                if factor != 0.0:
                    M[r, col] = 0.0
                    for c in range(col + 1, 20):
                        M[r, c] -= factor * M[col, c]
    return True


@njit(cache=True)
def _action_matrix(M, T):
    # action matrix for multiplication by x in the quotient ring basis
    # (x2, xy, xz, y2, yz, z2, x, y, z, 1); x * basis[0..5] are the degree-3
    # monomials in rows 0..5 of the reduced system, the rest are basis shifts
    for r in range(10):
        for c in range(10):
            T[r, c] = 0.0
    for r in range(6):
        for c in range(10):
            T[r, c] = -M[r, 10 + c]
    T[6, 0] = 1.0
    T[7, 1] = 1.0
    T[8, 2] = 1.0
    T[9, 6] = 1.0


@njit(cache=True, fastmath=True)
def _charpoly_danilevsky(T, coef, row, tmp_row):
    # characteristic polynomial of the 10x10 matrix T (destroyed) via
    # Danilevsky's method with pivoting; coef ascending, coef[10] = 1,
    # p(x) = x^10 - T[0,0] x^9 - T[0,1] x^8 - ... - T[0,9]
    n = 10
    for i in range(n - 1, 0, -1):
        piv_ind = i - 1
        piv = abs(T[i, i - 1])
        for j in range(i - 1):
            v = abs(T[i, j])
            if v > piv:
                piv = v
                piv_ind = j
        if piv < 1e-14:
            return False
        if piv_ind != i - 1:
            for c in range(n):
                t = T[piv_ind, c]
                T[piv_ind, c] = T[i - 1, c]
                T[i - 1, c] = t
            for r in range(n):
                t = T[r, piv_ind]
                T[r, piv_ind] = T[r, i - 1]
                T[r, i - 1] = t

        for c in range(n):
            row[c] = T[i, c]
        inv = 1.0 / row[i - 1]

        # similarity transform: column operations (T <- T M^-1) on rows 0..i
        for r in range(i + 1):
            colv = T[r, i - 1]
            if colv != 0.0:
                f = colv * inv
                for c in range(n):
                    T[r, c] -= row[c] * f
                # the loop above also subtracted from column i-1 itself
                T[r, i - 1] = f
            else:
                T[r, i - 1] = 0.0

        # row operation (T <- M T): row i-1 becomes row[.] combination
        for c in range(n):
            s = 0.0
            for k in range(n):
                s += row[k] * T[k, c]
            tmp_row[c] = s
        for c in range(n):
            T[i - 1, c] = tmp_row[c]

        # row i is exactly e_{i-1} now; set it to clean numerical noise
        for c in range(n):
            T[i, c] = 0.0
        T[i, i - 1] = 1.0

    coef[n] = 1.0
    for k in range(n):
        coef[n - 1 - k] = -T[0, k]
    return True


@njit(cache=True, inline='always', fastmath=True)
def _poly_eval(c, deg, x):
    v = c[deg]
    for d in range(deg - 1, -1, -1):
        v = v * x + c[d]
    return v


@njit(cache=True, fastmath=True)
def _sturm_sign_changes(chain, degs, nchain, x):
    count = 0
    prev = 0
    for k in range(nchain):
        v = _poly_eval(chain[k], degs[k], x)
        if v > 0.0:
            s = 1
        elif v < 0.0:
            s = -1
        else:
            s = 0
        if s != 0:
            if prev != 0 and s != prev:
                count += 1
            prev = s
    return count


@njit(cache=True, fastmath=True)
def _real_roots_sturm(coef, degree, chain, degs, roots, lo_stack, hi_stack,
                      slo_stack, shi_stack):
    # all real roots of the polynomial with ascending coefficients coef
    # (length degree+1) via a Sturm chain and interval bisection
    max_c = 0.0
    for d in range(degree + 1):
        v = abs(coef[d])
        if v > max_c:
            max_c = v
    if max_c == 0.0:
        return 0
    n = degree
    while n > 0 and abs(coef[n]) < 1e-13 * max_c:
        n -= 1
    if n < 1:
        return 0

    for r in range(degree + 1):
        for c in range(degree + 1):
            chain[r, c] = 0.0
    for d in range(n + 1):
        chain[0, d] = coef[d] / max_c
    degs[0] = n
    for d in range(n):
        chain[1, d] = chain[0, d + 1] * (d + 1)
    degs[1] = n - 1
    nchain = 2
    while degs[nchain - 1] > 0 and nchain <= degree:
        k = nchain
        dl = degs[k - 2]
        dm = degs[k - 1]
        for d in range(dl + 1):
            chain[k, d] = chain[k - 2, d]
        lead = chain[k - 1, dm]
        for d in range(dl, dm - 1, -1):
            q = chain[k, d] / lead
            chain[k, d] = 0.0
            if q != 0.0:
                for j in range(dm):
                    chain[k, d - dm + j] -= q * chain[k - 1, j]
        rem_max = 0.0
        for d in range(dm):
            v = abs(chain[k, d])
            if v > rem_max:
                rem_max = v
        if rem_max < 1e-13:
            break  # nontrivial gcd (multiple roots); chain ends here
        dr = dm - 1
        while dr > 0 and abs(chain[k, dr]) < 1e-13 * rem_max:
            chain[k, dr] = 0.0
            dr -= 1
        inv = 1.0 / rem_max
        for d in range(dr + 1):
            chain[k, d] = -chain[k, d] * inv
        degs[k] = dr
        nchain += 1

    # Fujiwara bound on root magnitudes for the monic-normalized polynomial
    lead = chain[0, n]
    bound = 0.0
    for d in range(n):
        v = abs(chain[0, d] / lead) ** (1.0 / (n - d))
        if v > bound:
            bound = v
    bound = 2.0 * bound + 1.0

    sp = 0
    lo_stack[0] = -bound
    hi_stack[0] = bound
    slo_stack[0] = _sturm_sign_changes(chain, degs, nchain, -bound)
    shi_stack[0] = _sturm_sign_changes(chain, degs, nchain, bound)
    sp = 1
    num_roots = 0
    while sp > 0:
        sp -= 1
        lo = lo_stack[sp]
        hi = hi_stack[sp]
        s_lo = slo_stack[sp]
        s_hi = shi_stack[sp]
        nr = s_lo - s_hi
        if nr <= 0:
            continue
        if nr == 1:
            p_lo = _poly_eval(chain[0], n, lo)
            p_hi = _poly_eval(chain[0], n, hi)
            if (p_lo < 0.0) != (p_hi < 0.0):
                # safeguarded Newton-bisection on the polynomial sign: the
                # bracket is always maintained and the midpoint is the
                # fallback, so this is never less robust than bisection but
                # converges quadratically for simple roots
                x = 0.5 * (lo + hi)
                for _ in range(80):
                    if hi - lo < 1e-13 * (1.0 + abs(x)):
                        break
                    p = _poly_eval(chain[0], n, x)
                    if p == 0.0:
                        break
                    if (p < 0.0) == (p_lo < 0.0):
                        lo = x
                    else:
                        hi = x
                    dp = _poly_eval(chain[1], n - 1, x)
                    x_new = 0.5 * (lo + hi)
                    if dp != 0.0:
                        step = x - p / dp
                        if lo < step < hi:
                            x_new = step
                    if x_new == x:
                        break
                    x = x_new
            else:
                # even multiplicity in the squarefree sense is impossible,
                # but the endpoints may straddle awkwardly: bisect on the
                # Sturm root count instead
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if hi - lo < 1e-11 * (1.0 + abs(mid)):
                        break
                    s_mid = _sturm_sign_changes(chain, degs, nchain, mid)
                    if s_lo - s_mid == 1:
                        hi = mid
                        s_hi = s_mid
                    else:
                        lo = mid
                        s_lo = s_mid
                x = 0.5 * (lo + hi)
            # Newton polish
            for _ in range(3):
                p = _poly_eval(chain[0], n, x)
                dp = _poly_eval(chain[1], n - 1, x)
                if dp != 0.0:
                    x_new = x - p / dp
                    if lo <= x_new <= hi:
                        x = x_new
            roots[num_roots] = x
            num_roots += 1
            continue
        if hi - lo < 1e-13 * max(1.0, abs(lo)):
            # unresolvable cluster of roots; emit the midpoint once
            roots[num_roots] = 0.5 * (lo + hi)
            num_roots += 1
            continue
        if sp + 2 > lo_stack.shape[0]:
            break
        mid = 0.5 * (lo + hi)
        s_mid = _sturm_sign_changes(chain, degs, nchain, mid)
        lo_stack[sp] = lo
        hi_stack[sp] = mid
        slo_stack[sp] = s_lo
        shi_stack[sp] = s_mid
        sp += 1
        lo_stack[sp] = mid
        hi_stack[sp] = hi
        slo_stack[sp] = s_mid
        shi_stack[sp] = s_hi
        sp += 1
    return num_roots


@njit(cache=True, fastmath=True)
def _nullspace_free_last(G, v):
    # nullspace vector of a rank-9 10x10 matrix with the last coordinate as
    # the free variable (v[9] = 1), via elimination with partial pivoting
    scale = 0.0
    for r in range(10):
        for c in range(10):
            val = abs(G[r, c])
            if val > scale:
                scale = val
    if scale == 0.0:
        return False
    tol = 1e-12 * scale
    for col in range(9):
        piv = col
        max_val = abs(G[col, col])
        for r in range(col + 1, 10):
            val = abs(G[r, col])
            if val > max_val:
                max_val = val
                piv = r
        if max_val < tol:
            return False
        if piv != col:
            for c in range(col, 10):
                t = G[col, c]
                G[col, c] = G[piv, c]
                G[piv, c] = t
        inv = 1.0 / G[col, col]
        for r in range(col + 1, 10):
            factor = G[r, col] * inv
            if factor != 0.0:
                G[r, col] = 0.0
                for c in range(col + 1, 10):
                    G[r, c] -= factor * G[col, c]
    v[9] = 1.0
    for r in range(8, -1, -1):
        s = G[r, 9]
        for c in range(r + 1, 9):
            s += G[r, c] * v[c]
        v[r] = -s / G[r, r]
    return True


@njit(cache=True, fastmath=True)
def _pose_from_essential(e, data, sample, pp1x, pp1y, pp2x, pp2y, f1, f2,
                         pose, R):
    # closed-form decomposition of the flat essential matrix e (no SVD):
    # t spans the left nullspace of E, computed as the largest cross product
    # of two columns; the twisted-pair rotations follow from Horn's formula
    # R = cof(E) -/+ [t]_x E for E scaled to unit nonzero singular values
    # and unit t. Of the four candidates {R_a, R_b} x {t, -t} the one with
    # the best cheirality count on the minimal sample is written to `pose`.
    # The cheirality vote calibrates points as (x - pp) / f, so problems
    # whose `data` already holds calibrated points pass pp = 0, f = 1.
    # R is a (2, 3, 3) scratch buffer. Returns False for degenerate e.
    x1_x, x1_y, x2_x, x2_y = data

    # scale so the nonzero singular values are 1: |E|_F^2 = 2 sigma^2
    norm_sq = 0.0
    for j in range(9):
        norm_sq += e[j] * e[j]
    if norm_sq < 1e-24:
        return False
    s = math.sqrt(2.0 / norm_sq)
    e0 = s * e[0]
    e1 = s * e[1]
    e2 = s * e[2]
    e3 = s * e[3]
    e4 = s * e[4]
    e5 = s * e[5]
    e6 = s * e[6]
    e7 = s * e[7]
    e8 = s * e[8]

    # left null vector (E^T t = 0): cross products of the column pairs,
    # keep the best-conditioned one
    u0 = e3 * e7 - e6 * e4
    u1 = e6 * e1 - e0 * e7
    u2 = e0 * e4 - e3 * e1
    un = u0 * u0 + u1 * u1 + u2 * u2
    v0 = e3 * e8 - e6 * e5
    v1 = e6 * e2 - e0 * e8
    v2 = e0 * e5 - e3 * e2
    vn = v0 * v0 + v1 * v1 + v2 * v2
    w0 = e4 * e8 - e7 * e5
    w1 = e7 * e2 - e1 * e8
    w2 = e1 * e5 - e4 * e2
    wn = w0 * w0 + w1 * w1 + w2 * w2
    if vn > un:
        u0 = v0
        u1 = v1
        u2 = v2
        un = vn
    if wn > un:
        u0 = w0
        u1 = w1
        u2 = w2
        un = wn
    if un < 1e-24:
        return False
    inv = 1.0 / math.sqrt(un)
    tx = u0 * inv
    ty = u1 * inv
    tz = u2 * inv

    # cofactor matrix: row i is the cross product of the other two rows
    # (cyclic), quadratic in E so invariant to the sign of e
    c00 = e4 * e8 - e5 * e7
    c01 = e5 * e6 - e3 * e8
    c02 = e3 * e7 - e4 * e6
    c10 = e7 * e2 - e8 * e1
    c11 = e8 * e0 - e6 * e2
    c12 = e6 * e1 - e7 * e0
    c20 = e1 * e5 - e2 * e4
    c21 = e2 * e3 - e0 * e5
    c22 = e0 * e4 - e1 * e3

    # [t]_x E rows
    s00 = -tz * e3 + ty * e6
    s01 = -tz * e4 + ty * e7
    s02 = -tz * e5 + ty * e8
    s10 = tz * e0 - tx * e6
    s11 = tz * e1 - tx * e7
    s12 = tz * e2 - tx * e8
    s20 = -ty * e0 + tx * e3
    s21 = -ty * e1 + tx * e4
    s22 = -ty * e2 + tx * e5

    R[0, 0, 0] = c00 - s00
    R[0, 0, 1] = c01 - s01
    R[0, 0, 2] = c02 - s02
    R[0, 1, 0] = c10 - s10
    R[0, 1, 1] = c11 - s11
    R[0, 1, 2] = c12 - s12
    R[0, 2, 0] = c20 - s20
    R[0, 2, 1] = c21 - s21
    R[0, 2, 2] = c22 - s22
    R[1, 0, 0] = c00 + s00
    R[1, 0, 1] = c01 + s01
    R[1, 0, 2] = c02 + s02
    R[1, 1, 0] = c10 + s10
    R[1, 1, 1] = c11 + s11
    R[1, 1, 2] = c12 + s12
    R[1, 2, 0] = c20 + s20
    R[1, 2, 1] = c21 + s21
    R[1, 2, 2] = c22 + s22

    best_count = -1
    best_r = 0
    best_sign = 1.0
    for ri in range(2):
        for si in range(2):
            sign = 1.0 if si == 0 else -1.0
            t0 = sign * tx
            t1 = sign * ty
            t2 = sign * tz
            count = 0
            for k in range(sample.shape[0]):
                idx = sample[k]
                x = (x1_x[idx] - pp1x) / f1
                y = (x1_y[idx] - pp1y) / f1
                xp = (x2_x[idx] - pp2x) / f2
                yp = (x2_y[idx] - pp2y) / f2
                rx0 = R[ri, 0, 0] * x + R[ri, 0, 1] * y + R[ri, 0, 2]
                rx1 = R[ri, 1, 0] * x + R[ri, 1, 1] * y + R[ri, 1, 2]
                rx2 = R[ri, 2, 0] * x + R[ri, 2, 1] * y + R[ri, 2, 2]
                # depth in camera 1 from x2 x (z1 * R x1 + t) = 0 with
                # c = x2h x (R x1h), d = x2h x t
                c0 = yp * rx2 - rx1
                c1 = rx0 - xp * rx2
                c2 = xp * rx1 - yp * rx0
                d0 = yp * t2 - t1
                d1 = t0 - xp * t2
                d2 = xp * t1 - yp * t0
                cc = c0 * c0 + c1 * c1 + c2 * c2
                if cc <= 0.0:
                    continue
                z1 = -(c0 * d0 + c1 * d1 + c2 * d2) / cc
                z2 = z1 * rx2 + t2
                if z1 > 0.0 and z2 > 0.0:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = ri
                best_sign = sign

    for i in range(3):
        for j in range(3):
            pose[3 * i + j] = R[best_r, i, j]
    pose[9] = best_sign * tx
    pose[10] = best_sign * ty
    pose[11] = best_sign * tz
    return True


@njit(cache=True, fastmath=True)
def _solve_5pt(data, sample, models, workspace):
    # minimal 5-point solver; writes up to 10 pose models [R | t] into
    # `models` (cheirality-checked on the sample) and returns their count
    o = 0
    A = workspace[o:o + 45].reshape(5, 9)
    o += 45
    N = workspace[o:o + 36].reshape(4, 9)
    o += 36
    M = workspace[o:o + 200].reshape(10, 20)
    o += 200
    G = workspace[o:o + 100].reshape(10, 10)
    o += 100
    EEt = workspace[o:o + 90].reshape(3, 3, 10)
    o += 90
    tr = workspace[o:o + 10]
    o += 10
    ct = workspace[o:o + 30].reshape(3, 10)
    o += 30
    av = workspace[o:o + 4]
    o += 4
    bv = workspace[o:o + 4]
    o += 4
    coef = workspace[o:o + 11]
    o += 11
    row = workspace[o:o + 10]
    o += 10
    tmp_row = workspace[o:o + 10]
    o += 10
    chain = workspace[o:o + 121].reshape(11, 11)
    o += 121
    roots = workspace[o:o + 10]
    o += 10
    v = workspace[o:o + 10]
    o += 10
    lo_stack = workspace[o:o + 64]
    o += 64
    hi_stack = workspace[o:o + 64]
    o += 64
    e = workspace[o:o + 9]
    o += 9
    Rbuf = workspace[o:o + 18].reshape(2, 3, 3)
    o += 18
    iw = np.empty(139, dtype=np.int64)
    degs = iw[0:11]
    slo_stack = iw[11:75]
    shi_stack = iw[75:139]

    fill_epipolar_matrix(data, sample, A)

    if not _nullspace_5pt(A, N):
        return 0
    _build_constraints(N, M, EEt, tr, ct, av, bv)
    if not _gauss_jordan_10x20(M):
        return 0
    _action_matrix(M, G)
    if not _charpoly_danilevsky(G, coef, row, tmp_row):
        return 0
    num_roots = _real_roots_sturm(coef, 10, chain, degs, roots,
                                  lo_stack, hi_stack, slo_stack, shi_stack)

    count = 0
    for r in range(num_roots):
        lam = roots[r]
        _action_matrix(M, G)
        for d in range(10):
            G[d, d] -= lam
        if not _nullspace_free_last(G, v):
            continue
        # eigenvector holds the basis monomials; v[9] = 1 by construction
        y_sol = v[7]
        z_sol = v[8]
        for j in range(9):
            e[j] = lam * N[0, j] + y_sol * N[1, j] + z_sol * N[2, j] + N[3, j]
        if _pose_from_essential(e, data, sample, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                                models[count], Rbuf):
            count += 1
    return count


# ---------------------------------------------------------------------------
# pluggable component class
# ---------------------------------------------------------------------------

class FivePointSolver():
    # minimal solver for calibrated relative pose from 5 correspondences;
    # outputs pose models [R | t] (cheirality-checked on the sample)
    sample_size = 5
    num_params = 12
    max_models = 10
    workspace_size = 946
    solve = staticmethod(_solve_5pt)
