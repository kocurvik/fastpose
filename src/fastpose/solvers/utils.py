"""Shared numba helpers for the minimal solvers."""

from fastpose.jit_backend import cpu_jit


def build_fill_epipolar_matrix(jit):
    # built through a factory so the CUDA backend can compile the same source
    # as a device function; see fastpose/jit_backend.py
    @jit(inline=True)
    def fill_epipolar_matrix(data, sample, A):
        # rows of the epipolar constraint matrix: x2^T F x1 = 0 linearized in
        # the 9 entries of F, one row per sampled correspondence; A is (k, 9)
        x1_x, x1_y, x2_x, x2_y = data
        for k in range(A.shape[0]):
            i = sample[k]
            x = x1_x[i]
            y = x1_y[i]
            xp = x2_x[i]
            yp = x2_y[i]
            A[k, 0] = xp * x
            A[k, 1] = xp * y
            A[k, 2] = xp
            A[k, 3] = yp * x
            A[k, 4] = yp * y
            A[k, 5] = yp
            A[k, 6] = x
            A[k, 7] = y
            A[k, 8] = 1.0

    return fill_epipolar_matrix


fill_epipolar_matrix = build_fill_epipolar_matrix(cpu_jit)
