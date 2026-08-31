"""Generic Levenberg-Marquardt engine shared by all refiners.

The LM loop itself (damping schedule, damped normal equations, accept/reject,
convergence checks) is problem-agnostic. A refinement problem is defined by a
flat float64 state vector plus four numba kernels handed to `build_lm_refine`:

    init_state(model, state) -> bool
        decompose the flat model vector into the state vector
    state_to_model(state, f)
        reassemble the flat model vector from the state
    accumulate(data, f, state, JtJ, Jtr, max_error_sq) -> num_residuals
        residuals + jacobian: accumulate the normal equations
        (JtJ symmetric, Jtr = J^T r) at the current state
    apply_step(state, delta, state_new)
        retraction: apply the tangent step `delta` to the state

plus a cost kernel with the scorer signature
`cost(f, data, max_error_sq, best_score) -> (cost, num_inliers)`; the LM loop
minimizes exactly this cost, so it matches the score used for model selection.

The returned kernel has the RANSAC refiner signature
`refine(data, model, refined, max_error_sq, num_iterations) -> bool`.
"""

import math

import numpy as np
from numba import njit

from fastpose.kernel_cache import stabilize


@njit(cache=True)
def _solve_damped(A, Jtr, delta, n):
    # Cholesky solve of A delta = -Jtr, in place on A (scratch). Returns False
    # for a non-positive or non-finite pivot - a singular or NaN/inf system -
    # where np.linalg.solve would raise LinAlgError. Raising is not an option
    # here: refine kernels run inside the parallel RANSAC driver, and numba
    # cannot propagate an exception out of a parallel=True function (it
    # surfaces as SystemError and kills the whole estimate). Deliberately not
    # fastmath so the pivot check is honest about NaN.
    for j in range(n):
        d = A[j, j]
        for k in range(j):
            d -= A[j, k] * A[j, k]
        if not (d > 0.0) or not math.isfinite(d):
            return False
        d = math.sqrt(d)
        A[j, j] = d
        inv = 1.0 / d
        for i in range(j + 1, n):
            s = A[i, j]
            for k in range(j):
                s -= A[i, k] * A[j, k]
            A[i, j] = s * inv
    # forward substitution L y = -Jtr, then back substitution L^T delta = y
    for i in range(n):
        s = -Jtr[i]
        for k in range(i):
            s -= A[i, k] * delta[k]
        delta[i] = s / A[i, i]
    for i in range(n - 1, -1, -1):
        s = delta[i]
        for k in range(i + 1, n):
            s -= A[k, i] * delta[k]
        delta[i] = s / A[i, i]
    # a finite factorization can still produce a non-finite step when Jtr
    # itself carries inf; the fastmath caller cannot test that reliably
    for i in range(n):
        if not math.isfinite(delta[i]):
            return False
    return True


def build_lm_refine(init_state, state_to_model, accumulate, apply_step, cost,
                    state_size, num_tangent, model_size):
    # compiles an LM refiner specialized for one problem; the kernels are
    # closure constants so numba can bind the calls statically (the same
    # pattern as build_ransac in the RANSAC engine, including the cache=True
    # plus stabilize combination - one specialization per problem and per
    # loss, each with its own process-independent cache key)
    stabilize(init_state, state_to_model, accumulate, apply_step, cost)

    @njit(cache=True, fastmath=True)
    def refine(data, model, refined, max_error_sq, num_iterations):
        state = np.empty(state_size)
        state_new = np.empty(state_size)
        if not init_state(model, state):
            return False

        f = np.empty(model_size)
        f_new = np.empty(model_size)
        state_to_model(state, f)
        best_cost, _ = cost(f, data, max_error_sq, 1e300)

        JtJ = np.empty((num_tangent, num_tangent))
        Jtr = np.empty(num_tangent)
        A = np.empty((num_tangent, num_tangent))
        delta = np.empty(num_tangent)

        lam = 1e-4
        recompute_jacobian = True
        for _ in range(num_iterations):
            if recompute_jacobian:
                num_residuals = accumulate(data, f, state, JtJ, Jtr, max_error_sq)
                if num_residuals < num_tangent:
                    return False

            # damped normal equations (JtJ + lambda * I) delta = -Jtr
            for p in range(num_tangent):
                for q in range(num_tangent):
                    A[p, q] = JtJ[p, q]
                A[p, p] += lam
            if not _solve_damped(A, Jtr, delta, num_tangent):
                # singular or non-finite normal equations: treat it as a
                # rejected step so a garbage model runs the damping schedule
                # dry instead of aborting the whole RANSAC call
                lam = lam * 10.0
                recompute_jacobian = False
                if lam > 1e10:
                    break
                continue

            step_sq = 0.0
            for p in range(num_tangent):
                step_sq += delta[p] * delta[p]

            apply_step(state, delta, state_new)
            state_to_model(state_new, f_new)
            cost_new, _ = cost(f_new, data, max_error_sq, 1e300)

            if cost_new < best_cost:
                best_cost = cost_new
                for k in range(state_size):
                    state[k] = state_new[k]
                for j in range(model_size):
                    f[j] = f_new[j]
                lam = max(lam * 0.1, 1e-10)
                recompute_jacobian = True
                if step_sq < 1e-20:
                    break
            else:
                lam = lam * 10.0
                recompute_jacobian = False
                if lam > 1e10:
                    break

        for j in range(model_size):
            refined[j] = f[j]
        return True

    return refine
