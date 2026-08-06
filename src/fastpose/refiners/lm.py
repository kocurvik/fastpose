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

import numpy as np
from numba import njit


def build_lm_refine(init_state, state_to_model, accumulate, apply_step, cost,
                    state_size, num_tangent, model_size):
    # compiles an LM refiner specialized for one problem; the kernels are
    # closure constants so numba can bind the calls statically (the same
    # pattern as build_ransac in the RANSAC engine)
    @njit(fastmath=True)
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
            delta = np.linalg.solve(A, -Jtr)

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
