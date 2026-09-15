"""L1 trend filtering via scipy.optimize.minimize (L-BFGS-B).

Same problem as the CVXPY version in pimr.filters.l1_trend:

    minimize_tau  (1/2)||y - tau||_2^2 + lambda * ||D tau||_1

scipy.optimize.minimize can't be pointed at this objective directly in a
reliable way, because ||D tau||_1 is non-differentiable exactly where the
solution lives (at second differences == 0). Quasi-Newton methods like
BFGS/L-BFGS assume a smooth gradient and will stall or chatter at the kinks.

The trick: reparametrize so the problem becomes SMOOTH + BOX CONSTRAINTS,
which L-BFGS-B handles natively.

 1. Parametrize the trend by its own "internal coordinates":
        t0 = tau_1              (initial level)
        s0 = tau_2 - tau_1      (initial slope)
        d_t = second differences, t = 3..n   (the things we penalize)
    Then tau is reconstructed by two cumulative sums.

 2. Split each second difference into positive and negative parts:
        d = p - q,   p >= 0, q >= 0
    so that  |d| = p + q  at the optimum, and the penalty
        lambda * ||d||_1  becomes the LINEAR term  lambda * sum(p + q).

The objective in (t0, s0, p, q) is quadratic + linear -> perfectly smooth.
The non-smoothness has been moved into the bound constraints p, q >= 0,
and L-BFGS-B's active-set handling pins variables EXACTLY at 0 -- which is
what recovers the exact piecewise-linear (sparse-kink) solution.

Extracted from tests/l1tf_scipy.py.txt.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags


def _second_difference_matrix(n):
    """Sparse (n-2) x n second-order difference matrix with rows [1, -2, 1]."""
    e = np.ones(n)
    return diags([e, -2 * e, e], offsets=[0, 1, 2], shape=(n - 2, n), format='csc')


def _revcumsum(x):
    """Reverse cumulative sum: out[i] = sum(x[i:])."""
    return np.cumsum(x[::-1])[::-1]


def _reconstruct_trend(t0, s0, d):
    """tau from initial level, initial slope, and second differences."""
    # first differences f_t = tau_t - tau_{t-1}, t = 2..n
    f = s0 + np.concatenate(([0.0], np.cumsum(d)))
    # levels
    tau = t0 + np.concatenate(([0.0], np.cumsum(f)))
    return tau


def l1_trend_filter_scipy(y, lambda_param=50.0, max_iter=200000, tol=0.0):
    """
    l1 trend filtering solved with scipy.optimize.minimize (L-BFGS-B).

    Same interface as the ADMM version: returns (trend, cycle).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 3:
        return y.copy(), np.zeros(n)
    m = n - 2  # number of second differences

    def unpack(x):
        t0, s0 = x[0], x[1]
        p, q = x[2:2 + m], x[2 + m:]
        return t0, s0, p, q

    def objective_and_grad(x):
        t0, s0, p, q = unpack(x)
        d = p - q
        tau = _reconstruct_trend(t0, s0, d)
        r = tau - y

        obj = 0.5 * r @ r + lambda_param * (p.sum() + q.sum())

        # Chain rule through the two cumulative sums (adjoint = revcumsum)
        g_f = _revcumsum(r)[1:]      # dObj/df_t,  t = 2..n   (length n-1)
        g_d = _revcumsum(g_f)[1:]    # dObj/dd_t,  t = 3..n   (length n-2)

        grad = np.empty_like(x)
        grad[0] = r.sum()            # d/dt0
        grad[1] = g_f.sum()          # d/ds0
        grad[2:2 + m] = g_d + lambda_param    # d/dp
        grad[2 + m:] = -g_d + lambda_param    # d/dq
        return obj, grad

    # Warm start from the data: exact-fit second differences of y
    d0 = np.diff(y, 2)
    x0 = np.concatenate(([y[0], y[1] - y[0]],
                         np.maximum(d0, 0), np.maximum(-d0, 0)))

    bounds = [(None, None), (None, None)] + [(0, None)] * (2 * m)

    # NOTE: the double-cumsum parametrization is ill-conditioned (condition
    # number grows like n^4), so L-BFGS-B needs tight tolerances and many
    # iterations to reach the exact optimum. With scipy's default ftol it
    # stops early at a visibly sub-optimal (too many kinks) solution.
    res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                   bounds=bounds,
                   options={'maxiter': max_iter, 'maxfun': max_iter,
                            'ftol': tol, 'gtol': 1e-12})

    t0, s0, p, q = unpack(res.x)
    trend = _reconstruct_trend(t0, s0, p - q)
    cycle = y - trend
    return trend, cycle
