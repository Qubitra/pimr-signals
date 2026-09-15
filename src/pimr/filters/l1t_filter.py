"""L1 trend filtering via convex optimization (CVXPY).

Extracted from tests/l1t_filter.py.
"""

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def l1tf_cvxpy(y, lambda_param=50.0, solver=cp.ECOS):
    """
    Apply L1 trend filtering to a time series using convex optimization. Minimizing the convex optimization problem:
        MIN: 0.5 * ||y - x||^2 + lambda * ||D^2 x||_1
    Parameters
    ----------
    y : array-like
        Observed time series.
    lambda_param : float, optional
        Regularization parameter controlling smoothness. Larger values produce smoother trends. Default is 50.0.
    solver : cvxpy.Problem.solver, optional
        Convex solver to use. Available options include ECOS, SCS, OSQP, CVXOPT.
        Default is ECOS. See cvxpy.installed_solvers() for available solvers.
    Returns
    -------
    trend : ndarray
        Estimated piecewise linear trend component.
    Reference
    ----------
    https://www.cvxpy.org/examples/applications/l1_trend_filter.html
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size

    e = np.ones((1, n))
    D = sp.diags_array([e, -2 * e, e], offsets=range(3), shape=(n - 2, n))

    # Solve l1 trend filtering problem.
    x = cp.Variable(shape=n)
    obj = cp.Minimize(0.5 * cp.sum_squares(y - x) + lambda_param * cp.norm(D @ x, 1))
    prob = cp.Problem(obj)

    # If ECOS and SCS solvers fail to converge before the iteration limit. Use CVXOPT instead.
    prob.solve(solver=solver, verbose=False)

    return x.value


def lambda_max(y):
    """
    Compute the smallest lambda value for which the l1 trend filter solution is affine. This helper uses the same second-difference matrix D  and solves (D D^T) z = D y, returning the infinity norm of the resulting vector.
    Parameters
    ----------
    y : array-like
        Observed/input time series.
    Returns
    -------
    float
        The lambda value that the l1 trend filter output is an affine (straight-line) fit to y.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 3:
        return 0.0

    e = np.ones(n)
    D = sp.diags_array([e, -2 * e, e], offsets=range(3), shape=(n - 2, n), format="csc")
    z = sp.linalg.spsolve(D @ D.T, D @ y)
    return float(np.max(np.abs(z)))


def l1_lengths(trend, tol=None, return_knots=False):
    """
    Compute the length of each linear segment of a piecewise linear trend (e.g. the output of l1t_filter_cvxpy). The l1 penalty on ||D^2 x||_1 makes the second difference of the trend exactly zero inside a segment and non-zero only at the kink (knot) points, so segments are recovered by locating second differences above a small tolerance (the solver returns an approximate solution, so exact zeros come back as numerical noise).
    Parameters
    ----------
    trend : array-like
        Piecewise linear trend, as returned by l1t_filter_cvxpy.
    tol : float, optional
        Threshold above which a second difference counts as a kink. Default is 1e-4 times the largest absolute second difference (separating real slope changes from solver noise), with a floor of 1e-8 times the trend scale so a fully affine trend yields a single segment. Pass a larger value to also ignore small real slope changes.
    return_knots : bool, optional
        If True, also return the sample indices of the kink points.
    Returns
    -------
    lengths : ndarray
        Length of each segment in samples/bars (number of intervals spanned). The lengths sum to len(trend) - 1. Multiply by the candle duration to convert to time.
    knots : ndarray, optional
        Indices of the kink points, only if return_knots is True.
    """
    trend = np.asarray(trend, dtype=float).ravel()
    n = trend.size
    if n < 3:
        lengths, knots = np.array([max(n - 1, 0)]), np.array([], dtype=int)
        return (lengths, knots) if return_knots else lengths

    d2 = np.diff(trend, n=2)                       # d2[i] = slope change at sample i+1
    if tol is None:
        scale = max(np.max(np.abs(trend)), 1.0)
        tol = max(1e-4 * np.max(np.abs(d2)), 1e-8 * scale)

    knots = np.flatnonzero(np.abs(d2) > tol) + 1   # sample indices where the slope changes
    bounds = np.concatenate(([0], knots, [n - 1]))
    lengths = np.diff(bounds)                      # segment k spans bounds[k]..bounds[k+1]

    return (lengths, knots) if return_knots else lengths
