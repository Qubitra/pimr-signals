"""
Hodrick-Prescott (HP) filter.
Basic mathematical representation obtained from:
https://xbe.at/index.php?filename=Implementing+Hodrick-Prescott+Filter+for+Technical+Analysis+in+Python.md
"""

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


def hp_filter(y, lambda_param=1600):
    """
    Apply the Hodrick-Prescott (HP) filter to the time series y.
    Parameters
    ----------
    y : array-like
        Observed time series.
    lambda_param : float, optional
        Smoothing parameter controlling the trade-off between fit and smoothness of the estimated trend. Default is 1600.
    Returns
    -------
    trend : ndarray
        Estimated smooth trend component.
    cycle : ndarray
        Cyclical component, defined as y - trend.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    e = np.ones(n)
    B = diags([e, -2 * e, e], offsets=[0, 1, 2], shape=(n - 2, n), format="csc")

    trend = spsolve(eye(n, format="csc") + lambda_param * (B.T @ B), y)
    cycle = y - trend

    return trend, cycle
