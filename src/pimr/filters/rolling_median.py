"""
Rolling median and median-absolute-deviation (MAD) estimator.
"""

import numpy as np


def rolling_median_mad(prices, window=20):
    """
    Estimate the typical price level of a price series using only the
    rolling median and the rolling median absolute deviation (MAD). For each point the trailing ``window`` samples (expanding at the start of the series) are summarised by their median and by the MAD around the median.

    Parameters
    ----------
    prices : array_like
        Price series to summarise.
    window : int
        Number of trailing samples in each rolling window. It will always be the next odd number (so index is at center!).

    Returns
    -------
    tuple of numpy.ndarray
        ``(median, mad)``, each the same length as ``prices``.
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    med = np.empty(n, dtype=np.float64)
    mad = np.empty(n, dtype=np.float64)
    for i in range(n):
        chunk = prices[max(0, i - window + 1): i + 1]  # chunk behind the index i
        med[i] = np.median(chunk)
        mad[i] = np.median(np.abs(chunk - med[i]))
    return med, mad
