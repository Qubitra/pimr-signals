"""Moving averages: simple (SMA) and exponential (EMA, with a fast IIR variant).

All three share the same NaN warm-up convention (the first period - 1 entries
are NaN), so they are interchangeable as the smoothing function of
pimr.indicators.rsi.RS.

Extracted from tests/relative_strength_index.py.
"""

import numpy as np
from scipy import signal


def SMA(x, period):
    """
    Simple moving average over a window of `period` samples.

    Args:
        x (numpy.ndarray): Samples to average (float64), e.g. gains or losses.
        period (int): Window length in samples (2 for a 2-minute RSI on
            1-minute bars).

    Returns:
        numpy.ndarray: Same length as x. The first period - 1 entries are NaN
        (not enough history); entry i is the mean of x[i - period + 1 : i + 1].

    Raises:
        ValueError: If period < 1 or x has fewer than `period` samples.
    """
    x = np.asarray(x, dtype=np.float64)
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if x.size < period:
        raise ValueError(f"need at least {period} samples, got {x.size}")

    out = np.full(x.size, np.nan)
    out[period - 1:] = np.convolve(x, np.ones(period) / period, mode="valid")
    return out


def EMA(x, period, alpha=None):
    """
    Exponential moving average: y[t] = alpha * x[t] + (1 - alpha) * y[t-1].
    Plain python-loop version, seeded with the simple mean of the first `period` samples. The NaN warm-up matches SMA(), so the two are interchangeable inside RS().

    Args:
        x (numpy.ndarray): Samples to average (float64).
        period (int): Window length in samples; sets the default weight alpha = 2 / (period + 1) and the length of the seed average.
        alpha (float or None): Smoothing weight in (0, 1]. Default None means 2 / (period + 1), the standard EMA weight.

    Returns:
        numpy.ndarray: Same length as x. The first period - 1 entries are NaN
        (not enough history), entry period - 1 is the seed (simple mean of x[:period]) and later entries follow the EMA recursion.

    Raises:
        ValueError: If period < 1, x has fewer than `period` samples, or
            alpha is outside (0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if x.size < period:
        raise ValueError(f"need at least {period} samples, got {x.size}")
    if alpha is None:
        alpha = 2.0 / (period + 1.0)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    out = np.full(x.size, np.nan)
    y = x[:period].mean()
    out[period - 1] = y
    for k in range(period, x.size):
        y = alpha * x[k] + (1.0 - alpha) * y
        out[k] = y
    return out


def EMA_fast(x, period, alpha=None):
    """
    Exponential moving average, fast version — same output as EMA().

    The recursion y[t] = alpha * x[t] + (1 - alpha) * y[t-1] is seeded with the simple mean of the first `period` samples and evaluated with
    scipy.signal.lfilter (IIR filter b = [alpha], a = [1, -(1 - alpha)]) in compiled C, so no python loop is needed. Roughly 30x faster than EMA();
    worth it on long series (above ~100k samples, e.g. raw tick data). The NaN warm-up matches SMA(), so the two are interchangeable inside RS().

    Args:
        x (numpy.ndarray): Samples to average (float64), e.g. gains or losses.
        period (int): Window length in samples; sets the default weight alpha = 2 / (period + 1) and the length of the seed average.
        alpha (float or None): Smoothing weight in (0, 1]. Larger alpha reacts
            faster to new samples. Default None means 2 / (period + 1), the standard EMA weight; alpha = 1 / period gives Wilder's smoothing the original 1978 RSI).

    Returns:
        numpy.ndarray: Same length as x. The first period - 1 entries are NaN
        (not enough history), entry period - 1 is the seed (simple mean of x[:period]) and later entries follow the EMA recursion.

    Raises:
        ValueError: If period < 1, x has fewer than `period` samples, or
            alpha is outside (0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if x.size < period:
        raise ValueError(f"need at least {period} samples, got {x.size}")
    if alpha is None:
        alpha = 2.0 / (period + 1.0)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    seed = x[:period].mean()

    out = np.full(x.size, np.nan)
    out[period - 1] = seed
    if x.size > period:
        b, a = [alpha], [1.0, -(1.0 - alpha)]
        zi = np.array([(1.0 - alpha) * seed])  # lfilter state so that y[0] = alpha*x[period] + (1-alpha)*seed
        out[period:], _ = signal.lfilter(b, a, x[period:], zi=zi)
    return out
