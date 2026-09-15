"""Relative Strength Index (RSI).

For 1-minute bars a 2-minute RSI period corresponds to period = 2 bars.
The simple-average (Cutler) recipe is used:

    1. delta[k]    = close[k+1] - close[k]
    2. gain / loss = positive / negative part of delta
    3. avg_gain, avg_loss = moving average of gain / loss over `period`
       (simple average by default; pass average=EMA to RS() / rsi() to
       smooth exponentially instead)
    4. RS  = avg_gain / avg_loss
    5. RSI = 100 - 100 / (1 + RS)

Extracted from tests/relative_strength_index.py; the moving averages live in
pimr.indicators.moving_averages.
"""

import numpy as np

from pimr.indicators.moving_averages import SMA


def gains_and_losses(prices):
    """
    Split the one-bar price changes into their positive and negative parts.

    Args:
        prices (numpy.ndarray): Close prices in dollars (float64), one per bar,
            in chronological order.

    Returns:
        tuple: Two numpy arrays of length len(prices) - 1:
            - gains  (float64): price increase per bar, 0.0 on down/flat bars.
            - losses (float64): price drop per bar as a POSITIVE number,
              0.0 on up/flat bars.
    """
    delta = np.diff(np.asarray(prices, dtype=np.float64))
    gains = np.where(delta > 0.0, delta, 0.0)
    losses = np.where(delta < 0.0, -delta, 0.0)
    return gains, losses


def RS(data, period=2, average=SMA):
    """
    Relative Strength: average gain / average loss over the last `period` bars.
    Args:
        data (numpy.ndarray): Close prices in dollars (float64), one per bar,
            in chronological order.
        period (int): RSI period in bars. The gold data is 1-minute bars, so
            period = 2 gives the 2-minute RSI (default: 2).
        average (callable): Smoothing function (samples, period) -> array with
            the same NaN warm-up convention: SMA (default) or EMA.

    Returns:
        numpy.ndarray: RS aligned with `data` (same length).
        RS[i] uses prices up to and including data[i]; the first `period` entries are NaN. Where the market only went up (avg_loss == 0) RS is +inf, which RSI() maps to 100. Where nothing moved at all RS is 1.0, which RSI() maps to aneutral 50.
    """
    gains, losses = gains_and_losses(data)
    avg_gain = average(gains, period)
    avg_loss = average(losses, period)

    rs = np.full(avg_gain.size, np.nan)
    np.divide(avg_gain, avg_loss, out=rs, where=avg_loss > 0.0)
    rs[(avg_loss == 0.0) & (avg_gain > 0.0)] = np.inf
    rs[(avg_loss == 0.0) & (avg_gain == 0.0)] = 1.0

    # np.diff consumed one sample: pad so RS stays aligned with the input prices
    return np.concatenate(([np.nan], rs))


def RSI(rs):
    """
    Map Relative Strength to the 0-100 RSI scale: RSI = 100 - 100 / (1 + RS).

    Args:
        rs (numpy.ndarray): Relative Strength values as returned by RS().

    Returns:
        numpy.ndarray: RSI in [0, 100], same shape as rs. NaN stays NaN and
        RS = +inf maps to 100.
    """
    rs = np.asarray(rs, dtype=np.float64)
    return 100.0 - 100.0 / (1.0 + rs)


def rsi(prices, period=2, average=SMA):
    """
    Convenience wrapper: RSI of a price series in one call.

    Args:
        prices (numpy.ndarray): Close prices in dollars, one per bar.
        period (int): RSI period in bars (default: 2 = 2-minute RSI on
            1-minute bars).
        average (callable): Smoothing function for the average gain / loss:
            SMA (default) or EMA.

    Returns:
        numpy.ndarray: RSI aligned with `prices` (same length, first `period`
        entries NaN).
    """
    return RSI(RS(prices, period, average))
