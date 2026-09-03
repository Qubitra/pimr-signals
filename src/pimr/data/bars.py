"""Time-bucket aggregation of event streams into OHLCV candles and mid prices.

Extracted from tests/gold_data.py. These functions are loader-agnostic: they
work on any chronological (timestamp, price, size) arrays, whether they came
from a GLBX CSV, a DBN file or a synthetic generator.
"""

import numpy as np


def build_ohlc(ts_ns, price, size, bucket_ns=300_000_000_000):
    """
    Aggregate a chronological stream of trades into OHLCV time buckets.

    Each trade is assigned to the bucket that contains its timestamp (buckets are aligned to multiples of bucket_ns since the epoch, so with a 5-minute
    bucket the boundaries fall on 00, 05, 10, ... of each hour). For every bucket that contains at least one trade the function computes the four
    candle statistics — open (first trade price), high (maximum), low (minimum) and close (last trade price) — plus the traded volume (sum of
    sizes). Buckets without any trades are omitted from the output. The input arrays must already be sorted by timestamp.
    Args:
        ts_ns (numpy.ndarray): Trade timestamps in nanoseconds since the epoch, sorted ascending (int64).
        price (numpy.ndarray): Trade prices in dollars (float64), same length as ts_ns.
        size (numpy.ndarray): Trade sizes in contracts (float64), same length as ts_ns.
        bucket_ns (int): Bucket width in nanoseconds (e.g. 5 minutes = 300_000_000_000).
    Returns:
        dict: Six numpy arrays of equal length, one entry per non-empty bucket, ordered by time:
            - bucket_start (int64): bucket start time in nanoseconds.
            - open, high, low, close (float64): the OHLC prices in dollars.
            - volume (float64): total contracts traded in the bucket.
    Raises:
        ValueError: If ts_ns is empty (no trades to aggregate).
    """
    if ts_ns.size == 0:
        raise ValueError("ts_ns is empty: no trades to aggregate into OHLC buckets")

    bucket_idx = ts_ns // bucket_ns
    uniq, start_pos = np.unique(bucket_idx, return_index=True)
    end_pos = np.append(start_pos[1:], ts_ns.size)

    o = price[start_pos].astype(np.float64)
    h = np.maximum.reduceat(price, start_pos).astype(np.float64)
    l = np.minimum.reduceat(price, start_pos).astype(np.float64)
    c = price[end_pos - 1].astype(np.float64)
    v = np.add.reduceat(size, start_pos).astype(np.float64)

    return {
        "bucket_start": uniq * bucket_ns,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
    }


def build_mid(ts_ns, mid, bucket_ns=300_000_000_000):
    """
    Aggregate a chronological stream of book events into per-bucket mid prices.
    For every bucket that contains at least one event the function returns the average top-of-book mid price over the events in that bucket.
    NaN mids (empty book side) are ignored. Buckets without any events are omitted from the output.
    Args:
        ts_ns (numpy.ndarray): Event timestamps in nanoseconds since the epoch, sorted ascending (int64).
        mid (numpy.ndarray): Top-of-book mid prices in dollars (float64), same length as ts_ns.
        bucket_ns (int): Bucket width in nanoseconds (e.g. 5 minutes = 300_000_000_000).
    Returns:
        dict: Two numpy arrays of equal length, one entry per non-empty bucket, ordered by time:
            - bucket_start (int64): bucket start time in nanoseconds.
            - mid (float64): average mid price in the bucket, NaN if the bucket contains no valid mid.
    Raises:
        ValueError: If ts_ns is empty (no events to aggregate).
    """
    if ts_ns.size == 0:
        raise ValueError("ts_ns is empty: no events to aggregate into mid-price buckets")

    bucket_idx = ts_ns // bucket_ns
    uniq, start_pos = np.unique(bucket_idx, return_index=True)

    valid = ~np.isnan(mid)
    sums = np.add.reduceat(np.where(valid, mid, 0.0), start_pos)
    counts = np.add.reduceat(valid.astype(np.float64), start_pos)
    m = np.divide(sums, counts, out=np.full(uniq.size, np.nan), where=counts > 0)

    return {
        "bucket_start": uniq * bucket_ns,
        "mid": m,
    }
