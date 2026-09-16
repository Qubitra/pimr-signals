"""
Both L1-trend linear-segment-length histograms in one figure over a whole
folder of Databento DBN files (.dbn.zst): a lambda sweep at a fixed candle
width (top panel) and a candle-width sweep at fixed lambda (bottom panel).

Combines scripts/plot_segment_hist.py (lambda sweep) and
scripts/plot_segment_hist_candles.py (candle-width sweep). Every daily MBP-1
file is loaded and reduced to its trades right away, the days are
concatenated in time and the most-traded instrument (the front month) is
selected once; each panel then builds its own candles and fits its own L1
trends. Not optimized -- the fits are simply run one after the other.

Usage:
    python scripts/plot_segment_hist_combined.py                 # every .dbn.zst in --data-dir
    python scripts/plot_segment_hist_combined.py file1 file2 ... # specific files
"""

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # until pimr is pip-installed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from pimr.data import build_ohlc
from pimr.data.dbn import load_dbn_many, most_traded_instrument
from pimr.filters import l1t_filter_cvxpy, lambda_max, linear_segment_lengths

DEFAULT_DATA_DIR = "/Users/arash/Qubitra/GC_202608"
DAY_NS = 86_400_000_000_000
LAMBDA_MULTS = [0.33, 0.5, 1.0, 2.0, 3.0]   # top panel: multiples of the baseline lambda
DEFAULT_CANDLE_MINS = [5, 10, 15, 30]       # bottom panel: candle widths in minutes
LAMBDA_FRAC = 1.e-3   # baseline lambda = LAMBDA_FRAC * lambda_max(close) of each series


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=None, help="mbp-1 .dbn.zst files")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="folder scanned for *.dbn.zst when no paths are given")
    parser.add_argument("--candle-min", type=float, default=1,
                        help="candle width in minutes for the lambda-sweep panel")
    parser.add_argument("--candle-mins", type=float, nargs="+", default=DEFAULT_CANDLE_MINS,
                        help="candle widths in minutes for the fixed-lambda panel")
    args = parser.parse_args(argv)

    candle_min = args.candle_min
    candle_mins = args.candle_mins
    paths = args.paths or sorted(glob.glob(os.path.join(args.data_dir, "*.dbn.zst")))

    print(f"L1 trend segments from {len(paths)} DBN file(s): lambda sweep at "
          f"{candle_min:g}-min candles, candle sweep at "
          f"{', '.join(f'{c:g}' for c in candle_mins)} min:")
    trades = load_dbn_many(paths, trades_only=True)
    print(f"  {trades['ts'].size:,} trades")

    front_id, front_symbol, n_front = most_traded_instrument(
        trades["instrument_id"], trades["symbols"])
    print(f"\nmost-traded instrument: {front_symbol} "
          f"({n_front:,} of {trades['ts'].size:,} trades)")

    is_front = trades["instrument_id"] == front_id
    ts = trades["ts"][is_front]
    price = trades["price"][is_front]
    size = trades["size"][is_front]
    n_days = np.unique(ts // DAY_NS).size

    # --- top panel data: lambda sweep at a fixed candle width --------------
    ohlc = build_ohlc(ts, price, size, bucket_ns=int(candle_min * 60 * 10**9))
    print(f"\n{ohlc['close'].size:,} candles of {candle_min:g} min "
          f"over {n_days} trading days (UTC)")

    lam0 = LAMBDA_FRAC * lambda_max(ohlc["close"])
    lam_sweep = []
    for mult in LAMBDA_MULTS:
        l1tf = l1t_filter_cvxpy(ohlc["close"], lambda_param=mult * lam0)
        seg_lengths = linear_segment_lengths(l1tf)
        print(f"{mult:g} x lambda ({mult * lam0:.3g}): {seg_lengths.size} linear segments, "
              f"mean {seg_lengths.mean():.1f} bars = {seg_lengths.mean() * candle_min:.0f} minutes, "
              f"min/max: {seg_lengths.min()}/{seg_lengths.max()} bars")
        lam_sweep.append((mult, seg_lengths))

    # --- bottom panel data: candle-width sweep at fixed lambda -------------
    # lambda re-anchored to each series' own lambda_max
    print()
    candle_sweep = []
    for c_min in candle_mins:
        ohlc_c = build_ohlc(ts, price, size, bucket_ns=int(c_min * 60 * 10**9))
        lam = LAMBDA_FRAC * lambda_max(ohlc_c["close"])
        l1tf = l1t_filter_cvxpy(ohlc_c["close"], lambda_param=lam)
        seg_lengths = linear_segment_lengths(l1tf)
        print(f"{c_min:g}-min candles: {ohlc_c['close'].size:,} candles, "
              f"lambda {lam:.3g}, {seg_lengths.size} linear segments, "
              f"mean {seg_lengths.mean():.1f} bars = {seg_lengths.mean() * c_min:.0f} minutes, "
              f"min/max: {seg_lengths.min()}/{seg_lengths.max()} bars")
        candle_sweep.append((c_min, lam, seg_lengths))

    fig, (ax_lam, ax_candle) = plt.subplots(2, 1, figsize=(10, 11))

    # --- top panel: one histogram per lambda, overlaid as translucent bars
    # on shared bins; light -> dark encodes small -> large lambda
    # (single-hue ramp, CVD-safe)
    lam_colors = plt.cm.YlOrBr(np.linspace(0.45, 1.0, len(lam_sweep)))

    # fine bins so short-segment structure stays visible; half-integer edges so
    # the (integer) bar counts never straddle a bin boundary
    max_len = max(lengths.max() for _, lengths in lam_sweep)
    bin_step = max(1, int(np.ceil(max_len / 60)))
    bins = np.arange(0.5, max_len + bin_step + 0.5, bin_step)

    # overlaid histogram per lambda: translucent fill with a solid outline in the
    # same hue; the x axis is in minutes so every bar reads directly as a duration
    bar_bins = np.zeros(bins.size - 1, dtype=bool)   # bins holding a bar in any series
    for (mult, lengths), color in zip(lam_sweep, lam_colors):
        counts, _, _ = ax_lam.hist(lengths * candle_min, bins=bins * candle_min,
                                   histtype="stepfilled",
                                   facecolor=to_rgba(color, 0.35), edgecolor=color, linewidth=1.2,
                                   label=f"{mult:g}$\\,\\lambda$: {lengths.size} segments, "
                                         f"mean {lengths.mean():.0f} bars ({lengths.mean() * candle_min:.0f} min)")
        bar_bins |= counts > 0

    # one tick directly below every bar column, labelled with its minutes value;
    # vertical labels so adjacent columns in the dense head don't collide
    centers_min = (bins[:-1] + bins[1:]) / 2 * candle_min
    ax_lam.set_xticks(centers_min[bar_bins])
    ax_lam.set_xticklabels([f"{c:.0f}" for c in centers_min[bar_bins]],
                           fontsize=7, rotation=90)
    ax_lam.set_xlabel("segment length (minutes)")
    ax_lam.set_ylabel("count (log scale)")
    ax_lam.set_yscale("log")   # keeps the smooth-lambda histograms visible next to the 0.33-lambda spike
    ax_lam.set_title(f"{front_symbol} L1 trend segment lengths over {n_days} days, "
                     f"${candle_min:g}-candles$, $\\lambda = {lam0:.3f}$",
                     loc="right")
    ax_lam.grid(linestyle="--", linewidth=0.15, axis="y")
    ax_lam.legend()

    # --- bottom panel: one histogram per candle width, overlaid on shared
    # minute-valued bins; light -> dark encodes narrow -> wide candles
    candle_colors = plt.cm.YlOrBr(np.linspace(0.45, 1.0, len(candle_sweep)))

    # bins live in minutes so the widths share an axis; half-integer edges so
    # the (integer-minute) segment lengths never straddle a bin boundary
    max_min = max(lengths.max() * c_min for c_min, _, lengths in candle_sweep)
    bin_step = max(1, int(np.ceil(max_min / 60)))
    bins = np.arange(0.5, max_min + bin_step + 0.5, bin_step)

    # overlaid histogram per width: translucent fill with a solid outline in the
    # same hue; each series is weighted to sum to 100% so widths with different
    # segment counts stay comparable; the per-series lambda sits in the legend
    bar_bins = np.zeros(bins.size - 1, dtype=bool)
    for (c_min, lam, lengths), color in zip(candle_sweep, candle_colors):
        counts, _, _ = ax_candle.hist(lengths * c_min, bins=bins,
                                      weights=np.full(lengths.size, 100.0 / lengths.size),
                                      histtype="stepfilled",
                                      facecolor=to_rgba(color, 0.35), edgecolor=color, linewidth=1.2,
                                      label=f"{c_min:g}-min candles ($\\lambda = {lam:.3f}$): "
                                            f"{lengths.size} segments, "
                                            f"mean {lengths.mean():.0f} bars ({lengths.mean() * c_min:.0f} min)")
        bar_bins |= counts > 0

    centers_min = (bins[:-1] + bins[1:]) / 2
    ax_candle.set_xticks(centers_min[bar_bins])
    ax_candle.set_xticklabels([f"{c:.0f}" for c in centers_min[bar_bins]],
                              fontsize=7, rotation=90)
    ax_candle.set_xlabel("segment length (minutes)")
    ax_candle.set_ylabel("share of segments (%, log scale)")
    ax_candle.set_yscale("log")   # keeps the wide-candle histograms visible next to the narrow-candle spike
    ax_candle.set_title(f"{front_symbol} L1 trend segment lengths over {n_days} days, "
                        f"$\\lambda = 10^{{-3}}\\,\\lambda_{{max}}$ per width",
                        loc="right")
    ax_candle.grid(linestyle="--", linewidth=0.15, axis="y")
    ax_candle.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
