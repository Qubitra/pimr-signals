"""
L1TF strategy on gold DBN data: L1 trend line with sigma bands plus RSI.

Figure 1: candles, L1 trend with ±sigma and ±sqrt(2)*sigma bands, 
and the distance-to-trend and its per-bar change (velocity) underneath.

Usage:
    python scripts/plot_l1tf_strategy.py [file1.dbn.zst ...]
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


## DEVELOPMENT: Until pimr is pip-installed use the following:
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))  

from pimr.data import build_ohlc
from pimr.data.dbn import load_dbn
from pimr.filters import l1tf_cvxpy, lambda_max, l1_lengths
from pimr.indicators import rsi

DEFAULT_DBN = "/... path to .dbn.zst file ..."
DAY_NS = 86_400_000_000_000  # matplotlib measures bar width in days on a datetime axis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", default=[DEFAULT_DBN], help=" a single .dbn.zst file")
    parser.add_argument("--candle-min", type=float, default=5, help="candle width in minutes")
    args = parser.parse_args(argv)

    candle_min= args.candle_min
    candle_ns = int(candle_min * 60 * 10**9)

    print(f"l1 trend of gold from GLBX {candle_min:g}-minute candles.\n")
    print(f"{len(args.path)} DBN file(s): ", args)

    trades = load_dbn(args.path, most_traded=True)
    
    print(f"{len(args.path)} DBN file(s): {trades['ts'].size:,} trades")

    ohlc = build_ohlc(trades["ts"],
                      trades["price"],
                      trades["size"],
                      bucket_ns=candle_ns)

    lam = 1.e-3 * lambda_max(ohlc["close"])
    l1_trend = l1tf_cvxpy(ohlc["close"], lambda_param=lam)
    l1_dist = ohlc["close"] - l1_trend

    seg_lengths, knots = l1_lengths(l1_trend, return_knots=True)
    print(f"{seg_lengths.size} linear segments" f"mean length: {seg_lengths.mean():.1f} bars = {seg_lengths.mean() * candle_min:.0f} minutes, "
          f"min/max: {seg_lengths.min()}/{seg_lengths.max()} bars\n")

    up_down = np.where(ohlc["close"] >= ohlc["open"], "green", "red")
    t_candle = (ohlc["bucket_start"] + candle_ns // 2).astype("datetime64[ns]")

    def draw_candles_and_trend(ax):
        ax.grid(linestyle="--", linewidth=0.15)
        ax.bar(t_candle, ohlc["high"] - ohlc["low"], bottom=ohlc["low"], width=0.1 * candle_ns / DAY_NS, color=up_down)    # wicks
        ax.bar(t_candle, ohlc["close"] - ohlc["open"], bottom=ohlc["open"], width=0.8 * candle_ns / DAY_NS, color=up_down)  # bodies
        ax.plot(t_candle, l1_trend, color="black", linewidth=0.8)
        ax.plot(t_candle, l1_trend + np.std(l1_dist) * np.sqrt(2), color="gray", linewidth=0.45, linestyle="dotted")
        ax.plot(t_candle, l1_trend + np.std(l1_dist), color="gray", linewidth=0.6)
        ax.plot(t_candle, l1_trend - np.std(l1_dist), color="gray", linewidth=0.6)
        ax.plot(t_candle, l1_trend - np.std(l1_dist) * np.sqrt(2), color="darkgray", linewidth=0.45, linestyle="dotted")
        ax.set_ylabel("price ($)")
        ax.set_title(f"L1RSI2 Strategy Gold: {candle_min:g}-minute candles, ", loc="right", y=0.9)

    # figure 1: strategy signals on candles + distances
    fig, (ax_candle, ax_dist) = plt.subplots(2, 1, sharex=True, height_ratios=[4, 2.5], figsize=(13, 8))

    draw_candles_and_trend(ax_candle)

    ax_dist.grid(linestyle="--", linewidth=0.15)
    ax_dist.plot(t_candle, ohlc["close"] - l1_trend, label="$d$", color="black", linewidth=0.8)
    ax_dist.plot(t_candle[0:-1], np.diff(ohlc["close"] - l1_trend), label="$v$", color="blue", linewidth=0.8)
    ax_dist.hlines(np.std(l1_dist), t_candle[0], t_candle[-1], label="$\\sigma$", linestyle="dotted", color="dimgray", linewidth=0.5)
    ax_dist.hlines(np.std(l1_dist) * np.sqrt(2), t_candle[0], t_candle[-1], label="$\\sqrt{{2}}*\\sigma$", linestyle="dotted", color="darkgray", linewidth=0.5)
    ax_dist.hlines(0, t_candle[0], t_candle[-1], linestyle="-", color="black", linewidth=0.5)

    ax_dist.set_ylabel("distance & velocity")
    ax_dist.legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()


if __name__ == "__main__":
    main()
