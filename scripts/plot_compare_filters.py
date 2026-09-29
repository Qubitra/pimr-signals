"""
Compare trend filters on gold GLBX candles: rolling median vs. HP vs. L1.

Three panels: candles, the three trend estimates over the mid price, and the
cyclical (detrended) components of each filter.

Usage:
    python scripts/compare_filters.py path/to/data.dbn.zst [--out fig.png]
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

## DEVELOPMENT: Until pimr is pip-installed use the following:
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))  

from pimr.data import build_mid, build_ohlc
from pimr.data.dbn import load_dbn
from pimr.filters import hp_filter, l1tf_cvxpy, lambda_max, median_mad

DEFAULT_DBN = "/... path to .dbn.zst file ..."
DAY_NS = 86_400_000_000_000  # matplotlib measures bar width in days on a datetime axis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", default=DEFAULT_DBN, help="A single .dbn.zst file")
    parser.add_argument("--candle-min", type=float, default=5, help="candle width in minutes")
    parser.add_argument("--max-rows", type=int, default=None, help="read at most this many CSV rows")
    parser.add_argument("--price-scale", type=float, default=1e-11, help="Databento fixed-point price scaling factor for this file")
    parser.add_argument("--out", default=None, help="save the figure here instead of showing it")
    parser.add_argument("--dpi", type=int, default=800, help="resolution when saving with --out")
    args = parser.parse_args(argv)

    candle_ns = int(args.candle_min * 60 * 10**9)
    
    gold_trades = load_dbn(args.path, most_traded=True)
    print(f"{len(args.path)} DBN file(s): {gold_trades['ts'].size:,} trades")

    ohlc = build_ohlc(gold_trades["ts"],
                      gold_trades["price"],
                      gold_trades["size"],
                      bucket_ns=candle_ns)

    gold_mid = build_mid(gold_trades['ts'], gold_trades['price'], bucket_ns=candle_ns)
    
    lam = 1.e-3 * lambda_max(ohlc["close"])

    med, mad = median_mad(ohlc["close"], window=int(len(ohlc["close"]) / 50))
    med_cyc = ohlc["close"] - med
    
    hp_trend, hp_cyc = hp_filter(ohlc["close"], lambda_param=1e+2 * lam)
    
    l1_trend = l1tf_cvxpy(ohlc["close"], lambda_param=lam)
    l1t_cyc = ohlc["close"] - l1_trend
    
    up_down = np.where(ohlc["close"] >= ohlc["open"], "green", "red")
    t_candle = (ohlc["bucket_start"] + candle_ns // 2).astype("datetime64[ns]")

    fig, (ax_candle, ax_price, ax_cycle) = plt.subplots(3, 1, sharex=True, figsize=(9, 9))

    ax_candle.grid(linestyle="--", linewidth=0.15)
    ax_candle.bar(t_candle, ohlc["high"] - ohlc["low"], bottom=ohlc["low"], width=0.1 * candle_ns / DAY_NS, color=up_down)    # wicks
    ax_candle.bar(t_candle, ohlc["close"] - ohlc["open"], bottom=ohlc["open"], width=0.8 * candle_ns / DAY_NS, color=up_down)  # bodies
    ax_candle.plot(t_candle, ohlc["close"], label="Closing Price", color="black", linewidth=0.5)
    ax_candle.set_ylabel("price ($)")
    ax_candle.set_title(f"{Path(args.path).stem} ({args.candle_min:g}min)", loc="right", y=0.9)

    ax_price.plot(t_candle, gold_mid['mid'], label="Mid price", color="black", linewidth=0.35)
    ax_price.plot(t_candle, med, label="Rolling Median", color="tab:blue", linewidth=1.0)
    ax_price.plot(t_candle, hp_trend, label="HP-trend", color="tab:green", linewidth=1.0)
    ax_price.plot(t_candle, l1_trend, label="l1-trend", color="tab:red", linewidth=1.0)
    ax_price.set_ylabel("filters")
    ax_price.legend()
    ax_price.grid(linestyle="--", linewidth=0.15)

    ax_cycle.plot(t_candle, med_cyc, label="mid-mad", color="black", linewidth=1)
    ax_cycle.plot(t_candle, hp_cyc, label="hp", color="tab:green", linewidth=1)
    ax_cycle.plot(t_candle, l1t_cyc, label="l1", color="tab:red", linewidth=1)
    ax_cycle.set_ylabel("Cycles")
    ax_cycle.legend()
    ax_cycle.grid(linestyle="--", linewidth=0.15)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=args.dpi)
        print(f"saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
