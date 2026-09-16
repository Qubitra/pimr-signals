"""Compare trend filters on gold GLBX candles: rolling median vs. HP vs. L1.

Three panels: candles, the three trend estimates over the mid price, and the
cyclical (detrended) components of each filter.

Extracted from tests/comparison.py.

Usage:
    python scripts/compare_filters.py [path/to/data.csv] [--out fig.png]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # until pimr is pip-installed

import matplotlib.pyplot as plt
import numpy as np

from pimr.data import build_mid, build_ohlc
from pimr.data.dbn import load_dbn_many, most_traded_instrument
from pimr.filters import hp_filter, l1tf_cvxpy, lambda_max, median_mad

DEFAULT_CSV = "/Users/arash/Qubitra/2023-07-14_2023-07-15/GLBX-20260718-CWTFHWMKGY/test-gold-data.csv"
DEFAULT_DBN = "/Users/arash/Qubitra/GC_202608/glbx-mdp3-20260812.mbp-1.dbn.zst"
DAY_NS = 86_400_000_000_000  # matplotlib measures bar width in days on a datetime axis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=[DEFAULT_DBN], help="mbp-1 .dbn.zst files")
    parser.add_argument("--candle-min", type=float, default=5, help="candle width in minutes")
    parser.add_argument("--max-rows", type=int, default=None, help="read at most this many CSV rows")
    parser.add_argument("--price-scale", type=float, default=1e-11,
                        help="Databento fixed-point price scaling factor for this file")
    parser.add_argument("--out", default=None, help="save the figure here instead of showing it")
    parser.add_argument("--dpi", type=int, default=1000, help="resolution when saving with --out")
    args = parser.parse_args(argv)

    candle_ns = int(args.candle_min * 60 * 10**9)

    # gold_data = load_GLBX(args.path, max_rows=args.max_rows, PRICE_SCALE=args.price_scale)
    # is_trade = gold_data["action"] == "T"
    # ohlc = build_ohlc(gold_data["ts"][is_trade],
    #                   gold_data["price"][is_trade],
    #                   gold_data["size"][is_trade],
    #                   bucket_ns=candle_ns)

    # mid_data = build_mid(gold_data["ts"][is_trade],
    #                      gold_data["price"][is_trade],
    #                      bucket_ns=candle_ns)
    
    trades = load_dbn_many(args.paths, trades_only=True)
    print(f"{len(args.paths)} DBN file(s): {trades['ts'].size:,} trades")

    front_id, front_symbol, n_front = most_traded_instrument(
        trades["instrument_id"], trades["symbols"])
    print(f"\nmost-traded instrument: {front_symbol} "
          f"({n_front:,} of {trades['ts'].size:,} trades)")

    is_front = trades["instrument_id"] == front_id
    ohlc = build_ohlc(trades["ts"][is_front],
                      trades["price"][is_front],
                      trades["size"][is_front],
                      bucket_ns=candle_ns)

    lam = 1.e-3 * lambda_max(ohlc["close"])

    med, mad = median_mad(ohlc["close"], window=int(len(ohlc["close"]) / 30))
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

    ax_price.plot(gold_data["ts"][is_trade].astype("datetime64[ns]"), gold_data["mid"][is_trade],
                  label="Mid price", color="black", linewidth=0.35)
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
