# pimr-signals
Physics-Informed Market Signal via Mean Reversion

Summary: 
This is the first phase of the project to build a mean-reversion signal engine grounded in mean reverting (oscillation physics) dynamics which will evolve into a physics-informed machine learning model, and ultimately paired with quantum features to produce a quantum-advantaged trading pipeline.


## Current repo structure
```
src/
├── pimr/                 the importable package
│   ├── data/             market-data loaders + bar aggregation
│   │   ├── glbx_csv.py   load_GLBX                      (from tests/gold_data.py)
│   │   ├── bars.py       build_ohlc, build_mid          (from tests/gold_data.py)
│   │   ├── dbn.py        load_dbn, load_dbn_many,       (from tests/dbn_data.py;
│   │   │                 most_traded_instrument          needs databento-dbn)
│   │   └── csv_io.py     save_csv, load_csv             (from tests/csv_manager.py)
│   ├── filters/
│   │   ├── hp.py               hp_filter                (from tests/hp_filter.py)
│   │   ├── l1_trend.py         l1t_filter_cvxpy, lambda_max,
│   │   │                       linear_segment_lengths   (from tests/l1t_filter.py)
│   │   ├── l1_trend_scipy.py   l1_trend_filter_scipy    (from tests/l1tf_scipy.py.txt)
│   │   └── rolling_median.py   rolling_median_mad       (from tests/median.py)
│   ├── indicators/
│   │   ├── moving_averages.py  SMA, EMA, EMA_fast       (from tests/relative_strength_index.py)
│   │   └── rsi.py              gains_and_losses, RS, RSI, rsi
│   ├── synthetic/
│   │   ├── random_walk_book.py generate_simple_l1_book  (from tests/synthetic_data.py)
│   │   └── ou_book.py          generate_synthetic_l1_book etc.
│   │                                                    (from tests/synthetic_l1_data.txt,
│   │                                                     ported torch -> numpy)
│   └── physics/
│       └── oscillator.py       simulate, FE, BE, RK2    (from tests/physical_oscillator_simulation.py)
│
├── scripts/              runnable demos (the old __main__ blocks), all argparse-based
│   ├── plot_gold_candles.py    candles + mid line from a GLBX CSV
│   ├── plot_dbn_candles.py     candles + L1 trend from DBN files
│   ├── plot_rsi_strategy.py    RSI(2) SMA-vs-EMA strategy signals
│   ├── plot_l1rsi_strategy.py  L1RSI2 strategy (trend bands + RSI + signals)
│   ├── compare_filters.py      rolling median vs HP vs L1 side by side
│   ├── plot_lambda_sweep.py    L1 trends for 0.33x..3x lambda
│   ├── plot_segment_hist.py    L1 segment-length histogram over many days
│   ├── demo_synthetic_book.py  random-walk book + rolling median/MAD
│   ├── demo_ou_book.py         OU book demo
│   └── demo_oscillator.py      solve_ivp vs FE/BE/RK2
│
└── experiments/          prototypes, not part of the package
    ├── l1tf_interior_point.py  hand-rolled IPM L1 trend filter (from tests/tess.py)
    └── random_arrays.py        random trend generators (from tests/random_plot.py.txt)
```

## Running (in progress)
Scripts insert `src/` on `sys.path` themselves, so from the repo root:

```sh
.venv/bin/python src/scripts/demo_synthetic_book.py
.venv/bin/python src/scripts/plot_dbn_candles.py /path/to/file.dbn.zst --candle-min 5
```

Every script accepts `--help`. Data-driven scripts default to the local
Databento paths used during development; pass the file path as the first
argument to override.