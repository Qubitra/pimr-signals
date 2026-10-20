# pimr-signals
Physics-Informed Market Signal via Mean Reversion

Summary: 
This is the first phase of the project to build a mean-reversion signal engine grounded in mean reverting (oscillation physics) dynamics which will evolve into a physics-informed machine learning model, and ultimately paired with quantum features to produce a quantum-advantaged trading pipeline.


## Current repo structure
```
src/
└── pimr/                       the importable package
    ├── data/                   market-data loaders + bar aggregation
    │   ├── bars.py             build_ohlc, build_mid
    │   ├── csv_io.py           save_csv, load_csv
    │   └── dbn.py              load_dbn, load_dbn_many, most_traded_instrument  (Databento)
    ├── filters/
    │   ├── hp_filter.py        hp_filter
    │   ├── l1t_filter.py       l1tf_cvxpy, lambda_max, l1_lengths
    │   ├── l1tf_scipy.py       l1_trend_filter_scipy (not re-exported yet)
    │   └── rolling_median.py   median_mad
    ├── indicators/
    │   ├── moving_averages.py  SMA, EMA, EMA_fast
    │   └── rsi.py              gains_and_losses, RS, RSI, rsi
    └── physics/
        ├── oscillator.py       simulate, FE, BE, RK2
        ├── ar_models.py        AR2, AR4, oscillatory_modes
        └── spectrogram.py      spectrogram, spectral_forecast

scripts/                        runnable demos, all argparse-based, run from repo root
├── plot_compare_filters.py     rolling median vs HP vs L1 side by side
├── plot_l1tf_strategy.py       L1 trend bands + RSI + signals
└── plot_l1_segment_histograms.py   L1 segment-length histogram over many days

tests/                          pytest (uv sync --extra dev)
├── test_ar_models.py
└── test_spectrogram.py
```


## Installation (pre-release)

`pimr` is not on PyPI yet, so install it from a clone of this repo. Requires Python 3.12+.

```sh
git clone https://github.com/Qubitra/pimr-signals.git   # or git@github.com:Qubitra/pimr-signals.git
cd pimr-signals
```

With [uv](https://docs.astral.sh/uv/) (recommended; it downloads a matching Python if you don't have one):

```sh
uv sync                # creates .venv/ with pimr + all dependencies
uv sync --extra dev    # optional: also installs pytest
```

With plain pip (Python 3.12+ already installed):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"   # optional: also installs pytest
```

Both routes install `pimr` in editable mode, so edits under `src/pimr/` take effect
immediately without reinstalling. 
