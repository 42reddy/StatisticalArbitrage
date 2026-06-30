# Statistical arbitrage

Backtest for a **pairs / spread** strategy: load two tickers, estimate a hedge ratio on training data, build z-score signals from the spread, run a leveraged backtest with costs, and report train, walk-forward, and holdout performance (plus charts).

## Prerequisites

- **Python 3.10+** recommended (matches typical use of the listed packages).
- **Internet** on first run: prices are pulled via `yfinance` (see `data.py`).

## Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure (`params.py`)

All strategy inputs live in the `PARAMS` dictionary at the top of `params.py`.

| Area | Keys (examples) | Role |
|------|-----------------|------|
| Universe | `T1`, `T2` | Yahoo Finance symbols for the pair (e.g. `HAL.NS`, `BDL.NS`). |
| Feature windows | `slow_window`, `medium_window`, `fast_span`, `vol_window` | Rolling stats for the spread and volatility features. |
| Entries / exits | `z_entry_long`, `z_entry_short`, `z_exit_long`, `z_exit_short` | Z-score levels to open and flatten. |
| Stops | `z_stop_long`, `z_stop_short` | Stop-outs by side. |
| Pyramid / risk | `z_add`, `vol_cap`, `max_hold` | Add-on level, vol filter, max holding period. |
| Legacy | `z_entry`, `z_exit`, `z_stop` | Kept for compatibility; some paths average the long/short pairs. |
| Regime filter | `autocorr_window`, `autocorr_threshold`, `ou_adapt_span` | Autocorrelation and OU-related tuning. |

`main.py` imports `PARAMS` and may **update** it after Bayesian optimisation on the training sample (e.g. refined thresholds, `beta`). Your edits are the **starting point** for that search.

**Account size:** starting capital is set in `main.py` as `CAPITAL` and should stay aligned with `backtest.TRADE_CAPITAL` if you change it.

## Run

```bash
python main.py
```

### What this does (high level)

1. **Load & split** — Historical data for `T1` / `T2`; last ~504 trading days reserved as **holdout**; the rest is **train**.
2. **Hedge ratio** — `beta` is estimated **once** on the full train window (not re-fit on holdout).
3. **Bayesian optimisation (Optuna)** — Tunes parameters on **train only** (see `diagnosis.py`); results are merged back into `PARAMS`.
4. **Walk-forward validation** — Rolling train/test windows **inside** train, with fixed post-Bayes parameters, to stress out-of-sample behaviour.
5. **Full train backtest** — Metrics and diagnostics printed; **robustness gate** compares train vs walk-forward Sharpe/Calmar.
6. **Holdout** — If the gate passes (limited Sharpe degradation), the same fixed `beta` and optimised params are applied to the holdout window; otherwise holdout is skipped and plots use train only.

Expect console tables (returns, drawdown, Sharpe, Calmar, trade stats) and matplotlib output from the plotting module.

## Project layout (short)

- `main.py` — Entry point: orchestrates data, features, diagnosis, backtest, metrics, plots.
- `params.py` — `PARAMS` dict for symbols and strategy knobs.
- `data.py`, `features.py`, `backtest.py`, `metrics.py`, `plotting.py`, `diagnosis.py` — Supporting modules.
