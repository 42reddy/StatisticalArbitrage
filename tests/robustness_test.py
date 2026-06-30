"""
Robustness / generalization stress test.

Every other test so far either looks at one fixed split (train/WFV/holdout
in main.py) or shuffles entry timing (entry_signal_permutation_test.py).
Neither asks: does this exact param set hold up across *many different
slices of history*, at *different horizon lengths*? A strategy that only
looks good on the one holdout window we happened to pick could just be
lucky regime alignment.

Method: hedge ratio (beta), OU mean, features and signals are computed
ONCE on the full price history with the params already in params.py
(no re-fitting per draw — we're testing whether the existing params
generalize, not searching for new ones). For each window size in
WINDOW_SIZES, randomly draw N_DRAWS_PER_SIZE start points and backtest
that slice in isolation (flat at the start of every window — no carried
position state across draws). Overlap between draws is allowed on
purpose: we want many independent looks at different historical regimes,
not a non-overlapping partition of a fairly short price history.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data import data_loader
from features import features
from backtest import backtest
from metrics import metrics
from params import PARAMS

RESULTS_TABLES  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'tables')
RESULTS_FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'figures')

T1, T2  = PARAMS['T1'], PARAMS['T2']
CAPITAL = 200_000

WINDOW_SIZES     = [126, 256, 386, 504, 752]
N_DRAWS_PER_SIZE = 20
MIN_TRADES       = 5
SEED             = 42

rng = np.random.default_rng(SEED)

data_loader_obj = data_loader()
feature_builder = features()
backtest_engine = backtest(leverage=4)
metrics_calc    = metrics()

df = data_loader_obj.load_data(needs_fx=False)
print(f"Full dataset : {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)}d)")

# ── Fixed beta / OU mean / features / signals, computed ONCE on the full
# history so every random window is scored against identical feature
# definitions — the only thing that varies between draws is which slice
# of history gets backtested.
beta    = feature_builder.estimate_hedge_ratio(df, T1, T2, lookback=None)
spread  = np.log(df[T1]) - beta * np.log(df[T2])
ou_mean = float(spread.mean())
print(f"  Beta    : {beta:.4f}")
print(f"  OU mean : {ou_mean:.5f}")

feat = feature_builder.build_features(df, PARAMS, ou_mean=ou_mean, beta=beta)[0]
sig  = feature_builder.generate_signals(feat, PARAMS)

# Full-sample reference performance
pnl_full, eq_full, trades_full = backtest_engine.backtest(feat, sig, PARAMS, cost=True)
met_full = metrics_calc.calc_metrics(pnl_full, eq_full, trades_full)
print(f"\nFULL-SAMPLE REFERENCE   Sharpe={met_full['sharpe']:.2f}  "
      f"Calmar={met_full['calmar']:.2f}  AnnRet={met_full['ann_ret']*100:.2f}%  "
      f"trades={met_full['n_trades']}")

# Earliest valid start — rolling features (slow/medium windows, vol-z
# window) need this many real bars behind them before they stop being
# mostly NaN, same warm-up floor used in main.py's WFV.
WARMUP_DAYS = max(3 * PARAMS['medium_window'],
                  int(PARAMS.get('vol_z_window', PARAMS['vol_window'] * 2)) * 2,
                  60)

# ─────────────────────────────────────────────
#  DRAW RANDOM WINDOWS AND BACKTEST EACH ONE
# ─────────────────────────────────────────────
records = []
for size in WINDOW_SIZES:
    max_start = len(df) - size
    if max_start <= WARMUP_DAYS:
        print(f"  Skipping {size}d window — not enough history after warm-up.")
        continue

    starts = rng.integers(WARMUP_DAYS, max_start + 1, size=N_DRAWS_PER_SIZE)
    for start in starts:
        start  = int(start)
        end    = start + size
        feat_w = feat.iloc[start:end]
        sig_w  = sig.iloc[start:end]

        pnl_w, eq_w, trades_w = backtest_engine.backtest(feat_w, sig_w, PARAMS, cost=True)
        met_w = metrics_calc.calc_metrics(pnl_w, eq_w, trades_w)
        met_w.update(
            window_size = size,
            start_date  = df.index[start].date(),
            end_date    = df.index[end - 1].date(),
            low_signal  = met_w['n_trades'] < MIN_TRADES,
        )
        records.append(met_w)

res = pd.DataFrame(records)
res.to_csv(os.path.join(RESULTS_TABLES, 'robustness_results.csv'), index=False)

# ─────────────────────────────────────────────
#  SUMMARY TABLE
# ─────────────────────────────────────────────
def pct(s, cond):
    return float(cond(s).mean() * 100)

summary_rows = []
for size in WINDOW_SIZES:
    sub = res[res['window_size'] == size]
    if len(sub) == 0:
        continue
    clean = sub[~sub['low_signal']]   # exclude degenerate (near-zero-trade) draws
    summary_rows.append(dict(
        window_size       = size,
        n_draws           = len(sub),
        pct_low_signal    = sub['low_signal'].mean() * 100,
        sharpe_median     = clean['sharpe'].median()  if len(clean) else np.nan,
        sharpe_p10        = clean['sharpe'].quantile(0.1) if len(clean) else np.nan,
        sharpe_p90        = clean['sharpe'].quantile(0.9) if len(clean) else np.nan,
        calmar_median     = clean['calmar'].median()  if len(clean) else np.nan,
        ann_ret_median_pct= clean['ann_ret'].median() * 100 if len(clean) else np.nan,
        max_dd_median_pct = clean['max_dd'].median()  * 100 if len(clean) else np.nan,
        win_rate_median   = clean['win_rate'].median() * 100 if len(clean) else np.nan,
        pct_profitable    = pct(clean['ann_ret'], lambda x: x > 0) if len(clean) else np.nan,
        pct_calmar_gt1    = pct(clean['calmar'],  lambda x: x > 1) if len(clean) else np.nan,
        pct_sharpe_gt_half= pct(clean['sharpe'],  lambda x: x > 0.5) if len(clean) else np.nan,
    ))

summary = pd.DataFrame(summary_rows).set_index('window_size')
summary.to_csv(os.path.join(RESULTS_TABLES, 'robustness_summary.csv'))

pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')
print("\n── ROBUSTNESS SUMMARY (degenerate / low-signal draws excluded from medians) ──")
print(summary)

verdict = ("✓ ROBUST across horizons" if (summary['pct_profitable'] > 55).all()
                                         and (summary['sharpe_median'] > 0.3).all() else
           "⚠ MIXED — generalizes on some horizons, not all" if (summary['pct_profitable'] > 50).any() else
           "✗ FRAGILE — does not generalize across random windows")
print(f"\n  Verdict: {verdict}")

# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
_palette = ['#4fc3f7', '#66bb6a', '#ffa726', '#ef5350', '#ce93d8', '#80cbc4']
colors = {size: _palette[i % len(_palette)] for i, size in enumerate(WINDOW_SIZES)}

metrics_to_plot = [
    ('sharpe',  'Sharpe ratio',           1.0),
    ('calmar',  'Calmar ratio',           1.0),
    ('ann_ret', 'Annualized return (%)', 100.0),
    ('max_dd',  'Max drawdown (%)',      100.0),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (col, title, scale) in zip(axes.flat, metrics_to_plot):
    for size in WINDOW_SIZES:
        sub = res.loc[(res['window_size'] == size) & (~res['low_signal']), col].dropna() * scale
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=18, alpha=0.45, label=f"{size}d (n={len(sub)})",
                color=colors[size], edgecolor='none')
    ref_val = met_full[col] * scale
    ax.axvline(ref_val, color='white', ls='--', lw=1.2, alpha=0.8,
               label='full-sample')
    ax.axvline(0, color='gray', ls=':', lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)

fig.suptitle(f"Distribution across {N_DRAWS_PER_SIZE} random draws per window size "
             f"(low-signal draws excluded, <{MIN_TRADES} trades)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(RESULTS_FIGURES, 'robustness_distributions.png'), dpi=140)
print("\nSaved robustness_distributions.png")

fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5))
for ax, (col, title, scale) in zip(axes2, metrics_to_plot):
    data = [res.loc[(res['window_size'] == s) & (~res['low_signal']), col].dropna() * scale
            for s in WINDOW_SIZES]
    bp = ax.boxplot(data, tick_labels=[f"{s}d" for s in WINDOW_SIZES], patch_artist=True)
    for patch, size in zip(bp['boxes'], WINDOW_SIZES):
        patch.set_facecolor(colors[size])
        patch.set_alpha(0.5)
    ax.axhline(met_full[col] * scale, color='black', ls='--', lw=1.0)
    ax.set_title(title, fontsize=10)

fig2.suptitle("Robustness across window sizes (dashed line = full-sample reference)", fontsize=10)
fig2.tight_layout(rect=(0, 0, 1, 0.95))
fig2.savefig(os.path.join(RESULTS_FIGURES, 'robustness_boxplots.png'), dpi=140)
print("Saved robustness_boxplots.png")
