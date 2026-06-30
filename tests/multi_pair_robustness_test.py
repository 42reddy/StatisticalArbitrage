"""
Multi-pair portfolio robustness test.

multi_pair_backtest.py's shared-capital portfolio looks very good (Sharpe
~1.4, 95% win rate, Calmar ~2.7) — good enough to be suspicious. This script
runs the portfolio result through several checks aimed specifically at
"too good to be true": is the edge stable through time, is it secretly
riding on one pair, does it survive resampling that breaks the lucky
ordering of trades, does it hold up on random sub-periods, and is the
result fragile to the (somewhat arbitrary) tie-break rule used to decide
which pair gets capital when two signal on the same bar?

Reuses MultiPairBacktest from multi_pair_backtest.py — data is loaded and
every pair's standalone trades are computed once; everything below operates
on that already-computed per-pair P&L/trades, so no re-fitting happens.

[1] Rolling 252d Sharpe & annualized return — is performance stable through
    time, or is the headline number an average of a few great stretches and
    long mediocre/negative ones?
[2] Leave-one-pair-out — drop each pair in turn and recompute portfolio
    Sharpe. A portfolio that collapses when one pair is removed isn't
    really diversified, whatever the correlation table says.
[3] Block bootstrap of daily P&L — resamples contiguous blocks (preserves
    short-range autocorrelation from multi-day holds) to build a Sharpe
    confidence interval and estimate P(Sharpe <= 0). Tests whether the
    point-estimate Sharpe is solid or a few lucky trades extrapolated.
[4] Random sub-window robustness — many random slices of history at a few
    window lengths, profitable-window / Sharpe>0.5 hit rates. Same idea as
    robustness_test.py but at the portfolio level.
[5] Capital tie-break sensitivity — the live allocator breaks same-day ties
    by largest |entry_z|. Re-run with randomized tie-breaks many times: if
    portfolio Sharpe swings wildly, the headline result is an artifact of
    that specific rule, not the underlying edge.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import sys, os
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'scripts'))

from scripts.multi_pair_backtest import MultiPairBacktest
from metrics import metrics

RESULTS_TABLES  = os.path.join(_REPO_ROOT, 'results', 'tables')
RESULTS_FIGURES = os.path.join(_REPO_ROOT, 'results', 'figures')

SEED = 42
rng = np.random.default_rng(SEED)

ROLL_WINDOW = 252          # [1] rolling Sharpe/ann-ret window
BOOT_BLOCK  = 21           # [3] block bootstrap block length (~1 month)
N_BOOT      = 2000         # [3] bootstrap draws
SUBWIN_SIZES     = [32, 63, 126, 252, 504, 756]   # [4] random sub-window lengths
N_DRAWS_PER_SIZE = 50                      # [4]
N_TIEBREAK_DRAWS = 500     # [5]

metrics_calc = metrics()


def sharpe_of(pnl):
    ann_ret = pnl.mean() * 252
    ann_vol = pnl.std() * np.sqrt(252)
    return ann_ret / ann_vol if ann_vol > 0 else 0.0


# ─────────────────────────────────────────────
#  BUILD THE BASE PORTFOLIO (reused by every test below)
# ─────────────────────────────────────────────
print("Loading pairs and running the base multi-pair portfolio...")
mp = MultiPairBacktest(verbose=True)
mp.load_pairs()
mp.run()

pnl_s, equity_s, trades_df = mp.pnl_s, mp.equity_s, mp.trades_df
print(f"\nBase portfolio: Sharpe={mp.met['sharpe']:.2f}  Calmar={mp.met['calmar']:.2f}  "
      f"AnnRet={mp.met['ann_ret']*100:.1f}%  n_trades={mp.met['n_trades']}  "
      f"span={pnl_s.index[0].date()} -> {pnl_s.index[-1].date()}")

print("\n" + "═" * 70)
print("  ROBUSTNESS TESTS")
print("═" * 70)

# ─────────────────────────────────────────────
#  [1] ROLLING 252-DAY SHARPE / ANNUALIZED RETURN
# ─────────────────────────────────────────────
print(f"\n[1] Rolling {ROLL_WINDOW}d Sharpe & annualized return")
roll_ann_ret = pnl_s.rolling(ROLL_WINDOW).mean() * 252
roll_ann_vol = pnl_s.rolling(ROLL_WINDOW).std() * np.sqrt(252)
roll_sharpe  = (roll_ann_ret / roll_ann_vol).replace([np.inf, -np.inf], np.nan)

roll = pd.DataFrame({'ann_ret': roll_ann_ret, 'sharpe': roll_sharpe}).dropna()
pct_negative_sharpe = (roll['sharpe'] < 0).mean() * 100 if len(roll) else float('nan')
pct_below_half      = (roll['sharpe'] < 0.5).mean() * 100 if len(roll) else float('nan')
flag = "  <-- RED: meaningful stretches of negative rolling Sharpe" if pct_negative_sharpe > 15 else ""
print(f"  Rolling Sharpe  : min={roll['sharpe'].min():.2f}  median={roll['sharpe'].median():.2f}  "
      f"max={roll['sharpe'].max():.2f}")
print(f"  % of days with rolling Sharpe < 0   : {pct_negative_sharpe:.1f}%{flag}")
print(f"  % of days with rolling Sharpe < 0.5 : {pct_below_half:.1f}%")

# ─────────────────────────────────────────────
#  [2] LEAVE-ONE-PAIR-OUT
# ─────────────────────────────────────────────
print("\n[2] Leave-one-pair-out (does the portfolio collapse without one pair?)")
loo_rows = []
for dropped in mp.names:
    subset = {n: r for n, r in mp.per_pair.items() if n != dropped}
    sub_trades, sub_pnl, sub_equity = mp.allocate_capital(subset)
    sub_met = metrics_calc.calc_metrics(sub_pnl, sub_equity, sub_trades)
    loo_rows.append(dict(dropped_pair=dropped, sharpe=sub_met['sharpe'],
                          calmar=sub_met['calmar'], ann_ret=sub_met['ann_ret'],
                          n_trades=sub_met['n_trades']))

loo = pd.DataFrame(loo_rows).set_index('dropped_pair')
loo['sharpe_drop_pct'] = (mp.met['sharpe'] - loo['sharpe']) / mp.met['sharpe'] * 100
print(f"  Full portfolio Sharpe: {mp.met['sharpe']:.2f}")
print(loo[['n_trades', 'sharpe', 'ann_ret', 'sharpe_drop_pct']].round(2).to_string())
worst = loo['sharpe_drop_pct'].idxmax()
flag = ("  <-- RED: portfolio leans heavily on one pair"
        if loo.loc[worst, 'sharpe_drop_pct'] > 40 else "  <-- GREEN: no single pair is load-bearing")
print(f"  Largest single-pair dependence: dropping {worst} cuts Sharpe by "
      f"{loo.loc[worst, 'sharpe_drop_pct']:.0f}%{flag}")

# ─────────────────────────────────────────────
#  [3] BLOCK BOOTSTRAP OF DAILY P&L
# ─────────────────────────────────────────────
print(f"\n[3] Block bootstrap ({N_BOOT} draws, {BOOT_BLOCK}d blocks) — Sharpe confidence interval")
pnl_vals = pnl_s.values
n = len(pnl_vals)
n_blocks = int(np.ceil(n / BOOT_BLOCK))

boot_sharpes = np.empty(N_BOOT)
for i in range(N_BOOT):
    starts = rng.integers(0, n - BOOT_BLOCK + 1, size=n_blocks)
    sample = np.concatenate([pnl_vals[s:s + BOOT_BLOCK] for s in starts])[:n]
    ann_ret = sample.mean() * 252
    ann_vol = sample.std() * np.sqrt(252)
    boot_sharpes[i] = ann_ret / ann_vol if ann_vol > 0 else 0.0

boot_ci_lo, boot_ci_hi = np.percentile(boot_sharpes, [2.5, 97.5])
p_sharpe_le_0 = float((boot_sharpes <= 0).mean())
print(f"  Point-estimate Sharpe : {mp.met['sharpe']:.2f}")
print(f"  Bootstrap 95% CI      : [{boot_ci_lo:.2f}, {boot_ci_hi:.2f}]")
print(f"  P(Sharpe <= 0)        : {p_sharpe_le_0*100:.1f}%")
flag = "  <-- RED: Sharpe not reliably distinguishable from zero" if p_sharpe_le_0 > 0.05 else "  <-- GREEN: Sharpe robust to resampling"
print(f"  {flag.strip()}")

# ─────────────────────────────────────────────
#  [4] RANDOM SUB-WINDOW ROBUSTNESS (portfolio level)
# ─────────────────────────────────────────────
print(f"\n[4] Random sub-window robustness ({N_DRAWS_PER_SIZE} draws per window size)")
subwin_records = []
for size in SUBWIN_SIZES:
    max_start = n - size
    if max_start <= 0:
        print(f"  Skipping {size}d window — longer than available history.")
        continue
    starts = rng.integers(0, max_start + 1, size=N_DRAWS_PER_SIZE)
    for start in starts:
        start = int(start)
        seg = pnl_s.iloc[start:start + size]
        ann_ret = seg.mean() * 252
        ann_vol = seg.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        subwin_records.append(dict(window_size=size, sharpe=sharpe, ann_ret=ann_ret))

subwin = pd.DataFrame(subwin_records)
print(f"  {'Window':<8}{'pct_profitable':>16}{'pct_sharpe>0.5':>16}{'sharpe_median':>16}")
for size in SUBWIN_SIZES:
    sub = subwin[subwin['window_size'] == size]
    if len(sub) == 0:
        continue
    pct_profit = (sub['ann_ret'] > 0).mean() * 100
    pct_sharpe_half = (sub['sharpe'] > 0.5).mean() * 100
    print(f"  {size:<8}{pct_profit:>15.1f}%{pct_sharpe_half:>15.1f}%{sub['sharpe'].median():>16.2f}")

# ─────────────────────────────────────────────
#  [5] CAPITAL TIE-BREAK SENSITIVITY
# ─────────────────────────────────────────────
print(f"\n[5] Capital tie-break sensitivity ({N_TIEBREAK_DRAWS} randomized-tiebreak draws)")


def allocate_capital_random_tiebreak(mp_obj, per_pair_results, rng):
    all_trades = pd.concat([r['trades'] for r in per_pair_results.values()], ignore_index=True)
    all_trades['_tiebreak'] = rng.random(len(all_trades))
    all_trades = all_trades.sort_values(['entry_date', '_tiebreak']).reset_index(drop=True)

    accepted = []
    busy_until = None
    for _, t in all_trades.iterrows():
        if busy_until is None or t['entry_date'] > busy_until:
            accepted.append(t)
            busy_until = t['exit_date']

    trades = (pd.DataFrame(accepted).drop(columns=['_tiebreak']).reset_index(drop=True)
              if accepted else all_trades.drop(columns=['_tiebreak']).iloc[0:0])

    master_idx = sorted(set().union(*(r['pnl'].index for r in per_pair_results.values())))
    master_idx = pd.DatetimeIndex(master_idx)
    pnl = pd.Series(0.0, index=master_idx)
    for _, t in trades.iterrows():
        seg = per_pair_results[t['pair']]['pnl'].loc[t['entry_date']:t['exit_date']]
        pnl.loc[seg.index] += seg.values
    return pnl


tiebreak_sharpes = np.empty(N_TIEBREAK_DRAWS)
for i in range(N_TIEBREAK_DRAWS):
    pnl_i = allocate_capital_random_tiebreak(mp, mp.per_pair, rng)
    tiebreak_sharpes[i] = sharpe_of(pnl_i)

tb_lo, tb_hi = np.percentile(tiebreak_sharpes, [2.5, 97.5])
print(f"  Live tie-break (largest |entry_z|) Sharpe : {mp.met['sharpe']:.2f}")
print(f"  Random tie-break Sharpe range (95%)        : [{tb_lo:.2f}, {tb_hi:.2f}]  "
      f"median={np.median(tiebreak_sharpes):.2f}")
flag = ("  <-- RED: result is sensitive to an arbitrary tie-break rule"
        if not (tb_lo <= mp.met['sharpe'] <= tb_hi or abs(np.median(tiebreak_sharpes) - mp.met['sharpe']) / mp.met['sharpe'] < 0.25)
        else "  <-- GREEN: tie-break rule isn't doing the heavy lifting")
print(f"  {flag.strip()}")

print("\n" + "═" * 70 + "\n")

# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(15, 13))

ax = axes[0, 0]
ax.plot(roll.index, roll['sharpe'], color='steelblue', lw=1.0)
ax.axhline(0, color='gray', ls=':', lw=0.8)
ax.axhline(mp.met['sharpe'], color='black', ls='--', lw=1.0, label='full-sample')
ax.set_title(f'Rolling {ROLL_WINDOW}d Sharpe')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(roll.index, roll['ann_ret'] * 100, color='seagreen', lw=1.0)
ax.axhline(0, color='gray', ls=':', lw=0.8)
ax.axhline(mp.met['ann_ret'] * 100, color='black', ls='--', lw=1.0, label='full-sample')
ax.set_title(f'Rolling {ROLL_WINDOW}d annualized return (%)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
order = loo['sharpe_drop_pct'].sort_values()
colors = ['firebrick' if v > 40 else 'steelblue' for v in order.values]
ax.barh(order.index, order.values, color=colors)
ax.axvline(0, color='gray', lw=0.8)
ax.set_title('Sharpe drop (%) when pair is removed')
ax.grid(alpha=0.3, axis='x')

ax = axes[1, 1]
ax.hist(boot_sharpes, bins=40, color='slateblue', alpha=0.7, edgecolor='none')
ax.axvline(mp.met['sharpe'], color='black', ls='--', lw=1.2, label='point estimate')
ax.axvline(0, color='gray', ls=':', lw=0.8)
ax.set_title(f'Block-bootstrap Sharpe distribution\n(P(Sharpe<=0)={p_sharpe_le_0*100:.1f}%)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2, 0]
_palette = ['#4fc3f7', '#66bb6a', '#ffa726', '#ef5350']
for i, size in enumerate(SUBWIN_SIZES):
    sub = subwin.loc[subwin['window_size'] == size, 'sharpe']
    if len(sub) == 0:
        continue
    ax.hist(sub, bins=15, alpha=0.45, label=f'{size}d', color=_palette[i % len(_palette)])
ax.axvline(mp.met['sharpe'], color='black', ls='--', lw=1.2)
ax.axvline(0, color='gray', ls=':', lw=0.8)
ax.set_title('Random sub-window Sharpe distribution')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2, 1]
ax.hist(tiebreak_sharpes, bins=40, color='darkorange', alpha=0.7, edgecolor='none')
ax.axvline(mp.met['sharpe'], color='black', ls='--', lw=1.2, label='live tie-break')
ax.set_title('Sharpe under randomized capital tie-break')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle('Multi-pair portfolio robustness tests', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(RESULTS_FIGURES, 'multi_pair_robustness_test.png'), dpi=130)
print("Saved plot -> multi_pair_robustness_test.png")

roll.to_csv(os.path.join(RESULTS_TABLES, 'multi_pair_rolling_sharpe.csv'))
loo.to_csv(os.path.join(RESULTS_TABLES, 'multi_pair_leave_one_out.csv'))
subwin.to_csv(os.path.join(RESULTS_TABLES, 'multi_pair_subwindow_robustness.csv'), index=False)
print("Saved multi_pair_rolling_sharpe.csv, multi_pair_leave_one_out.csv, "
      "multi_pair_subwindow_robustness.csv")
