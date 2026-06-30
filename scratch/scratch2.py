import os
import pandas as pd
import numpy as np

RESULTS_TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'tables')

# z_fast/agreement/regime_score/vol_z/vol_ratio are entry-gate inputs —
# generate_signals() conditions on them directly (agree<=-2, vr<vol_cap,
# etc.), so at the moment of entry their distribution is artificially
# narrow/saturated by construction. Any "separation" test run only on
# entry-time snapshots is structurally blind for these — exclude them
# rather than reading the resulting ~0 as "no signal".
DIAG_COLS = [
    'z_fast_slope',
    'z_fast_accel',
    'rsi_lr',
    'rsi_slope',
    'lr_mom',
    'lr_mom_slope',
    'autocorr',
]

# ── Load ──────────────────────────────────────────────────────────────────────
# MODE='single_pair': original behaviour — train+WFV for one pair (holdout
#   deliberately excluded; see note below). 'split' = train/wfv.
# MODE='multipair': trades_multipair.csv from multi_pair_backtest.py, pooled
#   across several pairs with guards disabled. 'split' = pair name, so the
#   stability check below asks "does this feature's separation hold up
#   across DIFFERENT PAIRS" rather than across train/wfv — exactly what's
#   needed to set COLLECTIVE (not single-pair-overfit) guard thresholds.
#
# Holdout is deliberately excluded from single_pair mode. This script's whole
# job is to mine entry-time features for filters that then get hardcoded into
# generate_signals() — that's a form of fitting, and holdout exists to give
# one untouched, unbiased read on the final strategy. If we use it here too,
# it stops being holdout. WFV is the largest genuinely out-of-sample bucket
# (each fold scored on data the Bayes search never touched) — combine it with
# train for a bigger n, but keep the split label so we can still check
# whether a feature's separation is stable across the two rather than a
# fluke of one small sample.
MODE = 'multipair'   # 'single_pair' or 'multipair'

if MODE == 'single_pair':
    splits = {
        'train': "trades_train.csv",
        'wfv':   "trades_wfv.csv",
    }
    frames = []
    for split_name, fname in splits.items():
        try:
            df = pd.read_csv(os.path.join(RESULTS_TABLES, fname), parse_dates=['entry_date', 'exit_date'])
        except FileNotFoundError:
            continue
        df['split'] = split_name
        frames.append(df)
    trades = pd.concat(frames, ignore_index=True)
else:
    trades = pd.read_csv(os.path.join(RESULTS_TABLES, "trades_multipair.csv"), parse_dates=['entry_date', 'exit_date'])
    trades['split'] = trades['pair']

wins   = trades[trades['pnl'] > 0]
losses = trades[trades['pnl'] < 0]

print(f"\nTotal trades : {len(trades)}  |  Wins : {len(wins)}  |  Losses : {len(losses)}")
print(f"Win rate     : {len(wins)/len(trades)*100:.1f}%")
print("By split     : " +
      ", ".join(f"{s}={len(g)}" for s, g in trades.groupby('split')) + "\n")

# ── Per-feature discrimination table ─────────────────────────────────────────
# For each feature, compare its median value at entry for wins vs losses.
# separation = (win_median - loss_median) / pooled_std
# Higher absolute separation → feature better distinguishes good trades.
#
# For cols where a LOWER value is "correct" at entry (oversold RSI,
# negative autocorr, negative momentum for a long), we flip the sign
# so that separation is still positive when wins have the "right" value.

FLIP_COLS = {'rsi_lr', 'autocorr', 'lr_mom'}   # lower = better signal for longs

rows = []
for col in DIAG_COLS:
    if col not in trades.columns:
        continue

    for direction in ['long', 'short']:
        w = wins[wins['direction']   == direction][col].dropna()
        l = losses[losses['direction'] == direction][col].dropna()

        if len(w) < 3 or len(l) < 3:
            continue

        w_med      = w.median()
        l_med      = l.median()
        pooled_std = pd.concat([w, l]).std()
        sep        = (w_med - l_med) / pooled_std if pooled_std > 0 else 0.0

        if col in FLIP_COLS:
            sep = -sep   # lower is better, so flip so positive = wins are better

        rows.append({
            'feature':     col,
            'direction':   direction,
            'n_wins':      len(w),
            'n_losses':    len(l),
            'win_median':  round(w_med, 4),
            'loss_median': round(l_med, 4),
            'separation':  round(sep, 3),
        })

df_summary = (pd.DataFrame(rows)
                .sort_values('separation', key=abs, ascending=False))

print("── Feature Discrimination Table (pooled across splits) ───────────────────")
print("  separation = (win_median − loss_median) / pooled_std")
print("  Positive → wins entered with a better reading of this feature.")
print("  Negative → feature was actually worse at entry for winning trades.\n")
print(df_summary.to_string(index=False))

# ── Stability across splits ──────────────────────────────────────────────────
# A separation computed on ~20-45 trades is noisy enough to flip sign on the
# next run. Only trust a feature if it points the same direction across all
# splits — not just in the pooled number above. In single_pair mode that's
# train vs WFV (holdout stays out of this entirely, by design — see the note
# on `MODE`/`splits` above). In multipair mode that's pair vs pair — a
# feature only earns a COLLECTIVE guard threshold if it separates wins from
# losses the same way in every pair, not just on average.
stability_label = "train vs WFV — holdout untouched" if MODE == 'single_pair' else "pair vs pair"
print(f"\n── Stability Across Splits ({stability_label}) ────────────")
print("  Per-split separation for the top pooled features. 'n/a' = too few\n"
      "  trades in that split/direction (<3) to compute a median split.\n")

top_feats = df_summary.head(8)[['feature', 'direction']].drop_duplicates()
if MODE == 'single_pair':
    split_order = [s for s in ['train', 'wfv'] if s in trades['split'].unique()]
else:
    split_order = sorted(trades['split'].unique())

for _, row in top_feats.iterrows():
    col, direction = row['feature'], row['direction']
    per_split = []
    for split_name in split_order:
        sub = trades[trades['split'] == split_name]
        w = sub[(sub['pnl'] > 0) & (sub['direction'] == direction)][col].dropna()
        l = sub[(sub['pnl'] < 0) & (sub['direction'] == direction)][col].dropna()
        if len(w) < 3 or len(l) < 3:
            per_split.append(f"{split_name}=n/a")
            continue
        pooled_std = pd.concat([w, l]).std()
        sep = (w.median() - l.median()) / pooled_std if pooled_std > 0 else 0.0
        if col in FLIP_COLS:
            sep = -sep
        per_split.append(f"{split_name}={sep:+.2f}")
    print(f"  {col:<14} {direction:<6}  " + "  ".join(per_split))

# ── Stop-loss profile ─────────────────────────────────────────────────────────
# Restricted to DIAG_COLS — agreement/regime_score/vol_z/vol_ratio are entry
# gate inputs (see note above). Split by direction too: longs sit on the
# negative side of z_fast_slope/lr_mom/etc. and shorts on the positive side
# by construction, so pooling both would make a "difference" that's really
# just a long/short mix-shift among stops, not a per-direction feature effect.
print("\n── Stop-Loss Entry Profile (by direction) ────────────────────────────────")
stops = trades[trades['exit_reason'] == 'stop']
other = trades[trades['exit_reason'] != 'stop']
avail = [c for c in DIAG_COLS if c in trades.columns]

print(f"  Stop trades : {len(stops)} / {len(trades)}  ({len(stops)/len(trades)*100:.1f}%)\n")

# Sanity check: exit_stop_long/short fires on z_slow vs a ROLLING mean, not
# vs entry price — so the spread can recover back above entry_lr (a winner)
# while z_slow still reads as "extreme" if the rolling mean drifted during
# the hold. So exit_reason=='stop' is not a clean loss label; flag it loudly
# if a direction's stop count exceeds its loss count (mathematically only
# possible if some stop-tagged trades closed profitably).
for direction in ['long', 'short']:
    n_stop_dir = (stops['direction'] == direction).sum()
    n_loss_dir = (losses['direction'] == direction).sum()
    stop_wins  = stops[(stops['direction'] == direction) & (stops['pnl'] > 0)]
    if len(stop_wins) > 0 or n_stop_dir > n_loss_dir:
        print(f"  [{direction}] WARNING: {len(stop_wins)}/{n_stop_dir} stop-tagged trades "
              f"were actually profitable (n_stop={n_stop_dir} vs n_loss={n_loss_dir}).\n"
              f"            'stop' fires on z_slow vs a rolling mean, not entry price — "
              f"it isn't a clean loss label. Treat stop-profile comparisons below with that "
              f"in mind, especially when n_stop is close to or above n_loss.\n")

for direction in ['long', 'short']:
    s = stops[stops['direction'] == direction]
    o = other[other['direction'] == direction]
    if len(s) < 3:
        print(f"  [{direction}] only {len(s)} stop trades — skipping (too few to read).\n")
        continue
    comp = pd.DataFrame({
        'stopped_median': s[avail].median().round(4),
        'other_median':   o[avail].median().round(4),
        'diff':          (s[avail].median() - o[avail].median()).round(4),
    })
    print(f"  [{direction}]  n_stop={len(s)}  n_other={len(o)}")
    print(comp.to_string())
    print()
