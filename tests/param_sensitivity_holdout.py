"""
Local parameter-sensitivity check on the true holdout set (last 504 bars).

Idea: a strategy that's overfit tends to sit on a narrow, isolated peak in
parameter space — nudge any one threshold or window by a few percent and
performance collapses. A robust strategy sits on a broad "plateau" — small
perturbations barely move Sharpe/Calmar/returns (Pardo's parameter-plateau
analysis / robustness-surface check).

This script takes the chosen STATIC params one at a time, perturbs each
param independently across a small grid around its current value, and
re-backtests ONLY on the holdout (last 504 bars) — the same split main.py
uses. Beta and the OU equilibrium are estimated once on the training
portion (everything before the holdout) and held fixed for every run, so
no holdout information ever leaks into the feature/spread construction.
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data import data_loader
from features import features
from backtest import backtest
from metrics import metrics
from params import PARAMS

RESULTS_TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'tables')

STATIC = PARAMS.copy()
STATIC.update(dict(
    z_entry_long   = 1.1,
    z_entry_short  = 1.1,
    z_stop_long    = 2.9,
    z_stop_short   = 2.9,
    z_exit_long    = 0.18,
    z_exit_short   = 0.18,
    slow_window    = 22,
    medium_window  = 39,
    max_hold       = 18,
))

T1, T2       = STATIC['T1'], STATIC['T2']
CAPITAL      = 200_000
HOLDOUT_DAYS = 504   # same split as main.py

# Relative grid for continuous thresholds (z_entry/z_exit/z_stop), and
# absolute bar-count grid for integer windows.
PCT_GRID    = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
WINDOW_GRID = [-4, -2, -1, 0, 1, 2, 4]

PARAM_SPECS = {
    'z_entry_long':  ('pct', PCT_GRID),
    'z_entry_short': ('pct', PCT_GRID),
    'z_stop_long':   ('pct', PCT_GRID),
    'z_stop_short':  ('pct', PCT_GRID),
    'z_exit_long':   ('pct', PCT_GRID),
    'z_exit_short':  ('pct', PCT_GRID),
    'slow_window':   ('abs', WINDOW_GRID),
    'medium_window': ('abs', WINDOW_GRID),
    'max_hold':      ('abs', WINDOW_GRID),
}

data_loader_obj = data_loader()
feature_builder = features()
metrics_calc    = metrics()
backtest_engine = backtest(leverage=4)

df         = data_loader_obj.load_data(needs_fx=False)
df_train   = df.iloc[:-HOLDOUT_DAYS].copy()
df_holdout = df.iloc[-HOLDOUT_DAYS:].copy()

print(f"Full    : {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)}d)")
print(f"Train   : {df_train.index[0].date()} -> {df_train.index[-1].date()}  ({len(df_train)}d)")
print(f"Holdout : {df_holdout.index[0].date()} -> {df_holdout.index[-1].date()}  ({len(df_holdout)}d)")

# Beta and OU equilibrium fixed from training data only — never
# re-estimated on the holdout, exactly as in main.py.
beta_train   = feature_builder.estimate_hedge_ratio(df_train, T1, T2, lookback=None)
spread_train = np.log(df_train[T1]) - beta_train * np.log(df_train[T2])
ou_holdout   = float(spread_train.iloc[-HOLDOUT_DAYS:].mean())
print(f"Beta (train) : {beta_train:.4f}   OU mean (holdout) : {ou_holdout:.5f}")

WARMUP_DAYS = max(3 * STATIC['medium_window'],
                  int(STATIC.get('vol_z_window', STATIC['vol_window'] * 2)) * 2,
                  60)
warm_buf = df_train.tail(WARMUP_DAYS)
df_buf   = pd.concat([warm_buf, df_holdout])


def run_holdout(params):
    feat_buf, _ = feature_builder.build_features(df_buf, params, ou_mean=ou_holdout, beta=beta_train)
    feat        = feat_buf.loc[df_holdout.index[0]:]
    sig         = feature_builder.generate_signals(feat, params)
    pnl, eq, trades = backtest_engine.backtest(feat, sig, params, cost=True)
    if len(trades) == 0 or pnl.std() == 0:
        return dict(sharpe=0.0, calmar=0.0, ann_ret=0.0, max_dd=0.0, n_trades=0, win_rate=np.nan)
    return metrics_calc.calc_metrics(pnl, eq, trades)


baseline = run_holdout(STATIC)
print(f"\nBASELINE (holdout)  Sharpe={baseline['sharpe']:.2f}  Calmar={baseline['calmar']:.2f}  "
      f"AnnRet={baseline['ann_ret']*100:.2f}%  MaxDD={baseline['max_dd']*100:.2f}%  "
      f"trades={baseline['n_trades']}")

rows = []
for pname, (kind, grid) in PARAM_SPECS.items():
    base_val = STATIC[pname]
    for g in grid:
        trial = STATIC.copy()
        if kind == 'pct':
            new_val = base_val * (1 + g)
        else:
            new_val = base_val + g
            if pname in ('slow_window', 'medium_window', 'max_hold'):
                new_val = max(2, int(round(new_val)))
        trial[pname] = new_val

        met = run_holdout(trial)
        rows.append(dict(
            param=pname, perturb=g, value=new_val,
            sharpe=met['sharpe'], calmar=met['calmar'],
            ann_ret=met['ann_ret'], max_dd=met['max_dd'],
            n_trades=met['n_trades'],
            d_sharpe=met['sharpe'] - baseline['sharpe'],
            d_calmar=met['calmar'] - baseline['calmar'],
            d_ann_ret=met['ann_ret'] - baseline['ann_ret'],
        ))

res_df = pd.DataFrame(rows)
res_df.to_csv(os.path.join(RESULTS_TABLES, 'param_sensitivity_holdout.csv'), index=False)

print(f"\n{'='*88}")
print("PARAMETER SENSITIVITY ON HOLDOUT (504d)  —  Δ vs baseline")
print(f"{'='*88}")
for pname, (kind, grid) in PARAM_SPECS.items():
    sub = res_df[res_df['param'] == pname].sort_values('perturb')
    print(f"\n── {pname}  (base={STATIC[pname]}) ──")
    print(f"  {'perturb':>8}  {'value':>8}  {'Sharpe':>7}  {'ΔSharpe':>8}  "
          f"{'Calmar':>7}  {'AnnRet%':>8}  {'trades':>6}")
    for _, r in sub.iterrows():
        tag = '*' if abs(r['perturb']) < 1e-9 else ' '
        print(f" {tag}{r['perturb']:>7.2f}  {r['value']:>8.3f}  {r['sharpe']:>7.2f}  "
              f"{r['d_sharpe']:>8.2f}  {r['calmar']:>7.2f}  {r['ann_ret']*100:>7.2f}  "
              f"{int(r['n_trades']):>6}")

    sharpe_range = sub['sharpe'].max() - sub['sharpe'].min()
    flag = "  ⚠ HIGH sensitivity (narrow peak)" if sharpe_range > 1.0 else \
           "  ~ moderate sensitivity"           if sharpe_range > 0.5 else \
           "  ✓ flat / robust to this param"
    print(f"  Sharpe range across grid: {sharpe_range:.2f}{flag}")

print(f"\n{'='*88}")
print("Params with the widest Sharpe range above are where the strategy is")
print("most fragile — a real edge should show a broad plateau, not a single")
print("sharp spike at the chosen value.")
print(f"{'='*88}")
