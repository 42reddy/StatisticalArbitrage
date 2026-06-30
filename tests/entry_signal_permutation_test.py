"""
Entry-signal permutation test (Monte Carlo significance check).

Every test so far has asked "is performance stable" (random windows,
param sensitivity). Neither asks the more basic question: does the
*timing* of entries actually carry information, or would any set of
entries with the same frequency, fed through the same exit/stop/max-hold
risk management, produce similar P&L? If random entries do just as well,
the apparent edge isn't coming from the z-score crossing logic at all —
it's coming from generic structural properties of the spread (cost
structure, exit rules, a long-run drift) that have nothing to do with the
signal being "smart" (classic Aronson / White's Reality Check style
randomization test).

Method: take the real long_entry / short_entry boolean columns produced
by generate_signals() and randomly permute them in time (same exact
number of long-entry and short-entry trigger bars, just shuffled to
random dates). Exit logic, stops, time-stops and the underlying feature
columns are left untouched — they don't depend on entries, only on
feat — so this isolates the value of entry *timing* specifically. Repeat
many times, then see where the real strategy's Sharpe/Calmar/return sits
relative to the resulting null distribution.
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

T1, T2   = STATIC['T1'], STATIC['T2']
CAPITAL  = 200_000
N_TRIALS = 500
SEED     = 42

rng = np.random.default_rng(SEED)

data_loader_obj = data_loader()
feature_builder = features()
metrics_calc    = metrics()
backtest_engine = backtest(leverage=4)

df = data_loader_obj.load_data(needs_fx=False)
print(f"Full dataset : {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)}d)")

beta    = feature_builder.estimate_hedge_ratio(df, T1, T2, lookback=None)
spread  = np.log(df[T1]) - beta * np.log(df[T2])
ou_mean = float(spread.mean())

feat = feature_builder.build_features(df, STATIC, ou_mean=ou_mean, beta=beta)[0]
sig_real = feature_builder.generate_signals(feat, STATIC)

pnl_real, eq_real, trades_real = backtest_engine.backtest(feat, sig_real, STATIC, cost=True)
met_real = metrics_calc.calc_metrics(pnl_real, eq_real, trades_real)

n_long_triggers  = int(sig_real['long_entry'].sum())
n_short_triggers = int(sig_real['short_entry'].sum())
print(f"\nREAL STRATEGY   Sharpe={met_real['sharpe']:.2f}  Calmar={met_real['calmar']:.2f}  "
      f"AnnRet={met_real['ann_ret']*100:.2f}%  trades={met_real['n_trades']}")
print(f"Signal trigger bars  long={n_long_triggers}  short={n_short_triggers}  "
      f"(of {len(sig_real)} total bars)")

long_vals  = sig_real['long_entry'].values
short_vals = sig_real['short_entry'].values

records = []
for trial in range(N_TRIALS):
    sig_rand = sig_real.copy()
    sig_rand['long_entry']  = rng.permutation(long_vals)
    sig_rand['short_entry'] = rng.permutation(short_vals)

    pnl_r, eq_r, trades_r = backtest_engine.backtest(feat, sig_rand, STATIC, cost=True)
    if len(trades_r) == 0 or pnl_r.std() == 0:
        continue
    met_r = metrics_calc.calc_metrics(pnl_r, eq_r, trades_r)
    records.append(dict(sharpe=met_r['sharpe'], calmar=met_r['calmar'],
                         ann_ret=met_r['ann_ret'], max_dd=met_r['max_dd'],
                         n_trades=met_r['n_trades']))

null_df = pd.DataFrame(records)
null_df.to_csv(os.path.join(RESULTS_TABLES, 'entry_signal_permutation_null.csv'), index=False)

print(f"\n{'='*72}")
print(f"NULL DISTRIBUTION  (n={len(null_df)} valid permutations, random entry timing,")
print("same trigger counts, same exits/stops/max-hold as the real strategy)")
print(f"{'='*72}")
print(f"  Sharpe  : mean={null_df['sharpe'].mean():.2f}  median={null_df['sharpe'].median():.2f}  "
      f"std={null_df['sharpe'].std():.2f}  "
      f"[{null_df['sharpe'].quantile(.05):.2f}, {null_df['sharpe'].quantile(.95):.2f}]")
print(f"  Calmar  : mean={null_df['calmar'].mean():.2f}  median={null_df['calmar'].median():.2f}")
print(f"  AnnRet% : mean={null_df['ann_ret'].mean()*100:.2f}  median={null_df['ann_ret'].median()*100:.2f}")

p_sharpe  = (null_df['sharpe']  >= met_real['sharpe']).mean()
p_calmar  = (null_df['calmar']  >= met_real['calmar']).mean()
p_annret  = (null_df['ann_ret'] >= met_real['ann_ret']).mean()

print(f"\n  p-value (Sharpe)   : {p_sharpe:.3f}   "
      f"(fraction of random-entry trials beating the real Sharpe)")
print(f"  p-value (Calmar)   : {p_calmar:.3f}")
print(f"  p-value (AnnRet)   : {p_annret:.3f}")

print(f"\n{'='*72}")
if p_sharpe < 0.05:
    print("RESULT: real entry timing clears random-entry noise (p<0.05) —")
    print("the z-score crossing logic appears to add genuine information,")
    print("not just the general structure of the spread/cost/exit rules.")
elif p_sharpe < 0.20:
    print("RESULT: real entry timing beats most random-entry trials but not")
    print("decisively (0.05<=p<0.20) — some signal value, but weak relative")
    print("to noise given the available trade count.")
else:
    print("WARNING: random entry timing with the same trigger frequency does")
    print("about as well as the real signal (p>=0.20) — most of the apparent")
    print("edge may come from exits/stops/max-hold/cost structure and general")
    print("spread drift, not from the entry signal itself.")
print(f"{'='*72}")
