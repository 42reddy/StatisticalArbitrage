import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data import data_loader
from features import features
from diagnosis import diagnosis
from backtest import backtest
from metrics import metrics
from plotting import plotting
from params import PARAMS

# ─────────────────────────────────────────────
#  PARAMS — use params.py exactly as configured.
#  This script exists to validate those params on an out-of-sample
#  window, not to test a different hand-tuned variant. If experiment1
#  and main.py's holdout disagree, the params must be identical or the
#  comparison is meaningless.
# ─────────────────────────────────────────────
STATIC = PARAMS.copy()

T1      = STATIC['T1']
T2      = STATIC['T2']
CAPITAL = 200_000

# Match main.py's HOLDOUT_DAYS so this is the same OOS window main.py scores.
TEST_DAYS = 1000

# ─────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────
# PARAMS['start'] is pinned in params.py — both main.py and this script
# must load identical history, since beta/OU mean are estimates over
# whatever the training window is. A different start date here than in
# main.py is exactly what produced mismatched results previously.
data_loader_obj = data_loader()
feature_builder = features()
metrics_calc    = metrics()
diagnosis_obj   = diagnosis()
backtest_engine = backtest(leverage=4)
plotter         = plotting()

# ─────────────────────────────────────────────
#  LOAD & SPLIT  (train = everything before the test window)
# ─────────────────────────────────────────────
df_full = data_loader_obj.load_data(needs_fx=False)

if len(df_full) <= TEST_DAYS + 30:
    raise ValueError(
        f"Only {len(df_full)} bars loaded — need > {TEST_DAYS + 30} to "
        f"hold out {TEST_DAYS}d of test data with a usable train period. "
        f"Check PARAMS['start'].")

df_train = df_full.iloc[:-TEST_DAYS].copy()
df_test  = df_full.iloc[-TEST_DAYS:].copy()

print(f"  Full    : {df_full.index[0].date()} → {df_full.index[-1].date()}  ({len(df_full)}d)")
print(f"  Train   : {df_train.index[0].date()} → {df_train.index[-1].date()}  ({len(df_train)}d)")
print(f"  Test    : {df_test.index[0].date()} → {df_test.index[-1].date()}  ({len(df_test)}d)")

# Beta estimated ONCE on train, applied unchanged to the test window —
# no re-estimation on the data being scored.
beta = feature_builder.estimate_hedge_ratio(df_train, T1, T2, lookback=None)
STATIC['beta'] = beta

# OU equilibrium: mean of the last 504d of the TRAIN spread (matches
# main.py's ou_holdout windowing) — frozen before the test window starts.
spread_train = np.log(df_train[T1]) - beta * np.log(df_train[T2])
ou_mean      = float(spread_train.iloc[-504:].mean())

# OU half-life from the full train spread (diagnostic only, not used as
# a feature input — same as main.py).
y         = spread_train.replace([np.inf, -np.inf], np.nan).dropna().values
dy        = np.diff(y)
y_lag     = y[:-1]
slope, *_ = scipy_stats.linregress(y_lag, dy)
lambda_ou = -slope
ou_hl     = np.log(2) / lambda_ou if lambda_ou > 0 else 100

print(f"  Beta         : {beta:.4f}  (OLS on train, applied unchanged to test)")
print(f"  OU mean      : {ou_mean:.5f}  (train spread, last 504d — frozen, not test data)")
print(f"  OU half-life : {ou_hl:.1f}d  (train spread)")

# ─────────────────────────────────────────────
#  HELPER — performance print block
# ─────────────────────────────────────────────
def print_perf(label, met, n_days, trades_df=None):
    C            = CAPITAL
    ann_ret_dol  = met['ann_ret']  * C
    max_dd_dol   = met['max_dd']   * C
    ann_vol_dol  = met['ann_vol']  * C
    avg_win_dol  = met['avg_win']  * C
    avg_loss_dol = met['avg_loss'] * C
    n            = met['n_trades']
    trades_yr    = n / (n_days / 252) if n_days > 0 else 0

    util = np.nan
    if trades_df is not None and len(trades_df) > 0:
        util = trades_df['hold_days'].sum() / n_days * 100

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  {label:<63}║
╠══════════════════════════════════════════════════════════════════╣
║  RETURNS  ({C:,} capital)
║    Ann. return ()    :  {ann_ret_dol:>10,.0f}  ({met['ann_ret']*100:.2f}%)
║    Ann. vol ()       :  {ann_vol_dol:>10,.0f}  ({met['ann_vol']*100:.2f}%)
║    Max drawdown ()   :  {max_dd_dol:>10,.0f}  ({met['max_dd']*100:.2f}%)
║    Calmar ratio       :  {met['calmar']:>10.2f}   ← primary risk-adj metric
║    Sharpe ratio       :  {met['sharpe']:>10.2f}   ← secondary
║    Sortino ratio      :  {met['sortino']:>10.2f}
╠══════════════════════════════════════════════════════════════════╣
║  TRADE QUALITY
║    Total trades       :  {n:>10}
║    Trades / year      :  {trades_yr:>10.1f}
║    Win rate           :  {met['win_rate']*100:>9.1f}%
║    Avg win ()        :  {avg_win_dol:>10,.2f}
║    Avg loss ()       :  {avg_loss_dol:>10,.2f}
║    Payoff ratio       :  {abs(avg_win_dol/avg_loss_dol) if avg_loss_dol!=0 else 0:>10.2f}
║    Profit factor      :  {met['profit_factor']:>10.2f}
║    Avg hold           :  {met['avg_hold']:>9.1f}d
║    Capital util.      :  {util:>9.1f}%
╚══════════════════════════════════════════════════════════════════╝""")

    if trades_df is not None and len(trades_df) > 0 and 'direction' in trades_df.columns:
        print(f"  DIRECTION BREAKDOWN  ({C:,} capital)")
        print(f"  {'Dir':>6}  {'n':>4}  {'WR':>6}  {'Avg P&L':>10}  "
              f"{'Total P&L':>12}  {'Avg Hold':>9}  {'Payoff':>7}")
        print("  " + "─" * 65)
        for d, grp in trades_df.groupby('direction'):
            wr_d   = (grp['pnl'] > 0).mean() * 100
            avg_d  = grp['pnl'].mean() * C
            tot_d  = grp['pnl'].sum()  * C
            hold_d = grp['hold_days'].mean()
            wins   = grp.loc[grp['pnl'] > 0, 'pnl'].mean() * C if (grp['pnl'] > 0).any() else 0
            loss   = grp.loc[grp['pnl'] < 0, 'pnl'].mean() * C if (grp['pnl'] < 0).any() else 0
            pay    = abs(wins / loss) if loss != 0 else np.nan
            print(f"  {d:>6}  {len(grp):>4}  {wr_d:>5.1f}%  {avg_d:>9,.2f}  "
                  f"{tot_d:>11,.2f}  {hold_d:>8.1f}d  {pay:>7.2f}")

# ─────────────────────────────────────────────
#  BACKTEST — test window only, frozen train-derived beta/OU mean
# ─────────────────────────────────────────────
print("\n── Test-window backtest (params.py params, beta/OU frozen from train) ──")
feat, _         = feature_builder.build_features(df_test, STATIC, ou_mean=ou_mean, beta=beta)
sig             = feature_builder.generate_signals(feat, STATIC)
pnl, eq, trades = backtest_engine.backtest(feat, sig, STATIC, cost=True)
stat            = diagnosis_obj.run_stat_diag(df_test, feat)
met             = metrics_calc.calc_metrics(pnl, eq, trades)

plotter.print_report(met, stat, trades, STATIC, ou_mean, beta=beta, label="TEST (OOS)")
print_perf("TEST-WINDOW PERFORMANCE  (params.py params, no leakage)", met, len(df_test), trades)
plotter.print_trade_audit(trades, df_test)

plotter.plot_all(df_test, feat, sig, pnl, eq, trades, stat, met, None, STATIC, label="EXPERIMENT 1")
