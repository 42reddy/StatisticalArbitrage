import numpy as np
import pandas as pd
from data import data_loader
from features import features
from diagnosis import diagnosis
from backtest import backtest
from metrics import metrics
from plotting import plotting
from params import PARAMS
import matplotlib.pyplot as plt

T1       = PARAMS['T1']
T2       = PARAMS["T2"]
CAPITAL = 200_000   # must match backtest.TRADE_CAPITAL

data_loader_obj = data_loader()
feature_builder = features()
metrics_calc    = metrics()
diagnosis_obj   = diagnosis()
backtest_engine = backtest(leverage=3)
plotter         = plotting()

# ─────────────────────────────────────────────
#  LOAD & SPLIT
# ─────────────────────────────────────────────
df           = data_loader_obj.load_data(needs_fx=False)
df.tail()


HOLDOUT_DAYS = 504
df_train     = df.iloc[:-HOLDOUT_DAYS].copy()
df_holdout   = df.iloc[-HOLDOUT_DAYS:].copy()

# ── Hedge ratio estimated ONCE on the full training set ──
# This is the structural cointegrating coefficient.
# Do NOT re-estimate on holdout or per-fold — that leaks
# information and destroys the statistical basis.
beta_train = feature_builder.estimate_hedge_ratio(
    df_train, T1, T2, lookback=None)

# OU equilibrium on beta-adjusted spread
spread_train = np.log(df_train[T1]) - beta_train * np.log(df_train[T2])
plt.plot(spread_train)
ou_mean      = float(spread_train.iloc[:504].mean())

print(f"  Full    : {df.index[0].date()} → {df.index[-1].date()}  ({len(df)}d)")
print(f"  Train   : {df_train.index[0].date()} → {df_train.index[-1].date()}  ({len(df_train)}d)")
print(f"  Holdout : {df_holdout.index[0].date()} → {df_holdout.index[-1].date()}  ({len(df_holdout)}d)")
print(f"  Beta    : {beta_train:.4f}  (hedge ratio, OLS on train)")
print(f"  OU mean : {ou_mean:.5f}  (β-adjusted spread equilibrium)")

# ─────────────────────────────────────────────
#  HELPER — print a concise performance block
# ─────────────────────────────────────────────
def print_perf(label, met, n_days, trades_df=None):
    C = CAPITAL
    ann_ret_dol  = met['ann_ret']   * C
    max_dd_dol   = met['max_dd']    * C
    ann_vol_dol  = met['ann_vol']   * C
    avg_win_dol  = met['avg_win']   * C
    avg_loss_dol = met['avg_loss']  * C
    n            = met['n_trades']
    trades_yr    = n / (n_days / 252) if n_days > 0 else 0

    util = np.nan
    if trades_df is not None and len(trades_df) > 0:
        deployed = trades_df['hold_days'].sum()
        util     = deployed / n_days * 100

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
║    % pyramided        :  {met['pct_pyramided']:>9.1f}%
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
            wins   = grp.loc[grp['pnl']>0,'pnl'].mean()*C if (grp['pnl']>0).any() else 0
            loss   = grp.loc[grp['pnl']<0,'pnl'].mean()*C if (grp['pnl']<0).any() else 0
            pay    = abs(wins/loss) if loss != 0 else np.nan
            print(f"  {d:>6}  {len(grp):>4}  {wr_d:>5.1f}%  {avg_d:>9,.2f}  "
                  f"{tot_d:>11,.2f}  {hold_d:>8.1f}d  {pay:>7.2f}")

# ─────────────────────────────────────────────
#  OU HALF-LIFE  (on β-adjusted spread)
# ─────────────────────────────────────────────
y      = spread_train.replace([np.inf, -np.inf], np.nan).dropna().values
dy     = np.diff(y)
y_lag  = y[:-1]

from scipy import stats as scipy_stats
slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(y_lag, dy)

lambda_ou = -slope
ou_hl     = np.log(2) / lambda_ou if lambda_ou > 0 else 100

print(f"  OU HL   : {ou_hl:.2f} days  (β-adjusted spread)")

# ─────────────────────────────────────────────
#  BAYESIAN OPTIMISATION  (train only)
# ─────────────────────────────────────────────
print("\n── Bayesian Optimisation ──")
# diagnosis internally estimates beta but we also store it in PARAMS
best_params = diagnosis_obj.sensitivity_analysis(df_train, ou_hl, n_trials=120, n_seeds=15)

# Sync beta into backtest engine (in case it was updated)
PARAMS.update(best_params)
PARAMS['z_entry'] = (PARAMS['z_entry_long'] + PARAMS['z_entry_short']) / 2
PARAMS['z_exit']  = (PARAMS['z_exit_long']  + PARAMS['z_exit_short'])  / 2
PARAMS['z_stop']  = (PARAMS['z_stop_long']  + PARAMS['z_stop_short'])  / 2
PARAMS['beta']    = best_params.get('beta', beta_train)

# ─────────────────────────────────────────────
#  WALK-FORWARD VALIDATION  (train, fixed params)
# ─────────────────────────────────────────────
TRAIN_DAYS = 504
TEST_DAYS  = 126
STEP_DAYS  = 63

wf_pnl_all, wf_equity_all, wf_trades_all = [], [], []
fold_log = []

n_folds = (len(df_train) - TRAIN_DAYS - TEST_DAYS) // STEP_DAYS + 1
print(f"\n── Walk-Forward Validation ({n_folds} folds) ──")
print(f"  {'Fold':>4}  {'Test Period':>23}  {'n':>4}  "
      f"{'Sharpe':>7}  {'Ret':>9}  {'DD':>9}  {'Calmar':>7}  {'WR':>6}  {'Util':>6}")
print("  " + "─" * 85)

for fold in range(n_folds):
    ts = fold * STEP_DAYS
    te = ts + TRAIN_DAYS
    xe = te + TEST_DAYS
    if xe > len(df_train):
        break

    df_ftr = df_train.iloc[ts:te].copy()
    df_fts = df_train.iloc[te:xe].copy()

    # OU mean computed on beta-adjusted train portion of THIS fold
    ou_f = float(
        (np.log(df_ftr[T1]) - beta_train * np.log(df_ftr[T2])).mean())

    # Pass fixed beta — no re-estimation per fold
    feat_f, _ = feature_builder.build_features(
        df_fts, PARAMS, ou_mean=ou_f, beta=beta_train)
    sig_f     = feature_builder.generate_signals(feat_f, PARAMS)
    pnl_f, eq_f, tr_f = backtest_engine.backtest(feat_f, sig_f, PARAMS, cost=True)

    n_tr    = len(tr_f)
    sharpe  = pnl_f.mean() / pnl_f.std() * np.sqrt(252) if pnl_f.std() > 0 else 0
    ann_ret = pnl_f.mean() * 252
    roll_max = eq_f.cummax()
    max_dd  = float(((eq_f - roll_max) / roll_max.replace(0, np.nan)).min())
    calmar  = ann_ret / abs(max_dd) if max_dd < -1e-6 else np.nan
    wr      = float((tr_f['pnl'] > 0).mean()) if n_tr > 0 else np.nan
    util    = tr_f['hold_days'].sum() / TEST_DAYS * 100 if n_tr > 0 else 0

    wf_pnl_all.append(pnl_f)
    wf_equity_all.append(eq_f)
    wf_trades_all.append(tr_f)
    fold_log.append(dict(
        fold=fold+1,
        test_start=df_train.index[te],
        test_end=df_train.index[xe-1],
        n_trades=n_tr, sharpe=sharpe, ann_ret=ann_ret,
        max_dd=max_dd, calmar=calmar, win_rate=wr, util=util))

    calmar_str = f"{calmar:>7.2f}" if not np.isnan(calmar) else "    nan"
    print(f"  {fold+1:>4}  "
          f"{str(df_train.index[te].date())}→{str(df_train.index[xe-1].date())}  "
          f"{n_tr:>4}  {sharpe:>7.2f}  "
          f"{ann_ret*CAPITAL:>8,.0f}  {max_dd*CAPITAL:>8,.0f}  "
          f"{calmar_str}  {wr*100:>5.1f}%  {util:>5.1f}%")

# ── Aggregate ──────────────────────────────────────────
wf_pnl    = pd.concat(wf_pnl_all)
wf_equity = pd.concat(wf_equity_all)
wf_trades = pd.concat(wf_trades_all, ignore_index=True)
fold_df   = pd.DataFrame(fold_log)

wfv_sharpe  = wf_pnl.mean() / wf_pnl.std() * np.sqrt(252) if wf_pnl.std() > 0 else 0
wfv_ann_ret = wf_pnl.mean() * 252
wfv_ann_vol = wf_pnl.std()  * np.sqrt(252)
wfv_roll    = wf_equity.cummax()
wfv_dd      = float(((wf_equity - wfv_roll) / wfv_roll.replace(0, np.nan)).min())
wfv_calmar  = wfv_ann_ret / abs(wfv_dd) if wfv_dd < -1e-6 else np.nan
wfv_wr      = float((wf_trades['pnl'] > 0).mean()) if len(wf_trades) > 0 else np.nan
wfv_util    = fold_df['util'].mean()
wfv_tpy     = fold_df['n_trades'].mean() / (TEST_DAYS / 252)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  WALK-FORWARD SUMMARY  (OOS, fixed Bayes params)
╠══════════════════════════════════════════════════════════════════╣
║  CONSISTENCY
║    Folds profitable        :  {(fold_df['ann_ret']>0).mean()*100:.0f}%
║    Folds Calmar > 1        :  {(fold_df['calmar']>1).mean()*100:.0f}%
║    Folds Sharpe > 0.5      :  {(fold_df['sharpe']>0.5).mean()*100:.0f}%
║    Avg trades / fold       :  {fold_df['n_trades'].mean():.1f}  (~{wfv_tpy:.0f}/yr)
║    Avg capital utilisation :  {wfv_util:.1f}%
╠══════════════════════════════════════════════════════════════════╣
║  AGGREGATE OOS
║    Ann. return ()         :  {wfv_ann_ret*CAPITAL:>10,.0f}  ({wfv_ann_ret*100:.2f}%)
║    Ann. vol ()            :  {wfv_ann_vol*CAPITAL:>10,.0f}
║    Max drawdown ()        :  {wfv_dd*CAPITAL:>10,.0f}  ({wfv_dd*100:.2f}%)
║    Calmar                  :  {wfv_calmar:>10.2f}
║    Sharpe                  :  {wfv_sharpe:>10.2f}
║    Win rate                :  {wfv_wr*100:>9.1f}%
║    Total OOS trades        :  {len(wf_trades):>10}
╚══════════════════════════════════════════════════════════════════╝""")

# ─────────────────────────────────────────────
#  TRAIN FULL RUN
# ─────────────────────────────────────────────
feat_tr, ou_tr       = feature_builder.build_features(
    df_train, PARAMS, ou_mean=ou_mean, beta=beta_train)
sig_tr               = feature_builder.generate_signals(feat_tr, PARAMS)
pnl_tr, eq_tr, trades_tr = backtest_engine.backtest(feat_tr, sig_tr, PARAMS, cost=True)
stat_tr              = diagnosis_obj.run_stat_diag(df_train, feat_tr)
met_tr               = metrics_calc.calc_metrics(pnl_tr, eq_tr, trades_tr)

train_sharpe = float(met_tr.get('sharpe', 0))
train_calmar = float(met_tr.get('calmar', 0))
sharpe_deg   = ((train_sharpe - wfv_sharpe) / abs(train_sharpe) * 100
                if train_sharpe != 0 else 0)
calmar_deg   = ((train_calmar - wfv_calmar) / abs(train_calmar) * 100
                if train_calmar != 0 else 0)

verdict = ("✓ ROBUST"           if sharpe_deg < 20 else
           "⚠ MODERATE overfit" if sharpe_deg < 40 else
           "✗ OVERFIT — fix before holdout")

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ROBUSTNESS GATE
║    Train  Sharpe / Calmar  :  {train_sharpe:.2f} / {train_calmar:.2f}
║    WFV    Sharpe / Calmar  :  {wfv_sharpe:.2f} / {wfv_calmar:.2f}
║    Sharpe degradation      :  {sharpe_deg:.1f}%
║    Calmar degradation      :  {calmar_deg:.1f}%
║    Verdict                 :  {verdict}
╚══════════════════════════════════════════════════════════════════╝""")

plotter.print_report(met_tr, stat_tr, trades_tr, PARAMS, ou_tr,
                     beta=beta_train, label="TRAIN")
print_perf("TRAIN PERFORMANCE", met_tr, len(df_train), trades_tr)

# ─────────────────────────────────────────────
#  HOLDOUT
# ─────────────────────────────────────────────
if sharpe_deg < 40:
    # OU mean from last 504 days of training spread (not raw ratio)
    ou_holdout = float(spread_train.iloc[-504:].mean())

    # Use training beta on holdout — no re-estimation
    feat_h, _              = feature_builder.build_features(
        df_holdout, PARAMS, ou_mean=ou_holdout, beta=beta_train)
    sig_h                  = feature_builder.generate_signals(feat_h, PARAMS)
    pnl_h, eq_h, trades_h = backtest_engine.backtest(
        feat_h, sig_h, PARAMS, cost=True)
    stat_h                 = diagnosis_obj.run_stat_diag(df_holdout, feat_h)
    met_h                  = metrics_calc.calc_metrics(pnl_h, eq_h, trades_h)

    holdout_sharpe = float(met_h.get('sharpe', 0))
    holdout_calmar = float(met_h.get('calmar', 0))

    final_verdict = ("✓ EDGE CONFIRMED" if holdout_sharpe > 0.7 else
                     "~ MARGINAL EDGE"  if holdout_sharpe > 0.3 else
                     "✗ NO EDGE")

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  HOLDOUT
║    Ann. return ()         :  {met_h['ann_ret']*CAPITAL:>10,.0f}
║    Max drawdown ()        :  {met_h['max_dd']*CAPITAL:>10,.0f}
║    Calmar  (holdout/WFV)   :  {holdout_calmar:.2f} / {wfv_calmar:.2f}
║    Sharpe  (holdout/WFV)   :  {holdout_sharpe:.2f} / {wfv_sharpe:.2f}
║    Train→Holdout Sharpe Δ  :  {(train_sharpe-holdout_sharpe)/abs(train_sharpe)*100:.1f}%
║    Beta used               :  {beta_train:.4f}  (trained, not re-estimated)
║    Verdict                 :  {final_verdict}
╚══════════════════════════════════════════════════════════════════╝""")

    plotter.print_report(met_h, stat_h, trades_h, PARAMS, ou_holdout,
                         beta=beta_train, label="HOLDOUT")
    print_perf("HOLDOUT PERFORMANCE", met_h, len(df_holdout), trades_h)
    # plotter.print_trade_audit(trades_h, df_holdout)

    plotter.plot_all(df_holdout, feat_h, sig_h, pnl_h, eq_h, trades_h,
                     stat_h, met_h, None, PARAMS,
                     wf_pnl=wf_pnl, wf_equity=wf_equity, wf_params_df=fold_df,
                     label="HOLDOUT")
else:
    print("\n  ⚠ Holdout skipped — reduce overfitting first.")
    plotter.plot_all(df_train, feat_tr, sig_tr, pnl_tr, eq_tr, trades_tr,
                     stat_tr, met_tr, None, PARAMS,
                     wf_pnl=wf_pnl, wf_equity=wf_equity, wf_params_df=fold_df,
                     label="TRAIN")

