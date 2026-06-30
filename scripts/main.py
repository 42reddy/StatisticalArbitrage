import io
import contextlib
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data import data_loader
from features import features
from diagnosis import diagnosis
from backtest import backtest
from metrics import metrics
from plotting import plotting
from params import PARAMS

RESULTS_TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'tables')

T1       = PARAMS['T1']
T2       = PARAMS["T2"]
CAPITAL = 200_000   # must match backtest.TRADE_CAPITAL

# ─────────────────────────────────────────────
#  RUN SWITCH
# ─────────────────────────────────────────────
# True  → run the Bayesian sensitivity search on train and use the
#         consensus params it discovers.
# False → skip the search entirely and use the params already defined
#         in params.py (PARAMS). All constants needed downstream are
#         present in params.py, so the WFV / train / holdout pipeline
#         runs unchanged.
RUN_BAYES = False

# True  → replace the single frozen OLS beta with a causally-updated
#         Kalman-filter beta (see features.kalman_hedge_ratio and
#         rolling_beta_stability.py). Validated there: ~96% lower
#         window-to-window variance than rolling OLS re-estimation,
#         AND a more stationary spread (lower ADF p-value) than the
#         static full-sample OLS beta over the same window — so this
#         is tracking the real cointegrating relationship more
#         precisely, not just smoothing over real drift.
# False → original behaviour: one OLS beta fit on df_train, frozen and
#         reused everywhere (no re-estimation on holdout/per-fold).
USE_KALMAN_BETA    = True
KALMAN_DELTA       = 1e-6
KALMAN_INIT_WINDOW = 1000   # bars of warm-up before beta is valid —
                            # shorter windows don't identify the
                            # cointegrating vector for this pair

data_loader_obj = data_loader()
feature_builder = features()
metrics_calc    = metrics()
diagnosis_obj   = diagnosis()
backtest_engine = backtest(leverage=4)
plotter         = plotting()

# ─────────────────────────────────────────────
#  LOAD & SPLIT
# ─────────────────────────────────────────────
df           = data_loader_obj.load_data(needs_fx=False)
df.tail()


HOLDOUT_DAYS = 756
df_train     = df.iloc[:-HOLDOUT_DAYS].copy()
df_holdout   = df.iloc[-HOLDOUT_DAYS:].copy()

if USE_KALMAN_BETA:
    # Strictly causal: beta_t only ever uses data up to bar t, so it's
    # safe to compute over the FULL series (train+holdout) once and
    # slice per fold below — exactly like any other rolling feature
    # with a warm-up. This is NOT the same as re-estimating on holdout
    # data; nothing here ever looks ahead.
    beta_full  = feature_builder.kalman_hedge_ratio(
        df, T1, T2, delta=KALMAN_DELTA, init_window=KALMAN_INIT_WINDOW)
    beta_train = beta_full.loc[df_train.index]          # pd.Series, NaN for warm-up
    beta_holdout_series = beta_full.loc[df_holdout.index]
    beta_train_scalar = float(beta_train.dropna().iloc[-1])  # for printing/PARAMS only
else:
    # ── Hedge ratio estimated ONCE on the full training set ──
    # This is the structural cointegrating coefficient.
    # Do NOT re-estimate on holdout or per-fold — that leaks
    # information and destroys the statistical basis.
    beta_train = feature_builder.estimate_hedge_ratio(df_train, T1, T2, lookback=None)
    beta_holdout_series = beta_train
    beta_train_scalar = beta_train

# OU equilibrium on beta-adjusted spread
spread_train = np.log(df_train[T1]) - beta_train * np.log(df_train[T2])
ou_mean      = float(spread_train.dropna().iloc[:504].mean())

print(f"  Full    : {df.index[0].date()} → {df.index[-1].date()}  ({len(df)}d)")
print(f"  Train   : {df_train.index[0].date()} → {df_train.index[-1].date()}  ({len(df_train)}d)")
print(f"  Holdout : {df_holdout.index[0].date()} → {df_holdout.index[-1].date()}  ({len(df_holdout)}d)")
print(f"  Beta    : {beta_train_scalar:.4f}  "
      f"({'Kalman, last train value' if USE_KALMAN_BETA else 'hedge ratio, OLS on train'})")
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
# diagnosis internally estimates beta but we also store it in PARAMS
# With Kalman beta, the first fold's train window must extend past the
# filter's warm-up (KALMAN_INIT_WINDOW) or it would start with no valid
# beta at all.
TRAIN_DAYS = max(504, KALMAN_INIT_WINDOW) if USE_KALMAN_BETA else 504
TEST_DAYS  = 126    # non-overlapping WFV OOS window length


def beta_at(idx):
    """beta_train sliced/aligned to `idx` if it's a time-varying Series,
    or the scalar itself if static OLS. Use this everywhere beta needs
    to be combined with a price slice — naive Series*Series multiplication
    with mismatched indexes silently unions instead of aligning."""
    return beta_train.loc[idx] if isinstance(beta_train, pd.Series) else beta_train

# Refit Bayes per WFV fold (anchored train slice only) instead of reusing
# one global consensus fit on the entire df_train — see WFV section below
# for why reusing a single global fit makes WFV converge to a re-slicing
# of the same train backtest. Trial budget is cut ~7x vs. the global
# search (300→40) since this runs once per fold (18 folds here); each
# fold's objective already does its own internal nested CV (diagnosis.py
# _build_folds), so 40 trials is still a real search, just a cheaper one.
WFV_REFIT_BAYES  = False
WFV_REFIT_TRIALS = 120
WFV_REFIT_SEEDS  = 1
WFV_REFIT_TOPK   = 3

if RUN_BAYES:
    print("\n── Bayesian Optimisation ──")
    consensus_params, stability = diagnosis_obj.sensitivity_analysis(
        df_train,
        ou_hl=ou_hl,
        n_trials=300,
        n_seeds=1,
        top_k=5,      # optional: holdout for final check
        verbose=True,
    )
    # Sync the discovered params in
    PARAMS.update(consensus_params)
else:
    print("\n── Skipping Bayesian Optimisation — using params.py ──")
    consensus_params = {}

# Derive the symmetric aggregate thresholds + beta from whatever params
# are now active (Bayes consensus or params.py defaults).
PARAMS['z_entry'] = (PARAMS['z_entry_long'] + PARAMS['z_entry_short']) / 2
PARAMS['z_exit']  = (PARAMS['z_exit_long']  + PARAMS['z_exit_short'])  / 2
PARAMS['z_stop']  = (PARAMS['z_stop_long']  + PARAMS['z_stop_short'])  / 2
PARAMS['beta']    = consensus_params.get('beta', beta_train_scalar)

# ─────────────────────────────────────────────
#  WALK-FORWARD VALIDATION  (anchored train, non-overlapping OOS)
# ─────────────────────────────────────────────
# Industry-standard anchored walk-forward (Pardo; purging per Lopez de Prado):
#   - Train window is anchored at bar 0 and expands each fold — never
#     "forgets" earlier data, consistent with the inner CV in diagnosis.py.
#   - Test windows are contiguous and non-overlapping (step == test length),
#     so concatenating them into one OOS series never double-counts a day
#     or a trade. The previous version stepped by 63d with a 126d test
#     window, i.e. a 63d overlap between consecutive folds.
#   - A warm-up buffer of real, already-seen historical bars is fed into
#     the rolling feature calculations so the start of every fold isn't
#     scored on a half-built rolling window (no leakage: warm-up bars are
#     all ≤ train end, just used to seed the rolling stats before scoring
#     begins exactly at the fold's true test start).

WARMUP_DAYS = max(3 * PARAMS['medium_window'],
                  int(PARAMS.get('vol_z_window', PARAMS['vol_window'] * 2)) * 2,
                  60)

wf_pnl_all, wf_trades_all = [], []
fold_log = []

n_folds = (len(df_train) - TRAIN_DAYS) // TEST_DAYS
refit_note = (f", Bayes refit per fold ({WFV_REFIT_TRIALS} trials)"
              if WFV_REFIT_BAYES else ", fixed params")
print(f"\n── Walk-Forward Validation ({n_folds} folds, anchored train, "
      f"non-overlapping OOS, {WARMUP_DAYS}d warm-up{refit_note}) ──")
print(f"  {'Fold':>4}  {'Test Period':>23}  {'n':>4}  "
      f"{'Sharpe':>7}  {'Ret':>9}  {'DD':>9}  {'Calmar':>7}  {'WR':>6}  {'Util':>6}")
print("  " + "─" * 85)

for fold in range(n_folds):
    te = TRAIN_DAYS + fold * TEST_DAYS    # anchored: train is always [0, te)
    xe = te + TEST_DAYS
    if xe > len(df_train):
        break

    df_ftr     = df_train.iloc[:te].copy()

    # Refit Bayes on ONLY this fold's anchored train slice [0, te) — the
    # resulting params have never seen this fold's test window [te, xe),
    # unlike the single global consensus fit on the entire df_train.
    if WFV_REFIT_BAYES:
        print(f"  Fold {fold+1:>2}/{n_folds}: refitting Bayes on "
              f"{len(df_ftr)}d anchored train...", end="", flush=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fold_consensus, _ = diagnosis_obj.sensitivity_analysis(
                df_ftr, ou_hl=None,
                n_trials=WFV_REFIT_TRIALS, n_seeds=WFV_REFIT_SEEDS,
                top_k=WFV_REFIT_TOPK, verbose=False)
        fold_params = PARAMS.copy()
        fold_params.update(fold_consensus)
        fold_params['z_entry'] = (fold_params['z_entry_long'] + fold_params['z_entry_short']) / 2
        fold_params['z_exit']  = (fold_params['z_exit_long']  + fold_params['z_exit_short'])  / 2
        fold_params['z_stop']  = (fold_params['z_stop_long']  + fold_params['z_stop_short'])  / 2
        print(f"  z_entry={fold_params['z_entry']:.2f}  "
              f"z_exit={fold_params['z_exit']:.2f}  "
              f"slow_window={fold_params['slow_window']}")
    else:
        fold_params = PARAMS

    # Warm-up sized to THIS fold's own window params (they can differ
    # fold-to-fold under refit), not the global pre-loop PARAMS.
    warm_f = max(3 * fold_params['medium_window'],
                 int(fold_params.get('vol_z_window', fold_params['vol_window'] * 2)) * 2,
                 60)
    warm_s     = max(0, te - warm_f,
                      KALMAN_INIT_WINDOW if USE_KALMAN_BETA else 0)
    df_fts_buf = df_train.iloc[warm_s:xe].copy()

    # OU mean computed on beta-adjusted train portion of THIS fold
    ou_f = float(
        (np.log(df_ftr[T1]) - beta_at(df_ftr.index) * np.log(df_ftr[T2])).dropna().mean())

    # Static OLS: same frozen beta in every fold (no re-estimation).
    # Kalman: causal beta sliced to this fold's bars — no leakage,
    # since beta_t here never used data beyond bar t in the first place.
    feat_f_buf, _ = feature_builder.build_features(
        df_fts_buf, fold_params, ou_mean=ou_f, beta=beta_at(df_fts_buf.index))
    feat_f    = feat_f_buf.loc[df_train.index[te]:]   # drop warm-up rows
    sig_f     = feature_builder.generate_signals(feat_f, fold_params)
    pnl_f, eq_f, tr_f = backtest_engine.backtest(feat_f, sig_f, fold_params, cost=True)

    n_tr    = len(tr_f)
    sharpe  = pnl_f.mean() / pnl_f.std() * np.sqrt(252) if pnl_f.std() > 0 else 0
    ann_ret = pnl_f.mean() * 252
    roll_max = eq_f.cummax()
    max_dd  = float(((eq_f - roll_max) / roll_max.replace(0, np.nan)).min())
    calmar  = ann_ret / abs(max_dd) if max_dd < -1e-6 else np.nan
    wr      = float((tr_f['pnl'] > 0).mean()) if n_tr > 0 else np.nan
    util    = tr_f['hold_days'].sum() / TEST_DAYS * 100 if n_tr > 0 else 0

    wf_pnl_all.append(pnl_f)
    wf_trades_all.append(tr_f)
    fold_log.append(dict(
        fold=fold+1,
        test_start=df_train.index[te],
        test_end=df_train.index[xe-1],
        n_trades=n_tr, sharpe=sharpe, ann_ret=ann_ret,
        max_dd=max_dd, calmar=calmar, win_rate=wr, util=util,
        z_entry=fold_params['z_entry'], z_exit=fold_params['z_exit'],
        z_stop=fold_params['z_stop'], slow_window=fold_params['slow_window']))

    calmar_str = f"{calmar:>7.2f}" if not np.isnan(calmar) else "    nan"
    print(f"  {fold+1:>4}  "
          f"{str(df_train.index[te].date())}→{str(df_train.index[xe-1].date())}  "
          f"{n_tr:>4}  {sharpe:>7.2f}  "
          f"{ann_ret*CAPITAL:>8,.0f}  {max_dd*CAPITAL:>8,.0f}  "
          f"{calmar_str}  {wr*100:>5.1f}%  {util:>5.1f}%")

# ── Aggregate ──────────────────────────────────────────
# Folds are contiguous and non-overlapping, so pnl_f series concatenate
# cleanly with no double-counted days. Equity is rebuilt as one continuous
# curve from the concatenated daily pnl (each fold's raw equity_s otherwise
# resets to starting capital, which would fabricate a drawdown at every
# fold boundary if concatenated directly).
wf_pnl    = pd.concat(wf_pnl_all)
wf_equity = CAPITAL + (wf_pnl * CAPITAL).cumsum()
wf_trades = pd.concat(wf_trades_all, ignore_index=True)
fold_df   = pd.DataFrame(fold_log)

# Genuinely out-of-sample trades (each fold scored on data the Bayes
# search never saw) — the largest, least-overfit sample for the
# feature-discrimination analysis in scratch2.py.
wf_trades.to_csv(os.path.join(RESULTS_TABLES, 'trades_wfv.csv'), index=False)

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
║  WALK-FORWARD SUMMARY  (OOS, {'per-fold Bayes refit' if WFV_REFIT_BAYES else 'fixed params'})
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
# With Kalman beta the first KALMAN_INIT_WINDOW bars have no valid
# beta (NaN), which would otherwise poison pnl/equity permanently —
# 0 (flat position) * NaN (lr move) = NaN in the backtest accumulator,
# not 0. Drop those bars before scoring, exactly like the WFV per-fold
# warm-up buffer already does.
df_train_scored = (df_train.iloc[KALMAN_INIT_WINDOW:]
                   if USE_KALMAN_BETA else df_train)

feat_tr, ou_tr       = feature_builder.build_features(
    df_train_scored, PARAMS, ou_mean=ou_mean, beta=beta_at(df_train_scored.index))
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
                     beta=beta_train_scalar, label="TRAIN")
print_perf("TRAIN PERFORMANCE", met_tr, len(df_train_scored), trades_tr)

# Per-trade diagnostic snapshot — feeds scratch2.py's feature
# discrimination analysis (which entry-time readings actually
# separate winners from losers, rather than guessing up front).
trades_tr.to_csv(os.path.join(RESULTS_TABLES, 'trades_train.csv'), index=False)

# ─────────────────────────────────────────────
#  HOLDOUT
# ─────────────────────────────────────────────
if sharpe_deg < 40:
    # OU mean from last 504 days of training spread (not raw ratio)
    ou_holdout = float(spread_train.iloc[-504:].mean())

    # Static OLS: training beta reused on holdout, no re-estimation.
    # Kalman: causal beta continues forward from train into holdout —
    # still no leakage, since each beta_t only used data up to bar t.
    feat_h, _              = feature_builder.build_features(
        df_holdout, PARAMS, ou_mean=ou_holdout, beta=beta_holdout_series)
    sig_h                  = feature_builder.generate_signals(feat_h, PARAMS)
    pnl_h, eq_h, trades_h = backtest_engine.backtest(
        feat_h, sig_h, PARAMS, cost=True)
    stat_h                 = diagnosis_obj.run_stat_diag(df_holdout, feat_h)
    met_h                  = metrics_calc.calc_metrics(pnl_h, eq_h, trades_h)

    holdout_sharpe = float(met_h.get('sharpe', 0))
    holdout_calmar = float(met_h.get('calmar', 0))
    beta_holdout_scalar = (float(beta_holdout_series.dropna().iloc[-1])
                           if isinstance(beta_holdout_series, pd.Series)
                           else beta_holdout_series)

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
║    Beta used               :  {beta_holdout_scalar:.4f}  {'(Kalman, last holdout value)' if USE_KALMAN_BETA else '(trained, not re-estimated)'}
║    Verdict                 :  {final_verdict}
╚══════════════════════════════════════════════════════════════════╝""")

    plotter.print_report(met_h, stat_h, trades_h, PARAMS, ou_holdout,
                         beta=beta_holdout_scalar, label="HOLDOUT")
    print_perf("HOLDOUT PERFORMANCE", met_h, len(df_holdout), trades_h)
    plotter.print_trade_audit(trades_h, df_holdout)

    trades_h.to_csv(os.path.join(RESULTS_TABLES, 'trades_holdout.csv'), index=False)

    plotter.plot_all(df_holdout, feat_h, sig_h, pnl_h, eq_h, trades_h,
                     stat_h, met_h, None, PARAMS,
                     wf_pnl=wf_pnl, wf_equity=wf_equity, wf_params_df=fold_df,
                     label="HOLDOUT")
else:
    print("\n  ⚠ Holdout skipped — reduce overfitting first.")
    plotter.plot_all(df_train_scored, feat_tr, sig_tr, pnl_tr, eq_tr, trades_tr,
                     stat_tr, met_tr, None, PARAMS,
                     wf_pnl=wf_pnl, wf_equity=wf_equity, wf_params_df=fold_df,
                     label="TRAIN")
