import numpy as np
import pandas as pd
from data import data_loader
from features import features
from diagnosis import diagnosis
from backtest import backtest
from metrics import metrics
from plotting import plotting
from params import PARAMS

T1       = PARAMS['T1']
T2       = PARAMS["T2"]

data_loader_obj = data_loader()
feature_builder = features()
metrics_calc    = metrics()
diagnosis_obj   = diagnosis()
backtest_engine = backtest(leverage=5)
plotter         = plotting()

# ─────────────────────────────────────────────
#  LOAD & SPLIT
# ─────────────────────────────────────────────
df           = data_loader_obj.load_data(needs_fx=False)


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
ou_mean      = float(spread_train.iloc[:504].mean())

print(f"  Full    : {df.index[0].date()} → {df.index[-1].date()}  ({len(df)}d)")
print(f"  Train   : {df_train.index[0].date()} → {df_train.index[-1].date()}  ({len(df_train)}d)")
print(f"  Holdout : {df_holdout.index[0].date()} → {df_holdout.index[-1].date()}  ({len(df_holdout)}d)")
print(f"  Beta    : {beta_train:.4f}  (hedge ratio, OLS on train)")
print(f"  OU mean : {ou_mean:.5f}  (β-adjusted spread equilibrium)")


beta = beta_train

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
diagnosis_obj.sensitivity_analysis(df_train, ou_hl, n_trials=120, n_seeds=4)

# Sync beta into backtest engine (in case it was updated)
if 'beta' in PARAMS:
    beta_train = PARAMS['beta']

print("\n  Params after Bayes:")
print(f"  {'slow_window':>20} : {PARAMS['slow_window']}")
print(f"  {'z_entry long/short':>20} : {PARAMS['z_entry_long']:.3f} / {PARAMS['z_entry_short']:.3f}")
print(f"  {'z_exit  long/short':>20} : {PARAMS['z_exit_long']:.3f} / {PARAMS['z_exit_short']:.3f}")
print(f"  {'z_stop  long/short':>20} : {PARAMS['z_stop_long']:.3f} / {PARAMS['z_stop_short']:.3f}")
print(f"  {'z_add':>20} : {PARAMS['z_add']:.3f}")
print(f"  {'vol_cap':>20} : {PARAMS['vol_cap']:.3f}")
print(f"  {'max_hold':>20} : {PARAMS['max_hold']}")
print(f"  {'beta (hedge ratio)':>20} : {beta_train:.4f}")


def permutation_test(df, PARAMS, beta, ou_mean,
                     feature_builder, backtest_engine, metrics_calc,
                     T1, T2, n_permutations=200, seed=42):
    """
    Permutation test for statistical significance of strategy edge.

    Method:
    - Shuffle daily log-returns of T1 and T2 independently
    - Reconstruct price series from shuffled returns
    - Run the full strategy (fixed params, fixed beta) on shuffled data
    - Compare real Sharpe to the null distribution

    Shuffling log-returns independently:
    - Destroys the cross-serial correlation (the spread relationship)
    - Preserves each instrument's marginal return distribution
    - Preserves autocorrelation structure within each leg
      (because we shuffle returns, not prices)

    If real Sharpe > 95th percentile of null → p < 0.05 → edge is real.
    """
    np.random.seed(seed)

    # ── Real strategy Sharpe ───────────────────────────────────
    feat_real, _ = feature_builder.build_features(
        df, PARAMS, ou_mean=ou_mean, beta=beta)
    sig_real     = feature_builder.generate_signals(feat_real, PARAMS)
    pnl_real, eq_real, _ = backtest_engine.backtest(
        feat_real, sig_real, PARAMS, cost=True)

    if pnl_real.std() == 0:
        print("  ✗ Real strategy has zero std — cannot run permutation test")
        return np.nan, np.array([])

    real_sharpe = float(pnl_real.mean() / pnl_real.std() * np.sqrt(252))

    # ── Null distribution ──────────────────────────────────────
    # Compute log-returns once — these are what we shuffle
    log_r1 = np.log(df[T1]).diff().dropna().values   # shape: (n-1,)
    log_r2 = np.log(df[T2]).diff().dropna().values

    # Starting prices (first row of df)
    p1_start = float(df[T1].iloc[0])
    p2_start = float(df[T2].iloc[0])

    null_sharpes = []

    for i in range(n_permutations):

        # Shuffle returns independently — destroys cross-serial structure
        r1_shuf = log_r1.copy()
        r2_shuf = log_r2.copy()
        np.random.shuffle(r1_shuf)
        np.random.shuffle(r2_shuf)

        # Reconstruct log-prices from shuffled log-returns
        # log(P_t) = log(P_0) + cumsum(log_returns)
        log_p1 = np.log(p1_start) + np.concatenate([[0], np.cumsum(r1_shuf)])
        log_p2 = np.log(p2_start) + np.concatenate([[0], np.cumsum(r2_shuf)])

        p1_new = np.exp(log_p1)
        p2_new = np.exp(log_p2)

        # Build a fresh DataFrame with same index as df
        df_perm = pd.DataFrame({
            T1: p1_new,
            T2: p2_new,
        }, index=df.index)

        # Sanity check — prices must be positive
        if (df_perm[T1] <= 0).any() or (df_perm[T2] <= 0).any():
            continue

        try:
            feat_p, _ = feature_builder.build_features(
                df_perm, PARAMS, ou_mean=ou_mean, beta=beta)
            sig_p     = feature_builder.generate_signals(feat_p, PARAMS)
            pnl_p, _, _ = backtest_engine.backtest(
                feat_p, sig_p, PARAMS, cost=True)

            if pnl_p.std() > 0:
                sh = float(pnl_p.mean() / pnl_p.std() * np.sqrt(252))
                null_sharpes.append(sh)

        except Exception:
            continue

    if len(null_sharpes) < 10:
        print(f"  ✗ Too few valid permutations ({len(null_sharpes)}) — "
              f"check build_features for errors on shuffled data")
        return np.nan, np.array(null_sharpes)

    null_sharpes = np.array(null_sharpes)
    p_value      = float((null_sharpes >= real_sharpe).mean())
    percentile   = float((null_sharpes < real_sharpe).mean() * 100)

    # ── Report ─────────────────────────────────────────────────
    print(f"""
── Permutation Test ({n_permutations} shuffles, seed={seed}) ──
  Real Sharpe             : {real_sharpe:.3f}
  Null mean ± std         : {null_sharpes.mean():.3f} ± {null_sharpes.std():.3f}
  Null 5th  percentile    : {np.percentile(null_sharpes,  5):.3f}
  Null 50th percentile    : {np.percentile(null_sharpes, 50):.3f}
  Null 95th percentile    : {np.percentile(null_sharpes, 95):.3f}
  Null 99th percentile    : {np.percentile(null_sharpes, 99):.3f}
  Real Sharpe percentile  : {percentile:.1f}th
  p-value                 : {p_value:.4f}
  Valid shuffles          : {len(null_sharpes)} / {n_permutations}
  Result                  : {'✓ Significant at p<0.01 — strong evidence of edge'
                              if p_value < 0.01 else
                             '✓ Significant at p<0.05 — edge likely real'
                              if p_value < 0.05 else
                             '~ Marginal (p<0.10) — weak evidence'
                              if p_value < 0.10 else
                             '✗ NOT significant — Sharpe consistent with noise'}
""")

    # ── Distribution summary ───────────────────────────────────
    # Show what fraction of null Sharpes fall in each bucket
    buckets = [(-np.inf, -0.5), (-0.5, 0), (0, 0.5),
               (0.5, 1.0), (1.0, np.inf)]
    labels  = ['< -0.5', '-0.5–0', '0–0.5', '0.5–1.0', '> 1.0']
    print(f"  Null distribution breakdown:")
    for (lo, hi), lab in zip(buckets, labels):
        pct = ((null_sharpes >= lo) & (null_sharpes < hi)).mean() * 100
        bar = '█' * int(pct / 2)
        print(f"    {lab:>8}  {bar:<25}  {pct:.1f}%")
    print(f"    {'REAL':>8}  → {real_sharpe:.3f}")

    return p_value, null_sharpes


# ── Usage in main.py ───────────────────────────────────────────────────────────
#
# Run on holdout only — that's your OOS evidence.
# Running on train would be circular (params were optimised on train).
#
# p_val, null_dist = permutation_test(
#     df_holdout, PARAMS, beta_train, ou_holdout,
#     feature_builder, backtest_engine, metrics_calc,
#     T1=T1, T2=T2,
#     n_permutations=200,
# )
#
# For BZ/CL you should see p < 0.01 and real Sharpe > 99th percentile.
# For BAJFINANCE/KOTAK, p < 0.05 is the minimum bar to proceed.



def parameter_sensitivity(df_holdout, PARAMS, beta, ou_holdout,
                           feature_builder, backtest_engine):
    """
    Perturb each parameter by ±10% and ±20% and record
    holdout Sharpe. A robust strategy should show smooth
    degradation, not cliff-edges.
    """
    base_params = PARAMS.copy()

    # Parameters to test and their perturbation scale
    test_params = {
        'z_entry_long':  [0.8, 0.9, 1.0, 1.1, 1.2],  # multipliers
        'z_entry_short': [0.8, 0.9, 1.0, 1.1, 1.2],
        'z_exit_long':   [0.8, 0.9, 1.0, 1.1, 1.2],
        'z_exit_short':  [0.8, 0.9, 1.0, 1.1, 1.2],
        'slow_window':   [0.8, 0.9, 1.0, 1.1, 1.2],
        'max_hold':      [0.8, 0.9, 1.0, 1.1, 1.2],
    }

    print(f"── Parameter Sensitivity (holdout Sharpe) ──")
    print(f"  {'Param':<20} {'0.8×':>7} {'0.9×':>7} "
          f"{'BASE':>7} {'1.1×':>7} {'1.2×':>7}  Stability")
    print("  " + "─" * 65)

    results = {}
    for param, multipliers in test_params.items():
        sharpes = []
        for m in multipliers:
            p = base_params.copy()
            p[param] = base_params[param] * m
            try:
                feat, _ = feature_builder.build_features(
                    df_holdout, p, ou_mean=ou_holdout, beta=beta)
                sig     = feature_builder.generate_signals(feat, p)
                pnl, _, _ = backtest_engine.backtest(
                    feat, sig, p, cost=True)
                sh = (pnl.mean() / pnl.std() * np.sqrt(252)
                      if pnl.std() > 0 else 0)
                sharpes.append(sh)
            except Exception:
                sharpes.append(np.nan)

        stability = np.nanstd(sharpes)
        flag = ('✓ stable' if stability < 0.3 else
                '~ moderate' if stability < 0.6 else
                '✗ fragile')

        print(f"  {param:<20} "
              + "  ".join(f"{s:>7.2f}" for s in sharpes)
              + f"  {flag}")
        results[param] = sharpes

    return results


def regime_analysis( feat,  pnl, trades):
    """
    Check strategy performance across:
    1. Bull vs Bear equity regimes (Nifty direction)
    2. High vs Low volatility regimes (VIX proxy)
    3. High vs Low spread volatility regimes
    """
    # ── Regime 1: Spread volatility ───────────────────────────
    spread_vol = feat['lr'].rolling(63).std()
    vol_median = spread_vol.median()

    high_vol_mask = spread_vol > vol_median
    low_vol_mask  = spread_vol <= vol_median

    pnl_high_vol = pnl[high_vol_mask.reindex(pnl.index).fillna(False)]
    pnl_low_vol  = pnl[low_vol_mask.reindex(pnl.index).fillna(False)]

    def sharpe(s):
        return s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0

    print(f"""
── Regime Analysis ──
  Spread Vol Regime:
    High vol  Sharpe : {sharpe(pnl_high_vol):.2f}  (n={len(pnl_high_vol)}d)
    Low vol   Sharpe : {sharpe(pnl_low_vol):.2f}  (n={len(pnl_low_vol)}d)
    {'✓ works in both' if sharpe(pnl_high_vol) > 0 and sharpe(pnl_low_vol) > 0
     else '✗ regime-dependent — only works in one vol environment'}
""")

    # ── Regime 2: Trade-level by entry z-score magnitude ──────
    if trades is not None and len(trades) > 0:
        trades['entry_z_abs'] = trades['entry_z'].abs()
        low_z  = trades[trades['entry_z_abs'] < trades['entry_z_abs'].median()]
        high_z = trades[trades['entry_z_abs'] >= trades['entry_z_abs'].median()]

        low_z_wr  = (low_z['pnl']  > 0).mean() * 100
        high_z_wr = (high_z['pnl'] > 0).mean() * 100
        low_z_avg  = low_z['pnl'].mean()  * 200_000
        high_z_avg = high_z['pnl'].mean() * 200_000

        print(f"  Entry Z-Score Magnitude:")
        print(f"    Low  |z| trades  : WR={low_z_wr:.0f}%  avg={low_z_avg:,.0f}")
        print(f"    High |z| trades  : WR={high_z_wr:.0f}%  avg={high_z_avg:,.0f}")
        print(f"    {'✓ higher z = better entry' if high_z_wr > low_z_wr else '✗ z-score magnitude doesnt predict outcome'}")

    # ── Regime 3: Year by year ─────────────────────────────────
    print(f"\n  Year-by-Year Breakdown:")
    print(f"  {'Year':>6}  {'Sharpe':>7}  {'Ret':>10}  "
          f"{'n_trades':>8}  {'WR':>6}")
    print("  " + "─" * 45)
    for year in sorted(pnl.index.year.unique()):
        pnl_y = pnl[pnl.index.year == year]
        tr_y  = (trades[trades['entry_date'].dt.year == year]
                 if trades is not None and len(trades) > 0
                 else pd.DataFrame())
        sh_y  = sharpe(pnl_y)
        ret_y = pnl_y.sum() * 200_000
        n_y   = len(tr_y)
        wr_y  = (tr_y['pnl'] > 0).mean() * 100 if n_y > 0 else np.nan
        flag  = '✓' if sh_y > 0 else '✗'
        print(f"  {year:>6}  {sh_y:>7.2f}  {ret_y:>9,.0f}  "
              f"{n_y:>8}  {wr_y:>5.1f}%  {flag}")




def anchored_wfv(df, PARAMS, beta, feature_builder,
                 backtest_engine):
    """
    Anchored walk-forward: training always starts at the same date
    but the test window moves forward.
    Tests whether performance is consistent as more history is available.
    """
    ANCHOR_START  = 0           # always train from the beginning
    MIN_TRAIN     = 504         # minimum training bars
    TEST_DAYS     = 126
    STEP_DAYS     = 63

    print(f"── Anchored Walk-Forward Validation ──")
    print(f"  {'Fold':>4}  {'Train':>6}  {'Test Period':>23}  "
          f"{'n':>4}  {'Sharpe':>7}  {'Calmar':>7}  {'WR':>6}")
    print("  " + "─" * 70)

    fold = 0
    te   = MIN_TRAIN  # test starts after minimum training

    while te + TEST_DAYS <= len(df):
        xe = te + TEST_DAYS

        df_ftr = df.iloc[ANCHOR_START:te].copy()
        df_fts = df.iloc[te:xe].copy()

        ou_f = float(
            (np.log(df_ftr[T1]) - beta * np.log(df_ftr[T2])).mean())

        feat_f, _ = feature_builder.build_features(
            df_fts, PARAMS, ou_mean=ou_f, beta=beta)
        sig_f     = feature_builder.generate_signals(feat_f, PARAMS)
        pnl_f, eq_f, tr_f = backtest_engine.backtest(
            feat_f, sig_f, PARAMS, cost=True)

        n_tr    = len(tr_f)
        sharpe  = (pnl_f.mean() / pnl_f.std() * np.sqrt(252)
                   if pnl_f.std() > 0 else 0)
        ann_ret = pnl_f.mean() * 252
        roll_max = eq_f.cummax()
        max_dd  = float(((eq_f - roll_max) /
                          roll_max.replace(0, np.nan)).min())
        calmar  = ann_ret / abs(max_dd) if max_dd < -1e-6 else np.nan
        wr      = float((tr_f['pnl'] > 0).mean()) if n_tr > 0 else np.nan

        calmar_str = f"{calmar:>7.2f}" if not np.isnan(calmar) else "    nan"
        print(f"  {fold+1:>4}  {te:>6}d  "
              f"{str(df.index[te].date())}→{str(df.index[xe-1].date())}  "
              f"{n_tr:>4}  {sharpe:>7.2f}  {calmar_str}  "
              f"{wr*100:>5.1f}%")

        te   += STEP_DAYS
        fold += 1


def cost_stress_test(df, feat, sig, PARAMS, beta, ou_mean,
                     backtest_engine, metrics_calc):
    """
    Run the strategy at 1x, 2x, 3x, 5x, 10x transaction costs.
    A real edge should survive at least 3x costs.
    Pairs with thin edges collapse quickly.
    """
    cost_multipliers = [1.0, 2.0, 3.0, 5.0, 10.0]

    print(f"── Transaction Cost Stress Test ──")
    print(f"  {'Cost ×':>7}  {'Sharpe':>7}  {'Calmar':>7}  "
          f"{'Ann Ret':>8}  {'MaxDD':>7}  {'n':>4}  Verdict")
    print("  " + "─" * 65)

    base_commission = backtest_engine.COMMISSION
    base_stt        = backtest_engine.STT
    base_slippage   = backtest_engine.SLIPPAGE

    for m in cost_multipliers:
        # Temporarily scale costs
        backtest_engine.COMMISSION = base_commission * m
        backtest_engine.STT        = base_stt        * m
        backtest_engine.SLIPPAGE   = base_slippage   * m
        backtest_engine.COST_RT    = (backtest_engine.COMMISSION +
                                      backtest_engine.STT +
                                      backtest_engine.SLIPPAGE * 2)

        pnl_m, eq_m, tr_m = backtest_engine.backtest(
            feat, sig, PARAMS, cost=True)
        met_m = metrics_calc.calc_metrics(pnl_m, eq_m, tr_m)

        verdict = ('✓' if met_m['sharpe'] > 0.5 else
                   '~' if met_m['sharpe'] > 0   else '✗')

        print(f"  {m:>7.1f}×  {met_m['sharpe']:>7.2f}  "
              f"{met_m['calmar']:>7.2f}  "
              f"{met_m['ann_ret']*100:>7.1f}%  "
              f"{met_m['max_dd']*100:>6.1f}%  "
              f"{met_m['n_trades']:>4}  {verdict}")

    # Restore original costs
    backtest_engine.COMMISSION = base_commission
    backtest_engine.STT        = base_stt
    backtest_engine.SLIPPAGE   = base_slippage
    backtest_engine.COST_RT    = (base_commission + base_stt +
                                  base_slippage * 2)



def beta_stability(df, T1, T2, feature_builder, window=252):
    """
    Roll a 252-day OLS window through the full dataset.
    Reports rolling beta statistics to assess relationship stability.
    """
    betas = []
    dates = []

    for i in range(window, len(df)):
        df_window = df.iloc[i-window:i].copy()

        # Skip windows with bad data
        if (df_window[T1] <= 0).any() or (df_window[T2] <= 0).any():
            continue
        if df_window[[T1, T2]].isnull().any().any():
            continue

        b = feature_builder.estimate_hedge_ratio(
            df_window, T1, T2, lookback=None)

        # Sanity check — reject extreme betas
        # OLS can produce wild values on short windows with outliers.
        # A beta outside [-10, 10] is almost certainly a bad data window.
        if not np.isfinite(b) or abs(b) > 10:
            continue

        betas.append(b)
        dates.append(df.index[i])

    if len(betas) < 10:
        print(f"  ✗ Too few valid windows ({len(betas)}) for beta stability")
        return pd.Series(dtype=float)

    beta_series  = pd.Series(betas, index=dates)
    beta_std     = beta_series.std()
    beta_mean    = beta_series.mean()
    beta_median  = beta_series.median()
    beta_trend   = np.polyfit(range(len(betas)), betas, 1)[0]

    # Use median (not mean) as the reference for stability check.
    # Mean is distorted by outlier windows. Median is robust.
    # Use abs(median) to avoid divide-by-near-zero when mean ≈ 0.
    ref = abs(beta_median) if abs(beta_median) > 0.1 else 1.0
    rel_std = beta_std / ref

    verdict = ('✓ stable — β varies less than 15% of median'
               if rel_std < 0.15 else
               '⚠ moderate drift — monitor β every 6 months'
               if rel_std < 0.30 else
               '✗ unstable — β is drifting, re-estimate quarterly')

    print(f"""
── Rolling Beta Stability ({window}d window) ──
  Windows computed : {len(betas)}
  Beta mean        : {beta_mean:.4f}
  Beta median      : {beta_median:.4f}  ← use this as reference
  Beta std         : {beta_std:.4f}
  Beta range       : {beta_series.min():.4f} → {beta_series.max():.4f}
  Relative std     : {rel_std:.3f}  (std / |median|)
  Beta trend       : {beta_trend*252:.4f} per year
  {verdict}
""")


    return beta_series



# Step 1 — does the edge exist above chance?
p_val, null_dist = permutation_test(df, PARAMS, beta, ou_mean, feature_builder, backtest_engine, metrics_calc, T1=T1, T2=T2, n_permutations=200)

# If p_value > 0.10, stop here — it's noise

# Step 2 — is it regime-dependent?

# If it only works in 2024-2025 specifically, it's a recent artefact

# Step 3 — are params knife-edge?
parameter_sensitivity(df, PARAMS, beta, ou_mean, feature_builder,backtest_engine)

# Step 4 — does β drift?
beta_series = beta_stability(df, T1, T2, feature_builder)

# Step 5 — cost robustness


anchored_wfv(df, PARAMS, beta, feature_builder, backtest_engine)

