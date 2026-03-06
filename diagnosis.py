import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats
from features import features
from backtest import backtest
from metrics import metrics
from params import PARAMS
import optuna


class diagnosis():

    def __init__(self):
        self.T1 = "BZ=F"
        self.T2 = "CL=F"
        self.PARAMS   = PARAMS
        self.features = features()
        self.backtest = backtest()
        self.metrics  = metrics()

    # ══════════════════════════════════════════════════════
    # BAYESIAN OPTIMISATION  (replaces sensitivity_analysis)
    #
    # Objective: mean OOS Sharpe across inner WFV folds
    #            minus 0.5 × std  ← penalises instability
    #
    # Inner WFV: 3 folds of 1yr train / 3mo test carved
    #            entirely within the df passed in (train set)
    #
    # Parameters searched:
    #   slow_window, z_entry_long, z_entry_short,
    #   z_exit_long, z_exit_short, z_stop_long, z_stop_short,
    #   z_add, vol_cap, max_hold
    # ══════════════════════════════════════════════════════
    def sensitivity_analysis(self, df, ou_hl, n_trials=80):
        """
        Drop-in replacement for grid-search sensitivity_analysis.
        Returns (results_dict, best_sw, best_ze) for compatibility
        with existing main script.
        """
        print("\n── Bayesian Optimisation (Optuna)  ──")
        print(f"   Searching over asymmetric z-params + slow_window")
        print(f"   Trials: {n_trials}  |  Inner WFV: 3 folds\n")

        # ── Inner WFV fold definitions (fixed, carved from df) ──
        n      = len(df)
        fold_configs = []
        INNER_TRAIN = 252    # 1yr
        INNER_TEST  = 63     # 3mo
        INNER_STEP  = 63     # 3mo step

        for i in range(3):   # exactly 3 inner folds
            ts = i * INNER_STEP
            te = ts + INNER_TRAIN
            xs = te
            xe = xs + INNER_TEST
            if xe > n:
                break
            fold_configs.append((ts, te, xs, xe))

        if len(fold_configs) < 2:
            print("  ⚠ Not enough data for inner WFV — falling back to single eval")

        # ── Objective function ──
        def objective(trial):
            p = self.PARAMS.copy()

            HALF_LIFE = ou_hl

            # ── Search space ──
            # Remove medium_window from optimisation — keep fixed at 30
            # It has near-zero importance and adds degrees of freedom
            p['slow_window'] = trial.suggest_int('slow_window', 15, 90)
            p['z_entry_long'] = trial.suggest_float('z_entry_long', 0.7, 2.0)
            p['z_entry_short'] = trial.suggest_float('z_entry_short', 0.7, 2.0)
            p['z_exit_long'] = trial.suggest_float('z_exit_long', 0.05, 0.45)
            p['z_exit_short'] = trial.suggest_float('z_exit_short', 0.05, 0.45)
            p['z_stop_long'] = trial.suggest_float('z_stop_long', 2.0, 5.0)
            p['z_stop_short'] = trial.suggest_float('z_stop_short', 2.0, 5.0)
            p['z_add'] = trial.suggest_float('z_add', 1.5, 3.5)
            p['vol_cap'] = trial.suggest_float('vol_cap', 1.2, 4.0)
            min_hold = max(12, int(HALF_LIFE * 1.5))
            p['max_hold'] = trial.suggest_int('max_hold', min_hold, min_hold + 25)

            p['z_entry'] = (p['z_entry_long'] + p['z_entry_short']) / 2
            p['z_exit'] = (p['z_exit_long'] + p['z_exit_short']) / 2
            p['z_stop'] = (p['z_stop_long'] + p['z_stop_short']) / 2

            fold_results = []

            for (ts, te, xs, xe) in fold_configs:
                try:
                    df_ftr = df.iloc[ts:te].copy()
                    df_fts = df.iloc[xs:xe].copy()
                    ou_f = float(np.log(df_ftr[self.T1] / df_ftr[self.T2]).mean())
                    test_days = xe - xs

                    feat, _ = self.features.build_features(df_fts, p, ou_mean=ou_f)
                    sig = self.features.generate_signals(feat, p)
                    pnl, eq, tr = self.backtest.backtest(feat, sig, p, cost=True)

                    n = len(tr)

                    if n == 0 or pnl.std() == 0:
                        fold_results.append(None)
                        continue

                    # ── Core metrics ──
                    ann_ret = pnl.mean() * 252
                    ann_vol = pnl.std() * np.sqrt(252)
                    total_ret = pnl.sum()

                    roll_max = eq.cummax()
                    max_dd = float(((eq - roll_max) / roll_max.replace(0, np.nan)).min())

                    # ── Omega ratio at threshold=0 ──
                    # Theoretically correct for non-normal, negatively skewed returns
                    # = E[max(r-threshold, 0)] / E[max(threshold-r, 0)]
                    # At threshold=0: gains_mass / losses_mass
                    gains = pnl[pnl > 0].sum()
                    losses = abs(pnl[pnl < 0].sum())
                    omega = gains / losses if losses > 0 else (3.0 if gains > 0 else 0.0)
                    omega = float(np.clip(omega, 0, 8.0))  # cap at 8 to prevent inf

                    # ── Profit factor (trade-level omega equivalent) ──
                    t_wins = tr.loc[tr['pnl'] > 0, 'pnl'].sum()
                    t_loss = abs(tr.loc[tr['pnl'] < 0, 'pnl'].sum())
                    pf = t_wins / t_loss if t_loss > 0 else (3.0 if t_wins > 0 else 0.0)
                    pf = float(np.clip(pf, 0, 8.0))

                    # ── Calmar — only when drawdown is meaningful ──
                    calmar = ann_ret / abs(max_dd) if max_dd < -0.001 else np.nan

                    # ── Capital utilisation ──
                    util = tr['hold_days'].sum() / test_days

                    # ── Time-stop rate ──
                    ts_pct = (tr['exit_reason'] == 'time_stop').mean()

                    fold_results.append(dict(
                        n=n,
                        ann_ret=ann_ret,
                        total_ret=total_ret,
                        omega=omega,
                        pf=pf,
                        calmar=calmar,
                        util=util,
                        ts_pct=ts_pct,
                    ))

                except Exception:
                    fold_results.append(None)

            valid = [f for f in fold_results if f is not None]
            if len(valid) < 2:
                return -99.0

            avg_n = np.mean([f['n'] for f in valid])
            avg_ret = np.mean([f['ann_ret'] for f in valid])
            avg_tot_ret = np.mean([f['total_ret'] for f in valid])
            avg_omega = np.mean([f['omega'] for f in valid])
            std_omega = np.std([f['omega'] for f in valid])
            avg_pf = np.mean([f['pf'] for f in valid])
            avg_util = np.mean([f['util'] for f in valid])
            avg_ts_pct = np.mean([f['ts_pct'] for f in valid])
            pct_profit = np.mean([f['ann_ret'] > 0 for f in valid])

            valid_calmar = [f['calmar'] for f in valid
                            if f['calmar'] is not None
                            and not np.isnan(f['calmar'])
                            and -20 < f['calmar'] < 50]
            avg_calmar = np.mean(valid_calmar) if valid_calmar else -1.0

            # ── Hard floors — minimal, only kill truly broken configs ──
            if avg_n < 3:
                return -99.0  # not a strategy
            if pct_profit < 0.40:
                return -5.0  # unprofitable in >60% of folds
            if avg_ts_pct > 0.45:
                return -3.0  # mostly exiting via time-stop

            # ── Time-stop soft penalty ──
            ts_pen = -2.5 * max(0.0, avg_ts_pct - 0.20)

            # ── COMPOSITE OBJECTIVE ──
            #
            # Designed around three principles:
            #
            # 1. PRIMARY: Absolute dollar return (uncapped)
            #    We want more money, period. log1p scaling prevents one
            #    exceptional fold from dominating while still rewarding
            #    larger returns proportionally.
            #    Weight: 3.0 — this is the most important term
            #
            # 2. SECONDARY: Omega ratio consistency
            #    Omega is theoretically correct for our return distribution
            #    (negatively skewed, high win-rate, fat-tailed losses).
            #    Unlike Sharpe, it doesn't penalise upside vol.
            #    We reward the mean but penalise instability across folds.
            #    Weight: 2.0
            #
            # 3. TERTIARY: Profit factor consistency
            #    Trade-level quality — are individual trades worth taking?
            #    Separate from Omega (daily PnL) to capture trade structure.
            #    Weight: 1.0
            #
            # 4. QUATERNARY: Frequency + utilisation
            #    Both log-scaled to prevent over-trading solutions.
            #    Combined weight: ~0.7 — tiebreaker only
            #
            # 5. PENALTY: Time-stop rate
            #    Structural health signal — too many time-stops means
            #    entries are misaligned with the half-life

            # Term 1: absolute return — uncapped, log-scaled
            # log1p(ret * 200): at 1% ann_ret → 0.69, at 3% → 1.79, at 5% → 2.40
            # Negative returns: log1p(max(x, -0.99)) still gives negative score
            ret_score = np.log1p(np.clip(avg_tot_ret * 200, -0.99, 20.0)) * 3.0

            # Term 2: omega consistency
            # Mean omega penalised by cross-fold std — rewards stable quality
            omega_score = np.clip(avg_omega - 0.5 * std_omega, 0, 6.0) * 2.0

            # Term 3: profit factor
            pf_score = np.clip(avg_pf - 1.0, 0, 5.0) * 1.0  # excess above breakeven

            # Term 4: frequency and utilisation (combined tiebreaker)
            freq_score = np.log(max(avg_n, 3) / 3.0) * 0.4
            util_score = np.clip(avg_util, 0, 0.5) * 0.6

            score = ret_score + omega_score + pf_score + freq_score + util_score + ts_pen

            if trial.number < 15:
                print(f"  t{trial.number:>3} | n={avg_n:.1f} "
                      f"ret={avg_ret * 100:.2f}% ω={avg_omega:.2f}±{std_omega:.2f} "
                      f"pf={avg_pf:.2f} util={avg_util * 100:.0f}% ts={avg_ts_pct * 100:.0f}% | "
                      f"ret={ret_score:.2f} ω={omega_score:.2f} pf={pf_score:.2f} "
                      f"fr={freq_score:.2f} ut={util_score:.2f} ts={ts_pen:.2f} "
                      f"→ {score:.3f}")

            return float(score)

        # In sensitivity_analysis, before creating the study:
        stat_check = self.backtest.__class__  # just to confirm
        lr_check = np.log(df[self.T1] / df[self.T2]).dropna()
        s_lag = lr_check.shift(1).dropna()
        import statsmodels.api as sm
        res = sm.OLS(lr_check.iloc[1:], sm.add_constant(s_lag)).fit()
        phi = min(float(res.params.iloc[1]), 1 - 1e-8)
        kappa = -np.log(max(phi, 1e-8)) * 252
        self._half_life = np.log(2) / kappa * 252
        print(
            f"   Half-life detected: {self._half_life:.1f}d → min max_hold set to {max(10, int(self._half_life * 1.5))}d")

        # ── Run optimisation ──
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
        )
        study.optimize(objective, n_trials=120, show_progress_bar=True)  # more trials now

        best        = study.best_params
        best_score  = study.best_value

        # ── Print results ──
        print(f"\n  Best objective (mean_sharpe - 0.5×std): {best_score:.3f}")
        print(f"\n  Optimal parameters found:")
        print(f"    slow_window    : {best['slow_window']}")
        print(f"    z_entry_long   : {best['z_entry_long']:.3f}")
        print(f"    z_entry_short  : {best['z_entry_short']:.3f}")
        print(f"    z_exit_long    : {best['z_exit_long']:.3f}")
        print(f"    z_exit_short   : {best['z_exit_short']:.3f}")
        print(f"    z_stop_long    : {best['z_stop_long']:.3f}")
        print(f"    z_stop_short   : {best['z_stop_short']:.3f}")
        print(f"    z_add          : {best['z_add']:.3f}")
        print(f"    vol_cap        : {best['vol_cap']:.3f}")
        print(f"    max_hold       : {best['max_hold']}")

        # ── Asymmetry insight ──
        asym_entry = best['z_entry_short'] - best['z_entry_long']
        asym_stop  = best['z_stop_long']   - best['z_stop_short']
        print(f"\n  Asymmetry discovered:")
        print(f"    Entry asymmetry (short - long) : {asym_entry:+.3f}  "
              f"({'short harder to enter ✓' if asym_entry > 0.1 else 'roughly symmetric'})")
        print(f"    Stop  asymmetry (long - short) : {asym_stop:+.3f}  "
              f"({'long gets wider stop ✓'  if asym_stop  > 0.1 else 'roughly symmetric'})")

        # ── Importance ──
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\n  Parameter importance (top 5):")
            for k, v in list(importance.items())[:5]:
                bar = '█' * int(v * 30)
                print(f"    {k:<20} {bar}  {v:.3f}")
        except Exception:
            pass

        # ── Update PARAMS in-place with best found ──
        for k, v in best.items():
            self.PARAMS[k] = v
        self.PARAMS['z_entry'] = (best['z_entry_long'] + best['z_entry_short']) / 2
        self.PARAMS['z_exit']  = (best['z_exit_long']  + best['z_exit_short'])  / 2
        self.PARAMS['z_stop']  = (best['z_stop_long']  + best['z_stop_short'])  / 2

        # ── Compatibility return (same signature as old sensitivity_analysis) ──
        # results dict: (sw, ze) → (sharpe, n_trades) — sparse, best params only
        results = {(best['slow_window'], best['z_entry_long']): (best_score, -1)}

        return results, best['slow_window'], best['z_entry_long']

    # ══════════════════════════════════════════════════════
    # STAT DIAGNOSTICS  (unchanged)
    # ══════════════════════════════════════════════════════
    def run_stat_diag(self, df, feat):
        lr = feat['lr'].dropna()

        adf_stat, adf_p, _, _, crit, _ = adfuller(lr, autolag='AIC')

        lags = range(2, 60)
        tau  = [np.std(np.subtract(lr.values[l:], lr.values[:-l])) for l in lags]
        hurst, *_ = stats.linregress(np.log(list(lags)), np.log(tau))

        s_lag = lr.shift(1).dropna()
        s_cur = lr.iloc[1:]
        res   = sm.OLS(s_cur, sm.add_constant(s_lag)).fit()
        phi   = min(float(res.params.iloc[1]), 1 - 1e-8)
        kappa = -np.log(max(phi, 1e-8)) * 252
        hl    = np.log(2) / kappa * 252

        try:
            joh      = coint_johansen(df[[self.T1, self.T2]], det_order=0, k_ar_diff=1)
            joh_pass = bool(joh.lr1[0] > joh.cvt[0, 1])
            joh_trace= float(joh.lr1[0])
            joh_crit = float(joh.cvt[0, 1])
        except Exception:
            joh_pass = False; joh_trace = np.nan; joh_crit = np.nan

        return dict(adf_p=adf_p, adf_stat=adf_stat, adf_crit=crit,
                    hurst=hurst, half_life=hl, kappa=kappa,
                    joh_pass=joh_pass, joh_trace=joh_trace, joh_crit=joh_crit)



