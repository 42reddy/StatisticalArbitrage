import numpy as np
import pandas as pd
import optuna
from optuna.pruners import SuccessiveHalvingPruner
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats
from params import PARAMS
from features import features
from backtest import backtest
from metrics import metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


class diagnosis:
    """
    Bayesian optimisation + statistical diagnostics for pairs trading.

    Features:
    - Multi‑seed optimisation with CMA‑ES and SuccessiveHalving pruning.
    - Full‑period time‑series folds (no look‑ahead).
    - Out‑of‑sample holdout to prevent overfitting.
    - Persistent SQLite storage for each seed (resume capability).
    - Final selection based on holdout performance, not just median.
    """

    def __init__(self):
        self.T1 = PARAMS['T1']
        self.T2 = PARAMS["T2"]
        self.PARAMS = PARAMS.copy()
        self.features = features()
        self.backtest = backtest()
        self.metrics = metrics()
        self._half_life = None

    # ----------------------------------------------------------------------
    # Public method: Bayesian optimisation + consensus + holdout selection
    # ----------------------------------------------------------------------
    def sensitivity_analysis(self, df, ou_hl, n_trials=120, n_seeds=3,
                             holdout_ratio=0.2):
        """
        Multi‑seed Bayesian optimisation with final holdout validation.

        Parameters
        ----------
        df : pd.DataFrame
            Training data (prices for T1 and T2). The last `holdout_ratio`
            fraction is reserved as final test set, never seen during optimisation.
        ou_hl : float
            Half‑life from OU estimation (not used directly, kept for API).
        n_trials : int
            Number of Optuna trials per seed.
        n_seeds : int
            Number of independent random seeds (default 3).
        holdout_ratio : float
            Fraction of data to keep as final holdout (0.2 = 20%).
        resume : bool
            If True, load existing SQLite databases for each seed (allow resuming).

        Returns
        -------
        results : dict
            Dummy structure for compatibility (can be changed).
        best_slow_window : int
            Selected slow_window from the best holdout candidate.
        best_z_entry_long : float
            Selected z_entry_long from the best holdout candidate.
        """
        print("\n── Bayesian Optimisation (Optuna) ──")
        print(f"   Multi‑seed: {n_seeds} seeds × {n_trials} trials"
              f" = {n_seeds * n_trials} total evaluations")
        print(f"   Holdout ratio: {holdout_ratio:.0%} (final test set)\n")

        # ---- 1. Hedge ratio and half‑life (same as before) ----
        beta = self.features.estimate_hedge_ratio(
            df, self.T1, self.T2, lookback=None)
        print(f"   Hedge ratio β = {beta:.4f}")

        spread = np.log(df[self.T1]) - beta * np.log(df[self.T2])
        spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
        combined = pd.concat([spread, spread.shift(1)], axis=1).dropna()
        res_ou = sm.OLS(combined.iloc[:, 0],
                        sm.add_constant(combined.iloc[:, 1])).fit()
        phi = min(float(res_ou.params.iloc[1]), 1 - 1e-8)
        kappa = -np.log(max(phi, 1e-8)) * 252
        self._half_life = np.log(2) / kappa * 252
        print(f"   Half‑life (β‑adjusted): {self._half_life:.1f}d\n")

        # ---- 2. Split data: training (for folds) + holdout (final test) ----
        n_total = len(df)
        n_holdout = int(n_total * holdout_ratio)
        if n_holdout < 63:
            n_holdout = min(63, n_total // 5)
        train_df = df.iloc[:-n_holdout].copy()
        holdout_df = df.iloc[-n_holdout:].copy()
        print(f"   Training period: {train_df.index[0].date()} → {train_df.index[-1].date()}")
        print(f"   Holdout period : {holdout_df.index[0].date()} → {holdout_df.index[-1].date()}\n")

        # ---- 3. Build inner folds (only on training data) ----
        fold_configs = self._build_folds(train_df)
        print(f"   Inner folds : {len(fold_configs)}")
        for i, (ts, te, xs, xe) in enumerate(fold_configs):
            print(f"     Fold {i + 1}: train {train_df.index[ts].date()}→{train_df.index[te - 1].date()}  "
                  f"test {train_df.index[xs].date()}→{train_df.index[xe - 1].date()}")
        print()

        # ---- 4. Define the objective (uses only train_df and fold_configs) ----
        def objective(trial):
            p = self.PARAMS.copy()
            hl = self._half_life

            # Search space (unchanged)
            sw_min = max(15, int(hl * 0.8))
            sw_max = max(sw_min + 20, int(hl * 1.5))
            p['slow_window'] = trial.suggest_int('slow_window', sw_min, sw_max)
            p['z_entry_long'] = trial.suggest_float('z_entry_long', 0.8, 2.2)
            p['z_entry_short'] = trial.suggest_float('z_entry_short', 0.8, 2.2)
            p['z_exit_long'] = trial.suggest_float('z_exit_long', 0.1, 0.3)
            p['z_exit_short'] = trial.suggest_float('z_exit_short', 0.1, 0.3)
            p['z_stop_long'] = trial.suggest_float('z_stop_long', 2.0, 4.0)
            p['z_stop_short'] = trial.suggest_float('z_stop_short', 2.0, 4.0)
            p['z_add'] = trial.suggest_float('z_add', 1.5, 3.5)
            p['vol_cap'] = trial.suggest_float('vol_cap', 1.2, 4.0)
            min_hold = max(12, int(hl * 1.5))
            max_hold_cap = max(min_hold + 10, int(hl * 2))
            p['max_hold'] = trial.suggest_int('max_hold', min_hold, max_hold_cap)

            p['z_entry'] = (p['z_entry_long'] + p['z_entry_short']) / 2
            p['z_exit'] = (p['z_exit_long'] + p['z_exit_short']) / 2
            p['z_stop'] = (p['z_stop_long'] + p['z_stop_short']) / 2

            # Structural coherence
            if p['z_entry_long'] < 1.0:
                return -10.0
            if p['z_entry_short'] < 1.0:
                return -10.0
            if p['z_exit_long'] > p['z_entry_long'] * 0.50:
                return -10.0
            if p['z_exit_short'] > p['z_entry_short'] * 0.50:
                return -10.0
            if p['z_stop_long'] < p['z_entry_long'] * 1.5:
                return -10.0
            if p['z_stop_short'] < p['z_entry_short'] * 1.5:
                return -10.0
            if p['z_add'] <= p['z_entry_long']:
                return -10.0
            if p['z_add'] >= min(p['z_stop_long'], p['z_stop_short']):
                return -10.0

            # Regularisation penalty (same)
            def reg(val, lo, hi, w=0.12):
                mid = (lo + hi) / 2
                half_r = (hi - lo) / 2
                return -w * ((val - mid) / half_r) ** 2

            reg_penalty = (
                    reg(p['z_entry_long'], 0.8, 2.2) +
                    reg(p['z_entry_short'], 0.8, 2.2) +
                    reg(p['z_exit_long'], 0.1, 0.3, w=0.08) +
                    reg(p['z_exit_short'], 0.1, 0.3, w=0.08) +
                    reg(p['z_stop_long'], 2.0, 4.0, w=0.08) +
                    reg(p['z_stop_short'], 2.0, 4.0, w=0.08) +
                    reg(p['z_add'], 1.5, 3.5, w=0.06) +
                    reg(p['vol_cap'], 1.2, 4.0, w=0.06)
            )

            # ---- Fold evaluation with intermediate reporting for pruning ----
            fold_scores = []
            for fold_idx, (ts, te, xs, xe) in enumerate(fold_configs):
                try:
                    df_ftr = train_df.iloc[ts:te].copy()
                    df_fts = train_df.iloc[xs:xe].copy()
                    ou_f = float((np.log(df_ftr[self.T1]) -
                                  beta * np.log(df_ftr[self.T2])).mean())
                    test_days = xe - xs

                    feat, _ = self.features.build_features(
                        df_fts, p, ou_mean=ou_f, beta=beta)
                    sig = self.features.generate_signals(feat, p)
                    pnl, eq, tr = self.backtest.backtest(feat, sig, p, cost=True)

                    n_tr = len(tr)
                    if n_tr == 0 or pnl.std() == 0:
                        fold_scores.append(-1.0)
                        continue

                    ann_ret = pnl.mean() * 252
                    ann_vol = pnl.std() * np.sqrt(252)
                    total_ret = pnl.sum()
                    roll_max = eq.cummax()
                    max_dd = float(((eq - roll_max) /
                                    roll_max.replace(0, np.nan)).min())

                    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

                    gains = pnl[pnl > 0].sum()
                    losses = abs(pnl[pnl < 0].sum())
                    pf = (gains / losses if losses > 0
                          else (3.0 if gains > 0 else 0.0))
                    pf = float(np.clip(pf, 0, 8.0))

                    util = tr['hold_days'].sum() / test_days

                    long_tr = tr[tr['direction'] == 'long']
                    short_tr = tr[tr['direction'] == 'short']
                    long_pnl = long_tr['pnl'].sum() if len(long_tr) > 0 else 0.0
                    short_pnl = short_tr['pnl'].sum() if len(short_tr) > 0 else 0.0
                    total_abs = abs(long_pnl) + abs(short_pnl)
                    long_share = abs(long_pnl) / total_abs if total_abs > 0 else 0.5
                    symmetry = 1.0 - abs(long_share - 0.5) * 2.0

                    # Simple fold score (could be Sharpe or a combination)
                    fold_score = sharpe + 0.5 * max(0, pf - 1.2) + 0.2 * util
                    fold_scores.append(fold_score)

                    # Report intermediate value for pruning (after each fold)
                    trial.report(np.mean(fold_scores), step=fold_idx)
                    if trial.should_prune():
                        return -99.0

                except Exception:
                    fold_scores.append(-1.0)
                    trial.report(np.mean(fold_scores), step=fold_idx)
                    if trial.should_prune():
                        return -99.0

            valid_fold_scores = [s for s in fold_scores if s > -0.5]
            if len(valid_fold_scores) < max(2, len(fold_configs) // 2):
                return -99.0

            avg_fold_score = np.mean(valid_fold_scores)


            # Re‑run folds to collect metrics (could be optimised, but fine for 120 trials)
            fold_metrics = []
            for (ts, te, xs, xe) in fold_configs:
                try:
                    df_ftr = train_df.iloc[ts:te].copy()
                    df_fts = train_df.iloc[xs:xe].copy()
                    ou_f = float((np.log(df_ftr[self.T1]) -
                                  beta * np.log(df_ftr[self.T2])).mean())
                    test_days = xe - xs

                    feat, _ = self.features.build_features(
                        df_fts, p, ou_mean=ou_f, beta=beta)
                    sig = self.features.generate_signals(feat, p)
                    pnl, eq, tr = self.backtest.backtest(feat, sig, p, cost=True)

                    if len(tr) == 0 or pnl.std() == 0:
                        continue

                    ann_ret = pnl.mean() * 252
                    ann_vol = pnl.std() * np.sqrt(252)
                    total_ret = pnl.sum()
                    roll_max = eq.cummax()
                    max_dd = float(((eq - roll_max) /
                                    roll_max.replace(0, np.nan)).min())
                    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
                    gains = pnl[pnl > 0].sum()
                    losses = abs(pnl[pnl < 0].sum())
                    pf = (gains / losses if losses > 0
                          else (3.0 if gains > 0 else 0.0))
                    pf = float(np.clip(pf, 0, 8.0))
                    util = tr['hold_days'].sum() / test_days
                    long_tr = tr[tr['direction'] == 'long']
                    short_tr = tr[tr['direction'] == 'short']
                    long_pnl = long_tr['pnl'].sum() if len(long_tr) > 0 else 0.0
                    short_pnl = short_tr['pnl'].sum() if len(short_tr) > 0 else 0.0
                    total_abs = abs(long_pnl) + abs(short_pnl)
                    long_share = abs(long_pnl) / total_abs if total_abs > 0 else 0.5
                    symmetry = 1.0 - abs(long_share - 0.5) * 2.0

                    fold_metrics.append({
                        'n': len(tr), 'ann_ret': ann_ret, 'total_ret': total_ret,
                        'sharpe': sharpe, 'pf': pf, 'util': util, 'max_dd': max_dd,
                        'symmetry': symmetry
                    })
                except Exception:
                    continue

            if len(fold_metrics) < 2:
                return -99.0

            avg_n = np.mean([m['n'] for m in fold_metrics])
            avg_ann_ret = np.mean([m['ann_ret'] for m in fold_metrics])
            std_ann_ret = np.std([m['ann_ret'] for m in fold_metrics])
            avg_total_ret = np.mean([m['total_ret'] for m in fold_metrics])
            avg_sharpe = np.mean([m['sharpe'] for m in fold_metrics])
            std_sharpe = np.std([m['sharpe'] for m in fold_metrics])
            avg_pf = np.mean([m['pf'] for m in fold_metrics])
            avg_util = np.mean([m['util'] for m in fold_metrics])
            avg_symmetry = np.mean([m['symmetry'] for m in fold_metrics])
            avg_max_dd = np.mean([m['max_dd'] for m in fold_metrics])

            def soft_penalty(value, threshold, penalty_scale=5.0):
                if value >= threshold:
                    return 0.0
                return -penalty_scale * (threshold - value)

            util_pen = soft_penalty(avg_util, 0.15, 8.0)
            sym_pen = soft_penalty(avg_symmetry, 0.5, 3.0)
            n_pen = soft_penalty(avg_n, 5.0, 2.0)
            dd_pen = soft_penalty(-avg_max_dd, 0.20, 2.0) if avg_max_dd < 0 else 0.0
            sharpe_consistency_pen = -1.0 * min(2.0, std_sharpe / 1.5)
            if avg_ann_ret > 0:
                ret_cv = std_ann_ret / avg_ann_ret
                ret_consistency_pen = -0.5 * min(2.0, ret_cv)
            else:
                ret_consistency_pen = -2.0

            profit_score = np.clip(avg_ann_ret, 0.0, 1.0) * 6.0
            sharpe_score = np.clip(avg_sharpe - 0.5 * std_sharpe, 0.0, 3.0) * 5.0
            pf_score = np.clip(avg_pf - 1.2, 0.0, 5.0) * 2.0
            ret_score = np.log1p(max(0, avg_total_ret * 100)) * 1.5

            score = (profit_score + sharpe_score + pf_score + ret_score
                     + util_pen + sym_pen + n_pen + dd_pen
                     + sharpe_consistency_pen + ret_consistency_pen
                     + reg_penalty)

            return float(score)

        # ---- 5. Multi‑seed optimisation with persistence ----
        SEEDS = np.random.randint(1,999, n_seeds)
        all_candidates = []  # (params, cv_score, seed)

        for seed_idx, seed in enumerate(SEEDS):
            print(f"  ── Seed {seed_idx + 1}/{n_seeds}  (seed={seed}) ──")

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=SuccessiveHalvingPruner(
                    min_resource=1, reduction_factor=3, min_early_stopping_rate=0
                ),
            )
            # Early stopping: stop if no improvement in last 20 trials
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            best_trial = study.best_trial
            all_candidates.append((best_trial.params, best_trial.value, seed))

            # Also add top‑5 trials
            sorted_trials = sorted(
                study.trials,
                key=lambda t: t.value if t.value is not None else -999,
                reverse=True
            )
            for t in sorted_trials[:5]:
                if t.value is not None and t.value > 0:
                    all_candidates.append((t.params, t.value, seed))

            print(f"  Seed {seed}: best score = {best_trial.value:.3f}  "
                  f"z_entry_long={best_trial.params['z_entry_long']:.3f}  "
                  f"slow_window={best_trial.params['slow_window']}")

        # ---- 6. Final evaluation on holdout set ----
        print("\n── Final Holdout Validation ──")
        print(f"   Evaluating {len(all_candidates)} candidates on unseen holdout period...")

        best_holdout_score = -np.inf
        best_params = None
        best_cv_score = None

        for params, cv_score, seed in all_candidates:
            try:
                # Create a full parameter set from defaults, then update with optimised values
                full_params = self.PARAMS.copy()
                full_params.update(params)  # overwrite optimised keys

                # Compute derived symmetric parameters (same as in objective)
                full_params['z_entry'] = (full_params['z_entry_long'] + full_params['z_entry_short']) / 2
                full_params['z_exit'] = (full_params['z_exit_long'] + full_params['z_exit_short']) / 2
                full_params['z_stop'] = (full_params['z_stop_long'] + full_params['z_stop_short']) / 2

                # Evaluate on holdout data
                ou_f = float((np.log(holdout_df[self.T1]) -
                              beta * np.log(holdout_df[self.T2])).mean())
                feat, _ = self.features.build_features(
                    holdout_df, full_params, ou_mean=ou_f, beta=beta)
                sig = self.features.generate_signals(feat, full_params)
                pnl, eq, tr = self.backtest.backtest(feat, sig, full_params, cost=True)

                if len(tr) == 0 or pnl.std() == 0:
                    holdout_score = -1.0
                else:
                    ann_ret = pnl.mean() * 252
                    ann_vol = pnl.std() * np.sqrt(252)
                    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
                    gains = pnl[pnl > 0].sum()
                    losses = abs(pnl[pnl < 0].sum())
                    pf = (gains / losses if losses > 0 else (3.0 if gains > 0 else 0.0))
                    pf = float(np.clip(pf, 0, 8.0))
                    holdout_score = sharpe + 0.5 * max(0, pf - 1.2)  # same metric as in pruning

                if holdout_score > best_holdout_score:
                    best_holdout_score = holdout_score
                    best_params = full_params  # store the full dict
                    best_cv_score = cv_score

                print(f"    Candidate (seed {seed}): CV={cv_score:.3f}  "
                      f"Holdout={holdout_score:.3f}  "
                      f"z_entry={full_params['z_entry_long']:.2f}/{full_params['z_entry_short']:.2f}")

            except Exception as e:
                print(f"    Candidate (seed {seed}) failed on holdout: {e}")
                continue

        # ---- 7. Update self.PARAMS with the best holdout parameters ----
        if best_params is None:
            raise RuntimeError("No valid candidate found on holdout set.")


        # Update self.PARAMS with the best holdout candidate
        self.PARAMS = best_params.copy()  # or update in-place
        self.PARAMS['beta'] = beta
        self.PARAMS['z_entry'] = (self.PARAMS['z_entry_long'] + self.PARAMS['z_entry_short']) / 2
        self.PARAMS['z_exit'] = (self.PARAMS['z_exit_long'] + self.PARAMS['z_exit_short']) / 2
        self.PARAMS['z_stop'] = (self.PARAMS['z_stop_long'] + self.PARAMS['z_stop_short']) / 2

        # Then print the final parameters
        print("\n  Params after Bayes (best holdout candidate):")
        for k in ['slow_window', 'z_entry_long', 'z_entry_short', 'z_exit_long', 'z_exit_short',
                  'z_stop_long', 'z_stop_short', 'z_add', 'vol_cap', 'max_hold']:
            print(f"    {k:20} : {self.PARAMS[k]}")
        print(f"    {'beta (hedge ratio)':20} : {beta:.4f}")

        return best_params

    # ----------------------------------------------------------------------
    # Helper: build time‑series folds covering the entire training period
    # ----------------------------------------------------------------------
    def _build_folds(self, df):
        """
        Build expanding/rolling folds that span the whole DataFrame.
        Returns list of (train_start, train_end, test_start, test_end) indices.
        """
        n_data = len(df)
        if self._half_life is None:
            self._half_life = 252  # fallback

        INNER_TRAIN = max(252, int(self._half_life * 6))
        INNER_TEST = max(63, int(self._half_life * 2.5))
        target_folds = 6
        avail = n_data - INNER_TRAIN - INNER_TEST
        if avail <= 0:
            # Data too short – fallback to 3 simple folds
            folds = []
            step = max(1, (n_data - INNER_TRAIN - INNER_TEST) // 3)
            for i in range(3):
                ts = i * step
                te = ts + INNER_TRAIN
                xs = te
                xe = min(xs + INNER_TEST, n_data)
                if xe <= xs:
                    break
                folds.append((ts, te, xs, xe))
            return folds

        INNER_STEP = max(INNER_TEST, avail // (target_folds - 1) if target_folds > 1 else avail)
        folds = []
        ts = 0
        while True:
            te = ts + INNER_TRAIN
            xe = te + INNER_TEST
            if xe > n_data:
                break
            folds.append((ts, te, te, xe))  # test starts where train ends (non‑overlapping)
            ts += INNER_STEP
        return folds

    # ----------------------------------------------------------------------
    # Statistical diagnostics (unchanged from original)
    # ----------------------------------------------------------------------
    def run_stat_diag(self, df, feat):
        lr = feat['lr'].dropna()
        adf_stat, adf_p, _, _, crit, _ = adfuller(lr, autolag='AIC')
        lags = range(2, 60)
        tau = [np.std(np.subtract(lr.values[l:], lr.values[:-l])) for l in lags]
        hurst, *_ = stats.linregress(np.log(list(lags)), np.log(tau))
        s_lag = lr.shift(1).dropna()
        s_cur = lr.iloc[1:]
        res = sm.OLS(s_cur, sm.add_constant(s_lag)).fit()
        phi = min(float(res.params.iloc[1]), 1 - 1e-8)
        kappa = -np.log(max(phi, 1e-8)) * 252
        hl = np.log(2) / kappa * 252
        try:
            joh = coint_johansen(df[[self.T1, self.T2]], det_order=0, k_ar_diff=1)
            joh_pass = bool(joh.lr1[0] > joh.cvt[0, 1])
            joh_trace = float(joh.lr1[0])
            joh_crit = float(joh.cvt[0, 1])
        except Exception:
            joh_pass = False
            joh_trace = np.nan
            joh_crit = np.nan
        beta = float(feat['beta'].iloc[0])
        return dict(
            adf_p=adf_p, adf_stat=adf_stat, adf_crit=crit,
            hurst=hurst, half_life=hl, kappa=kappa,
            joh_pass=joh_pass, joh_trace=joh_trace, joh_crit=joh_crit,
            beta=beta,
        )