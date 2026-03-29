import numpy as np
import pandas as pd
import optuna
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from params import PARAMS
from features import features
from backtest import backtest

optuna.logging.set_verbosity(optuna.logging.WARNING)


class diagnosis:
    """
    Multi-seed Bayesian optimisation for the stat-arb strategy.

    Design priorities (in order)
    -----------------------------
    1. Stability      — parameters must converge across independent seeds
    2. Generalisation — objective rewards OOS consistency, not peak in-sample
    3. Variance       — pessimistic aggregation penalises cross-fold dispersion

    Key design decisions
    --------------------

    Fold construction  (_build_folds)
        Expanding window: training always starts at bar 0 and grows with each
        fold.  This means later folds have more context and the model is never
        penalised for "forgetting" early data.  Test windows are non-overlapping
        and anchored to half-life so each fold contains at least 2–3 full OU
        cycles — the minimum needed for a meaningful signal.

    Objective scoring
        Primary term: CV-Sharpe = avg_sharpe / (1 + std_sharpe)
            Encodes mean AND consistency in one term.  Optuna maximises this
            naturally without needing separate std penalties.

        Secondary terms are normalised to ≈ comparable scale before summing
        so no single metric dominates.  All weights are documented with their
        approximate contribution range.

        Pessimistic aggregation: every aggregate is mean - λ*std where λ > 0.
        This explicitly rewards parameter sets where ALL folds are good.

        Fold weighting: folds with more trades get higher weight (sqrt(n)).
        Noisy folds (few trades) are down-weighted rather than excluded.

    Consensus
        Median of top-K trials per seed (not just the single best).
        This is more robust than best-only: the single best trial often
        exploits a lucky random seed draw.

    Hard floors
        Return continuous degradation below each floor, not a fixed constant.
        Optuna's surrogate model can build a gradient across the boundary
        and steer future trials away from bad regions efficiently.

    Regularisation
        Bounded to [-0.3, 0] so it only acts as a tie-breaker between
        otherwise equal parameter sets.  It does not dominate the score.
    """

    def __init__(self):
        self.T1           = PARAMS['T1']
        self.T2           = PARAMS['T2']
        self.PARAMS       = PARAMS.copy()
        self.features     = features()
        self.backtest     = backtest()
        self._half_life   = None
        self.beta         = None
        self._opt_summary = {}

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    def sensitivity_analysis(
        self,
        df,
        ou_hl        = None,    # kept for API compatibility, not used
        n_trials     = 200,
        n_seeds      = 6,
        top_k        = 5,       # trials per seed used for consensus
        verbose      = True,
    ):
        """
        Run multi-seed Bayesian optimisation and return consensus parameters.

        Parameters
        ----------
        df        : pd.DataFrame  — full training data (prices, T1 and T2)
        n_trials  : int           — Optuna trials per seed
        n_seeds   : int           — independent random seeds
        top_k     : int           — top trials per seed used for consensus
        verbose   : bool

        Returns
        -------
        consensus : dict   — median parameters across seeds × top_k
        stability : dict   — per-parameter CV and stability flags
        """
        W = 64
        print(f"\n╔{'═'*W}╗")
        print(f"║  MULTI-SEED BAYESIAN OPTIMISATION  (Median Consensus)")
        print(f"║  Seeds: {n_seeds}  |  Trials/seed: {n_trials}"
              f"  |  Top-K: {top_k}  |  Total evals: {n_seeds*n_trials}")
        print(f"╠{'═'*W}╣")

        # ── 1. Hedge ratio and half-life (full dataset, deterministic) ────
        self.beta = self.features.estimate_hedge_ratio(
            df, self.T1, self.T2, lookback=None)
        spread     = (np.log(df[self.T1]) - self.beta * np.log(df[self.T2])).dropna()
        s_lag      = spread.shift(1).dropna()
        s_cur      = spread.iloc[1:]
        res        = sm.OLS(s_cur, sm.add_constant(s_lag)).fit()
        phi        = float(np.clip(res.params.iloc[1], 1e-8, 1 - 1e-8))
        kappa      = -np.log(phi) * 252
        self._half_life = np.log(2) / kappa * 252

        print(f"║  β = {self.beta:.4f}  |  OU half-life = {self._half_life:.1f}d")

        # ── 2. Build folds (deterministic, expanding window) ──────────────
        folds = self._build_folds(df)
        n_folds = len(folds)
        print(f"║  Folds: {n_folds}  (expanding train, non-overlapping test)")
        if verbose:
            for i, (_, tr_e, te_s, te_e) in enumerate(folds):
                print(f"║    Fold {i+1}: "
                      f"train →{df.index[tr_e-1].date()}  "
                      f"test {df.index[te_s].date()}→{df.index[te_e-1].date()}"
                      f"  ({te_e-te_s}d)")
        print(f"╚{'═'*W}╝\n")

        # ── 3. Multi-seed optimisation ────────────────────────────────────
        seeds          = [42, 137, 271, 512, 919, 1337, 2048][:n_seeds]
        all_top_params = []   # flat list of top_k × n_seeds param dicts
        all_best_scores= []   # best score per seed (for convergence check)

        for seed_idx, seed in enumerate(seeds):
            print(f"── Seed {seed_idx+1}/{n_seeds}  (seed={seed}) ──")

            study = optuna.create_study(
                direction = 'maximize',
                sampler   = optuna.samplers.TPESampler(
                    seed                    = seed,
                    n_startup_trials        = 30,   # more random exploration
                    multivariate            = True,  # model param correlations
                    constant_liar           = False,
                    warn_independent_sampling=False,
                ),
            )
            study.optimize(
                lambda trial: self._objective(trial, df, folds),
                n_trials          = n_trials,
                show_progress_bar = verbose,
            )

            # Collect top_k trials (not just the best)
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE
                         and t.value is not None]
            completed.sort(key=lambda t: t.value, reverse=True)
            top_trials = completed[:top_k]

            for t in top_trials:
                all_top_params.append(t.params)

            best_score = study.best_value
            all_best_scores.append(best_score)
            print(f"   Best score: {best_score:.3f}  "
                  f"|  z_entry_long={study.best_params['z_entry_long']:.3f}"
                  f"  slow_window={study.best_params['slow_window']}\n")

        # ── 4. Consensus (median of all top_k × n_seeds param sets) ──────
        consensus, stability = self._build_consensus(all_top_params)

        # ── 5. Seed convergence check ─────────────────────────────────────
        scores_arr = np.array(all_best_scores)
        score_cv   = scores_arr.std() / abs(scores_arr.mean()) if scores_arr.mean() != 0 else np.nan
        conv_flag  = '✓ strong' if score_cv < 0.15 else ('~ moderate' if score_cv < 0.30 else '✗ weak')

        print(f"\n  Scores across seeds : {[f'{s:.2f}' for s in all_best_scores]}")
        print(f"  Score mean ± std    : {scores_arr.mean():.3f} ± {scores_arr.std():.3f}")
        print(f"  Seed convergence    : {conv_flag} (CV={score_cv:.3f})")

        # ── 6. Store and return ───────────────────────────────────────────
        self.PARAMS.update(consensus)
        self.PARAMS['beta'] = self.beta
        self._opt_summary = {
            'seeds':     seeds,
            'scores':    all_best_scores,
            'stability': stability,
            'consensus': consensus,
            'score_cv':  score_cv,
        }
        return consensus, stability

    # ══════════════════════════════════════════════════════════════════════
    # OBJECTIVE
    # ══════════════════════════════════════════════════════════════════════

    def _objective(self, trial, df, folds):
        """
        Optuna objective.  Returns a scalar score ∈ (-∞, ~25].

        Scoring philosophy
        ------------------
        Every component is clipped and scaled so its contribution range is
        documented and bounded.  No single metric can dominate.

        The primary signal is CV-Sharpe (cross-fold mean / (1 + std)).
        All other terms are secondary and act as tie-breakers or guards.

        Pessimistic aggregation: mean - λ*std throughout.  This means the
        objective prefers a parameter set with Sharpe [0.8, 0.9, 0.85] over
        one with [1.5, 0.2, 0.6] even if both have the same mean.

        Fold weighting: folds with more trades are weighted more heavily via
        sqrt(n_trades).  This down-weights noisy short folds without
        hard-excluding them (which would give Optuna no gradient signal).
        """
        p  = self.PARAMS.copy()
        HL = self._half_life
        beta = self.beta

        # ── Search space ──────────────────────────────────────────────────
        sw_min = max(15, int(HL * 0.8))
        sw_max = max(sw_min + 20, int(HL * 1.5))
        p['slow_window']    = trial.suggest_int(  'slow_window',    sw_min,  sw_max)

        # Keep medium fixed to 1.5x slow to reduce search noise.
        p['medium_window']  = int(round(p['slow_window'] * 1.5))

        p['z_entry_long']   = trial.suggest_float('z_entry_long',   0.8,  2.2)
        p['z_entry_short']  = trial.suggest_float('z_entry_short',  0.8,  2.2)
        p['z_exit_long']    = trial.suggest_float('z_exit_long',    0.05, 0.40)
        p['z_exit_short']   = trial.suggest_float('z_exit_short',   0.05, 0.40)

        # ── Hard structural constraints (continuous degradation) ──────────
        # Return a score that degrades proportionally to the violation so
        # Optuna's surrogate can build a gradient at the boundary.
        def _prune(violation, base=-10.0, scale=20.0):
            # violation > 0 means constraint is violated by that amount
            if violation > 0:
                return base - scale * violation
            return None

        checks = [
            _prune(p['slow_window']   - p['medium_window'] + 1),   # medium > slow
            _prune(1.0 - p['z_entry_long']),                        # entry_long >= 1.0
            _prune(1.0 - p['z_entry_short']),                       # entry_short >= 1.0
            _prune(p['z_exit_long']   - p['z_entry_long']  * 0.45),
            _prune(p['z_exit_short']  - p['z_entry_short'] * 0.45),
        ]
        for r in checks:
            if r is not None:
                return float(r)

        # ── Regularisation  ∈ [-0.30, 0] ─────────────────────────────────
        # Small enough to only break ties; never overrides real signal.
        def _reg(val, lo, hi, w=0.04):
            mid  = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            return -w * ((val - mid) / half) ** 2

        reg = (
            _reg(p['z_entry_long'],  0.8,  2.2,  w=0.04) +
            _reg(p['z_entry_short'], 0.8,  2.2,  w=0.04) +
            _reg(p['z_exit_long'],   0.05, 0.40, w=0.02) +
            _reg(p['z_exit_short'],  0.05, 0.40, w=0.02)
        )  # reg ∈ [-0.30, 0]

        # ── Fold evaluation ───────────────────────────────────────────────
        fold_results = []

        for (tr_s, tr_e, te_s, te_e) in folds:
            try:
                df_tr    = df.iloc[tr_s:tr_e]
                df_te    = df.iloc[te_s:te_e]
                test_days = te_e - te_s

                ou_mean = float(
                    (np.log(df_tr[self.T1]) -
                     beta * np.log(df_tr[self.T2])).mean())

                feat, _      = self.features.build_features(
                    df_te, p, ou_mean=ou_mean, beta=beta)
                sig          = self.features.generate_signals(feat, p)
                pnl, eq, tr  = self.backtest.backtest(feat, sig, p, cost=True)

                n_tr = len(tr)
                if n_tr == 0 or pnl.std() == 0:
                    fold_results.append(None)
                    continue

                # ── Per-fold metrics ──────────────────────────────────────
                ann_ret  = pnl.mean() * 252
                ann_vol  = pnl.std()  * np.sqrt(252)
                sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0

                # Sortino
                r_down   = pnl[pnl < 0]
                down_vol = r_down.std() * np.sqrt(252) if len(r_down) > 1 else ann_vol
                sortino  = ann_ret / down_vol if down_vol > 0 else 0.0

                # Calmar (fixed-base drawdown: loss vs starting equity)
                start_eq = float(eq.iloc[0])
                dd_fixed = (eq - eq.cummax()) / start_eq   # fraction of starting capital
                max_dd   = float(dd_fixed.min())            # negative
                calmar   = ann_ret / abs(max_dd) if max_dd < -0.005 else np.nan

                # Profit factor (daily pnl)
                gains  = pnl[pnl > 0].sum()
                losses = abs(pnl[pnl < 0].sum())
                pf = float(np.clip(
                    gains / losses if losses > 0 else (3.0 if gains > 0 else 0.0),
                    0, 6.0))

                # Trade-level profit factor
                t_wins = tr.loc[tr['pnl'] > 0, 'pnl_dol'].sum()
                t_loss = abs(tr.loc[tr['pnl'] < 0, 'pnl_dol'].sum())
                tpf = float(np.clip(
                    t_wins / t_loss if t_loss > 0 else (3.0 if t_wins > 0 else 0.0),
                    0, 6.0))

                # Utilisation
                util = float(tr['hold_days'].sum() / max(test_days, 1))

                # Stop and time-stop rates
                stop_rate = float((tr['exit_reason'] == 'stop').mean())
                ts_rate   = float((tr['exit_reason'] == 'time_stop').mean())

                # Direction balance (P&L weighted, not trade-count weighted)
                long_tr  = tr[tr['direction'] == 'long']
                short_tr = tr[tr['direction'] == 'short']
                long_pnl  = float(long_tr['pnl'].sum())  if len(long_tr)  > 0 else 0.0
                short_pnl = float(short_tr['pnl'].sum()) if len(short_tr) > 0 else 0.0
                long_n    = len(long_tr)
                short_n   = len(short_tr)

                if long_n > 0 and short_n > 0:
                    tot_abs   = abs(long_pnl) + abs(short_pnl)
                    lshare    = abs(long_pnl) / tot_abs if tot_abs > 0 else 0.5
                    dir_bal   = 1.0 - abs(lshare - 0.5) * 2.0
                else:
                    dir_bal   = 0.0   # one side absent — heavily penalised

                long_wr  = float((long_tr['pnl']  > 0).mean()) if long_n  > 1 else 0.5
                short_wr = float((short_tr['pnl'] > 0).mean()) if short_n > 1 else 0.5
                short_stop_n = int(((short_tr['exit_reason'] == 'stop').sum())
                                   if short_n > 0 else 0)
                short_stop_rate = short_stop_n / max(short_n, 1)

                # Fold weight: sqrt(n_trades) — more trades = more reliable signal
                weight = np.sqrt(max(n_tr, 1))

                fold_results.append(dict(
                    weight       = weight,
                    n            = n_tr,
                    sharpe       = sharpe,
                    sortino      = sortino,
                    calmar       = calmar,
                    pf           = pf,
                    tpf          = tpf,
                    ann_ret      = ann_ret,
                    stop_rate    = stop_rate,
                    ts_rate      = ts_rate,
                    util         = util,
                    dir_bal      = dir_bal,
                    long_wr      = long_wr,
                    short_wr     = short_wr,
                    short_stop_rate = short_stop_rate,
                    short_n      = short_n,
                    profitable   = ann_ret > 0,
                ))

            except Exception:
                fold_results.append(None)

        valid  = [f for f in fold_results if f is not None]
        n_valid = len(valid)

        # Require at least half of folds to produce valid results
        if n_valid < max(2, len(folds) // 2):
            return -99.0

        # ── Weighted aggregation ──────────────────────────────────────────
        total_w = sum(f['weight'] for f in valid)

        def _wmean(key):
            return sum(f[key] * f['weight'] for f in valid) / total_w

        def _wstd(key):
            mu = _wmean(key)
            var = sum(f['weight'] * (f[key] - mu)**2 for f in valid) / total_w
            return float(np.sqrt(max(var, 0.0)))

        avg_n         = _wmean('n')
        avg_sharpe    = _wmean('sharpe')
        std_sharpe    = _wstd('sharpe')
        avg_sortino   = _wmean('sortino')
        avg_pf        = _wmean('pf')
        avg_tpf       = _wmean('tpf')
        avg_ann_ret   = _wmean('ann_ret')
        avg_stop_rate = _wmean('stop_rate')
        avg_ts_rate   = _wmean('ts_rate')
        avg_util      = _wmean('util')
        avg_dir_bal   = _wmean('dir_bal')
        avg_long_wr   = _wmean('long_wr')
        avg_short_wr  = _wmean('short_wr')
        avg_short_stop= _wmean('short_stop_rate')
        pct_profit    = float(np.mean([f['profitable'] for f in valid]))

        # Calmar: exclude NaN folds, cap outliers, fall back if none exist
        calmar_vals = [f['calmar'] for f in valid
                       if f['calmar'] is not None
                       and not np.isnan(f['calmar'])
                       and -15 < f['calmar'] < 30]
        if len(calmar_vals) >= 2:
            avg_calmar = float(np.mean(calmar_vals))
            std_calmar = float(np.std(calmar_vals))
        elif len(calmar_vals) == 1:
            avg_calmar = float(calmar_vals[0])
            std_calmar = 3.0    # conservative stand-in for unknown variance
        else:
            avg_calmar = 2.0    # no drawdown observed — modestly positive
            std_calmar = 0.0

        # Short-side profitability across folds
        short_folds   = [f for f in valid if f['short_n'] > 0]
        pct_short_pos = (float(np.mean([f['short_wr'] > 0.5 for f in short_folds]))
                         if short_folds else 0.5)

        sharpe_vals = np.array([f['sharpe'] for f in valid], dtype=float)
        p20_sharpe  = float(np.percentile(sharpe_vals, 20)) if len(sharpe_vals) > 0 else -2.0

        # ── CV-Sharpe  (primary stability signal) ─────────────────────────
        # = avg_sharpe / (1 + std_sharpe)
        # A strategy with mean=1.0, std=0.1 scores 0.91.
        # A strategy with mean=1.0, std=1.5 scores 0.40.
        # Consistency is rewarded continuously, not via a binary penalty.
        cv_sharpe = avg_sharpe / (1.0 + std_sharpe)
        robust_sharpe = avg_sharpe - 0.50 * std_sharpe

        # ── Fold consistency bonus ────────────────────────────────────────
        # Explicit reward for strategies where EVERY fold is profitable.
        # pct_profit ∈ [0, 1].  Bonus ∈ [0, 0.5].
        consistency_bonus = float(np.clip((pct_profit - 0.5) * 1.0, 0.0, 0.5))

        # ── Hard floors (continuous degradation) ──────────────────────────
        if avg_n < 2:
            return -99.0
        if pct_profit < 0.35:
            return -5.0 - 10.0 * (0.35 - pct_profit) / 0.35
        if cv_sharpe < -0.5:
            return -4.0 + 3.0 * (cv_sharpe + 0.5)
        if robust_sharpe < -0.25:
            return -3.5 + 2.0 * (robust_sharpe + 0.25)
        if avg_stop_rate > 0.55:
            return -3.0 - 5.0 * (avg_stop_rate - 0.55)
        if avg_ts_rate > 0.50:
            return -3.0 - 5.0 * (avg_ts_rate - 0.50)
        if avg_util < 0.08:
            return -4.0 - 10.0 * (0.08 - avg_util)
        if len(short_folds) >= 2 and pct_short_pos < 0.35:
            return -3.5 - 5.0 * (0.35 - pct_short_pos)

        # ── Soft penalties ────────────────────────────────────────────────
        ts_pen         = -2.0 * max(0.0, avg_ts_rate    - 0.20) ** 0.8
        stop_pen       = -1.5 * max(0.0, avg_stop_rate  - 0.20) ** 0.8
        util_pen       = -2.0 * max(0.0, 0.12 - avg_util)
        short_stop_pen = -1.5 * max(0.0, avg_short_stop - 0.20)
        wr_gap_pen     = -0.8 * max(0.0, avg_long_wr - avg_short_wr - 0.25)
        tail_pen       = -1.2 * max(0.0, 0.20 - p20_sharpe)
        dir_pen        = short_stop_pen + wr_gap_pen

        # ── Score components  (documented contribution ranges) ────────────
        #
        # cv_sharpe_score   ∈ [-5,  7.5]   weight 2.5   PRIMARY
        # calmar_score      ∈ [-1.5, 3.0]  weight 1.5
        # sortino_score     ∈ [-1,   2.0]  weight 1.0
        # pf_score          ∈ [ 0,   2.0]  weight 1.0
        # ret_score         ∈ [-1,   2.5]  weight 1.5   (annualised, length-neutral)
        # dir_bal_score     ∈ [ 0,   1.0]  weight 1.0
        # consistency_bonus ∈ [ 0,   0.5]  bonus
        # util_score        ∈ [ 0,   0.5]  minor
        # freq_score        ∈ [ 0,   0.4]  minor
        # ─────────────────────────────────────────────────────────────────
        # Penalties + reg can remove up to ≈ -4.0.
        # Maximum realistic score ≈ 20.  Minimum before floors ≈ -8.

        cv_sharpe_score = float(np.clip(cv_sharpe, -2.0, 3.0)) * 2.3
        robust_sharpe_score = float(np.clip(robust_sharpe, -2.0, 3.0)) * 0.8

        # Calmar: pessimistic = mean - 0.4*std (capped to avoid std dominance)
        calmar_adj   = float(np.clip(
            avg_calmar - 0.4 * min(std_calmar, 3.0), -1.0, 4.0))
        calmar_score = calmar_adj * 1.5

        sortino_score = float(np.clip(avg_sortino, -1.0, 2.0)) * 1.0

        # Profit factor: normalised — PF=1 scores 0, PF=3 scores 2
        pf_score = float(np.clip(avg_tpf - 1.0, 0.0, 2.0)) * 1.0

        # Return: annualised (length-neutral), log-compressed
        ret_score = float(np.clip(
            np.log1p(np.clip(avg_ann_ret * 5.0, -0.99, 12.0)),
            -1.0, 2.5)) * 1.3

        dir_bal_score = float(np.clip(avg_dir_bal, 0.0, 1.0)) * 1.0

        util_score  = float(np.clip(avg_util, 0.0, 0.40)) / 0.40 * 0.5
        freq_score  = float(np.clip(np.log(max(avg_n, 2) / 2.0), 0.0, 1.0)) * 0.4

        score = (
            cv_sharpe_score
            + robust_sharpe_score
            + calmar_score
            + sortino_score
            + pf_score
            + ret_score
            + dir_bal_score
            + consistency_bonus
            + util_score
            + freq_score
            # Penalties
            + ts_pen
            + stop_pen
            + util_pen
            + dir_pen
            + tail_pen
            # Regularisation
            + reg
        )

        # ── Verbose logging (first 20 trials of first seed) ───────────────
        if trial.number < 20:
            print(
                f"  t{trial.number:>3} | "
                f"n={avg_n:.1f} "
                f"cvSh={cv_sharpe:+.2f} "
                f"sh={avg_sharpe:+.2f}±{std_sharpe:.2f} "
                f"cal={avg_calmar:+.2f} "
                f"pf={avg_tpf:.2f} "
                f"lwr={avg_long_wr*100:.0f}% swr={avg_short_wr*100:.0f}% "
                f"bal={avg_dir_bal:.2f} util={avg_util*100:.0f}% "
                f"stop={avg_stop_rate*100:.0f}% | "
                f"→ {score:+.3f}"
            )

        return float(score)

    # ══════════════════════════════════════════════════════════════════════
    # FOLD CONSTRUCTION  (expanding window, HL-anchored)
    # ══════════════════════════════════════════════════════════════════════

    def _build_folds(self, df):
        """
        Expanding-window folds.  Train always starts at bar 0 and grows.
        Test windows are non-overlapping and sized to ~2.5 half-lives so
        each test period contains 2–3 full OU cycles.

        Minimum test length is 63 bars (one quarter) regardless of HL.
        Minimum train length is 252 bars (one year) or 6 × HL.

        If fewer than 3 folds can be constructed, a fallback with a smaller
        step size is used to ensure at least 2 folds for meaningful CV.
        """
        n        = len(df)
        HL       = self._half_life if self._half_life else 252
        min_train = max(252, int(HL * 6))
        test_len  = max(63,  int(HL * 2.5))
        step      = test_len   # non-overlapping tests

        folds = []
        te_end = min_train + test_len
        while te_end <= n:
            tr_s = 0
            tr_e = te_end - test_len
            te_s = tr_e
            te_e = te_end
            if tr_e >= min_train:           # guard: train window is large enough
                folds.append((tr_s, tr_e, te_s, te_e))
            te_end += step

        # Fallback: fewer than 3 folds — shrink step to get more coverage
        if len(folds) < 3:
            step_fb = max(1, (n - min_train - test_len) // 4)
            folds   = []
            te_end  = min_train + test_len
            while te_end <= n and len(folds) < 6:
                tr_s = 0
                tr_e = te_end - test_len
                te_s = tr_e
                te_e = te_end
                if tr_e >= min_train:
                    folds.append((tr_s, tr_e, te_s, te_e))
                te_end += step_fb

        return folds

    # ══════════════════════════════════════════════════════════════════════
    # CONSENSUS  (median of top_k × n_seeds param sets)
    # ══════════════════════════════════════════════════════════════════════

    def _build_consensus(self, all_params):
        """
        Compute median consensus and stability flags from a flat list of
        parameter dicts (top_k × n_seeds entries).

        Returns
        -------
        consensus : dict   — median value per parameter
        stability : dict   — per-parameter std, CV, flag
        """
        if not all_params:
            return {}, {}

        param_keys = list(all_params[0].keys())
        consensus  = {}
        stability  = {}

        W = 72
        print(f"\n╔{'═'*W}╗")
        print(f"║  MULTI-SEED CONSENSUS  (median of top-{len(all_params)} param sets)")
        print(f"║")
        print(f"  {'Param':<22} {'Min':>7} {'Median':>8} {'Max':>7} "
              f"{'Std':>7} {'CV':>7}  Stability")
        print(f"  {'─'*72}")

        for key in param_keys:
            vals     = np.array([p[key] for p in all_params], dtype=float)
            med      = float(np.median(vals))
            mean_v   = float(np.mean(vals))
            std_v    = float(np.std(vals))
            cv       = std_v / abs(mean_v) if abs(mean_v) > 1e-6 else np.nan
            min_v    = float(np.min(vals))
            max_v    = float(np.max(vals))

            flag = ('✓' if (np.isnan(cv) or cv < 0.15) else
                    '~' if cv < 0.35 else '✗')

            # Round integers back
            sample = all_params[0][key]
            if isinstance(sample, (int, np.integer)):
                med = int(round(med))

            consensus[key] = med
            stability[key] = {'mean': mean_v, 'std': std_v, 'cv': cv,
                               'min': min_v, 'max': max_v, 'flag': flag}

            print(f"  {key:<22} {min_v:>7.3f} {med:>8.3f} {max_v:>7.3f} "
                  f"{std_v:>7.3f} {cv:>7.3f}  {flag}")

        stable_n = sum(1 for v in stability.values() if v['flag'] == '✓')
        total_n  = len(param_keys)
        pct      = stable_n / total_n
        verdict  = ('✓ HIGH'     if pct >= 0.70 else
                    '~ MODERATE' if pct >= 0.40 else
                    '✗ LOW')

        print(f"\n  Stable params : {stable_n} / {total_n}")
        print(f"  {verdict} — {'use consensus' if pct >= 0.4 else 'increase n_trials or check data'}")

        # Keep medium locked to ~1.5x slow_window for consistency with objective.
        if 'slow_window' in consensus:
            consensus['medium_window'] = int(round(consensus['slow_window'] * 1.5))


        print(f"\n  Consensus parameters (median of top params × {len(all_params)} sets):")
        for k, v in consensus.items():
            print(f"    {k:<22} : {v:.4f}" if isinstance(v, float) else
                  f"    {k:<22} : {v}")
        print(f"    {'beta (fixed)':<22} : {self.beta:.4f}")
        print(f"╚{'═'*W}╝")

        return consensus, stability

    # ══════════════════════════════════════════════════════════════════════
    # STATISTICAL DIAGNOSTICS  (unchanged)
    # ══════════════════════════════════════════════════════════════════════

    def run_stat_diag(self, df, feat):
        lr = feat['lr'].dropna()
        adf_stat, adf_p, _, _, crit, _ = adfuller(lr, autolag='AIC')

        lags = range(2, 60)
        tau  = [np.std(np.subtract(lr.values[l:], lr.values[:-l])) for l in lags]
        hurst, *_ = stats.linregress(np.log(list(lags)), np.log(tau))

        s_lag = lr.shift(1).dropna()
        s_cur = lr.iloc[1:]
        res   = sm.OLS(s_cur, sm.add_constant(s_lag)).fit()
        phi   = float(np.clip(res.params.iloc[1], 1e-8, 1 - 1e-8))
        kappa = -np.log(phi) * 252
        hl    = np.log(2) / kappa * 252

        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            joh      = coint_johansen(df[[self.T1, self.T2]], det_order=0, k_ar_diff=1)
            joh_pass  = bool(joh.lr1[0] > joh.cvt[0, 1])
            joh_trace = float(joh.lr1[0])
            joh_crit  = float(joh.cvt[0, 1])
        except Exception:
            joh_pass  = False
            joh_trace = np.nan
            joh_crit  = np.nan

        return dict(
            adf_p=adf_p, adf_stat=adf_stat, adf_crit=crit,
            hurst=hurst, half_life=hl, kappa=kappa,
            joh_pass=joh_pass, joh_trace=joh_trace, joh_crit=joh_crit,
            beta=float(feat['beta'].iloc[0]),
        )