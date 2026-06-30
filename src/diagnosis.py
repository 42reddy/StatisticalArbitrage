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
        fold.  Test windows are non-overlapping and anchored to half-life so
        each test period contains 2–3 full OU cycles.  A purge gap (≥ one
        half-life, a proxy for average holding period) separates train end
        from test start in every fold, so the rolling-window features used
        in the last training bars and the first test bars don't share
        overlapping lookback information.

    Per-fold β and warm-up
        β is re-estimated from each fold's OWN training window (not a single
        global β fit on the full dataset, which would leak every fold's test
        period into the hedge ratio used to score it). The z-score
        normalizer (rolling mean/vol) is warmed up using the train-tail
        history immediately before the test window — never the test
        window's own data — then those warm-up rows are dropped before
        scoring. This avoids both lookahead and a noisy cold-start.

    Objective scoring
        The score IS CV-Sharpe = avg_sharpe / (1 + std_sharpe), clipped and
        scaled. Small, capped bonuses for annualised return and trade-level
        profit factor add at most ~20% on top — they can break ties, they
        can't override Sharpe. Calmar, Sortino, trade-frequency, utilisation
        scoring, and the long/short win-rate-gap penalty were removed: they
        were tunable knobs Optuna could trade off against Sharpe, and that
        trade-off space is exactly what let the optimiser overfit train-fold
        noise. Where they still matter (drawdown, utilisation, long/short
        $ balance), they're enforced as hard floors instead — violate one
        and the trial is rejected outright, with no scored alternative path.

        Fold weighting: folds with more trades get higher weight (sqrt(n)).
        Noisy folds (few trades) are down-weighted rather than excluded.

    Consensus
        Median of top-K trials per seed (not just the single best).
        This is more robust than best-only: the single best trial often
        exploits a lucky random seed draw.

    Symmetric entry/exit/stop
        Long and short legs share one z_entry / z_exit / z_stop each
        (previously independent _long/_short params) — there's no empirical
        basis for assuming the two sides of a mean-reverting spread should
        behave differently, and the asymmetric version just doubled the
        search space.

    Hard floors
        Return continuous degradation below each floor, not a fixed constant.
        Optuna's surrogate model can build a gradient across the boundary
        and steer future trials away from bad regions efficiently.

    Regularisation
        Bounded to [-0.2, 0] so it only acts as a tie-breaker between
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
        self._opt_importance = {}

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
        self.beta = self.features.estimate_hedge_ratio(df, self.T1, self.T2, lookback=None)

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
        seeds          = np.random.randint(1, 1000, size=n_seeds)
        all_top_params = []   # flat list of top_k × n_seeds param dicts
        all_best_scores= []   # best score per seed (for convergence check)
        all_importances= []   # per-seed {param: importance} dicts

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
                  f"|  z_entry={study.best_params['z_entry']:.3f}"
                  f"| z_exit={study.best_params['z_exit']:.3f}"
                  f"| z_stop={study.best_params['z_stop']:.3f}"
                  f"  slow_window={study.best_params['slow_window']}")

            # ── Top-K trials for this seed (not just the single best) ─────
            self._print_top_trials(top_trials)

            # ── Per-seed parameter importance (fANOVA over all trials) ────
            imp = self._param_importances(study)
            if imp:
                all_importances.append(imp)
            print()

        # ── 4. Consensus (median of all top_k × n_seeds param sets) ──────
        consensus, stability = self._build_consensus(all_top_params)

        # ── 4b. Aggregate parameter importance across seeds ──────────────
        importance = self._build_importance(all_importances)
        self._opt_importance = importance

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
            'seeds':      seeds,
            'scores':     all_best_scores,
            'stability':  stability,
            'consensus':  consensus,
            'score_cv':   score_cv,
            'importance': importance,
        }
        return consensus, stability

    # ══════════════════════════════════════════════════════════════════════
    # OBJECTIVE
    # ══════════════════════════════════════════════════════════════════════

    def _objective(self, trial, df, folds):
        """
        Optuna objective.  Returns a scalar score ∈ (-∞, ~16].

        Scoring philosophy
        ------------------
        The score is CV-Sharpe (cross-fold mean / (1 + std)), full stop,
        plus small bonuses (annualised return, trade-level profit factor)
        capped at under ~20% of the primary term. Drawdown, utilisation,
        and long/short $ balance are enforced as hard floors, not scored
        — a violated floor rejects the trial outright rather than letting
        Optuna trade it off against Sharpe.

        Fold weighting: folds with more trades are weighted more heavily via
        sqrt(n_trades).  This down-weights noisy short folds without
        hard-excluding them (which would give Optuna no gradient signal).

        Per-fold β is estimated from that fold's training window only, and
        the rolling z-score normalizer is warmed up using train-tail
        history (dropped before scoring) rather than the test window's
        own data — see the class docstring for why.
        """
        p  = self.PARAMS.copy()
        HL = self._half_life

        # ── Search space ──────────────────────────────────────────────────
        sw_min = max(15, int(HL * 0.7))
        sw_max = max(sw_min + 20, int(HL * 1.2))
        p['slow_window']    = trial.suggest_int(  'slow_window',    sw_min,  sw_max)

        # Keep medium fixed to 1.5x slow to reduce search noise.
        p['medium_window']  = int(round(p['slow_window'] * 0.66))

        # Symmetric entry/exit/stop — no empirical evidence long and short
        # legs of a mean-reverting spread should behave differently, and
        # the asymmetric version doubled the param count for no benefit.
        p['z_entry']  = trial.suggest_float('z_entry', 1.0,  2.5)
        p['z_exit']   = trial.suggest_float('z_exit',  0.05, 0.40)
        p['z_stop']   = trial.suggest_float('z_stop',  3.5,  4.0)
        p['z_entry_long'] = p['z_entry_short'] = p['z_entry']
        p['z_exit_long']  = p['z_exit_short']  = p['z_exit']
        p['z_stop_long']  = p['z_stop_short']  = p['z_stop']

        # ── Hard structural constraints (continuous degradation) ──────────
        # Return a score that degrades proportionally to the violation so
        # Optuna's surrogate can build a gradient at the boundary.
        def _prune(violation, base=-10.0, scale=20.0):
            # violation > 0 means constraint is violated by that amount
            if violation > 0:
                return base - scale * violation
            return None

        checks = [
            _prune(1.0 - p['z_entry']),                           # entry >= 1.0
            _prune(p['z_exit'] - p['z_entry'] * 0.45),
        ]
        for r in checks:
            if r is not None:
                return float(r)

        # ── Regularisation  ∈ [-0.20, 0] ─────────────────────────────────
        # Small enough to only break ties; never overrides real signal.
        def _reg(val, lo, hi, w=0.04):
            mid  = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            return -w * ((val - mid) / half) ** 2

        reg = (
            _reg(p['z_entry'], 1.0,  3.0,  w=0.04) +
            _reg(p['z_exit'],  0.05, 0.40, w=0.02)
        )  # reg ∈ [-0.20, 0]

        # ── Fold evaluation ───────────────────────────────────────────────
        fold_results = []

        # Warm-up so the rolling z-score normalizer isn't cold-started on
        # the test fold itself — it gets history from the train tail
        # (causal, pre-gap, never the test fold's own data) instead.
        warmup = int(max(p['medium_window'] * 3,
                          p.get('vol_z_window', p.get('vol_window', 60) * 2) * 2,
                          60))

        for (tr_s, tr_e, te_s, te_e) in folds:
            try:
                df_tr    = df.iloc[tr_s:tr_e]
                df_te    = df.iloc[te_s:te_e]
                test_days = te_e - te_s

                # β estimated on THIS fold's train window only — using the
                # globally-fit β here would leak future-fold information
                # (including this fold's own test window) into every fold.
                beta = self.features.estimate_hedge_ratio(
                    df_tr, self.T1, self.T2, lookback=None)

                ou_mean = float(
                    (np.log(df_tr[self.T1]) -
                     beta * np.log(df_tr[self.T2])).mean())

                ctx_s        = max(0, te_s - warmup)
                df_ctx       = df.iloc[ctx_s:te_e]
                feat_ctx, _  = self.features.build_features(
                    df_ctx, p, ou_mean=ou_mean, beta=beta)
                sig_ctx      = self.features.generate_signals(feat_ctx, p)

                # Drop warm-up rows — scoring/backtest only ever sees the
                # true test window, warm-up only seeded the rolling stats.
                feat         = feat_ctx.loc[df_te.index]
                sig          = sig_ctx.loc[df_te.index]
                pnl, eq, tr  = self.backtest.backtest(feat, sig, p, cost=True)

                n_tr = len(tr)
                if n_tr == 0 or pnl.std() == 0:
                    fold_results.append(None)
                    continue

                # ── Per-fold metrics ──────────────────────────────────────
                ann_ret  = pnl.mean() * 252
                ann_vol  = pnl.std()  * np.sqrt(252)
                sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0

                # Drawdown (fixed-base: loss vs starting equity) — used only
                # as a hard floor below, not as a scored component.
                start_eq = float(eq.iloc[0])
                dd_fixed = (eq - eq.cummax()) / start_eq   # fraction of starting capital
                max_dd   = float(dd_fixed.min())            # negative

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

                short_wr = float((short_tr['pnl'] > 0).mean()) if short_n > 1 else 0.5

                # Fold weight: sqrt(n_trades) — more trades = more reliable signal
                weight = np.sqrt(max(n_tr, 1))

                fold_results.append(dict(
                    weight       = weight,
                    n            = n_tr,
                    sharpe       = sharpe,
                    tpf          = tpf,
                    ann_ret      = ann_ret,
                    max_dd       = max_dd,
                    stop_rate    = stop_rate,
                    ts_rate      = ts_rate,
                    util         = util,
                    dir_bal      = dir_bal,
                    short_wr     = short_wr,
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
        avg_tpf       = _wmean('tpf')
        avg_ann_ret   = _wmean('ann_ret')
        avg_stop_rate = _wmean('stop_rate')
        avg_ts_rate   = _wmean('ts_rate')
        avg_util      = _wmean('util')
        avg_dir_bal   = _wmean('dir_bal')
        pct_profit    = float(np.mean([f['profitable'] for f in valid]))

        # Worst single-fold drawdown — pessimistic, not averaged, because a
        # max_dd *floor* should fire on the worst case, not get diluted away.
        worst_dd = float(min(f['max_dd'] for f in valid))

        # Short-side profitability across folds
        short_folds   = [f for f in valid if f['short_n'] > 0]
        pct_short_pos = (float(np.mean([f['short_wr'] > 0.5 for f in short_folds]))
                         if short_folds else 0.5)

        # ── CV-Sharpe  (the only stability signal — primary metric) ───────
        # = avg_sharpe / (1 + std_sharpe)
        # A strategy with mean=1.0, std=0.1 scores 0.91.
        # A strategy with mean=1.0, std=1.5 scores 0.40.
        # Consistency is rewarded continuously, not via a binary penalty.
        cv_sharpe = avg_sharpe / (1.0 + std_sharpe)

        # ── Hard floors (continuous degradation, no scored bonus version) ──
        # Everything here used to be a tunable soft penalty (calmar,
        # trade-freq, utilisation, sortino, wr_gap) that Optuna could trade
        # off against Sharpe — that trade-off space is exactly what let the
        # optimiser overfit train-fold noise. Floors instead draw a hard
        # line: violate them and the trial is rejected, full stop.
        MIN_UTIL    = 0.35
        MAX_DD_FLOOR = 0.40   # 30% of starting capital, worst fold
        MIN_DIR_BAL  = 0.30   # long/short P&L balance — directly targets
                               # the train/holdout PF asymmetry this rework
                               # was meant to fix
        if avg_n < 2:
            return -99.0
        if pct_profit < 0.35:
            return -5.0 - 10.0 * (0.35 - pct_profit) / 0.35
        if cv_sharpe < -0.5:
            return -4.0 + 3.0 * (cv_sharpe + 0.5)
        if avg_stop_rate > 0.55:
            return -3.0 - 5.0 * (avg_stop_rate - 0.55)
        if avg_ts_rate > 0.50:
            return -3.0 - 5.0 * (avg_ts_rate - 0.50)
        if avg_util < MIN_UTIL:
            return -4.0 - 15.0 * (MIN_UTIL - avg_util)
        if worst_dd < -MAX_DD_FLOOR:
            return -4.0 - 5.0 * (-MAX_DD_FLOOR - worst_dd)
        if avg_dir_bal < MIN_DIR_BAL:
            return -3.5 - 5.0 * (MIN_DIR_BAL - avg_dir_bal)
        if len(short_folds) >= 2 and pct_short_pos < 0.35:
            return -3.5 - 5.0 * (0.35 - pct_short_pos)

        # ── Score  (Sharpe is the score; everything else is a small,
        #    capped bonus — combined bonuses stay under ~20% of the
        #    primary term) ───────────────────────────────────────────────
        sharpe_score = float(np.clip(cv_sharpe, -2.0, 3.0)) * 5.0   # PRIMARY

        ret_bonus = float(np.clip(avg_ann_ret, -0.30, 0.60)) * 1.0  # ∈ [-0.3, 0.6]
        pf_bonus  = float(np.clip(avg_tpf - 1.0, 0.0, 2.0)) * 0.2   # ∈ [0, 0.4]
        # bonuses ≤ 1.0 vs a max primary term of 15.0 → ≤ ~7% at best case,
        # never more than ~20% near the floor boundary where sharpe_score is small.

        score = sharpe_score + ret_bonus + pf_bonus + reg

        # ── Verbose logging (first 20 trials of first seed) ───────────────
        if trial.number < 20:
            print(
                f"  t{trial.number:>3} | "
                f"n={avg_n:.1f} "
                f"cvSh={cv_sharpe:+.2f} "
                f"sh={avg_sharpe:+.2f}±{std_sharpe:.2f} "
                f"dd={worst_dd:+.2f} "
                f"pf={avg_tpf:.2f} "
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

        A purge gap separates train end from test start in every fold —
        sized to at least one half-life (a proxy for the strategy's average
        holding period, since exits are mean-reversion-driven). Without
        this gap, the rolling-window features computed at the tail of
        train and the head of test share overlapping lookback bars, which
        silently correlates "in-sample" and "out-of-sample" performance.

        Minimum test length is 63 bars (one quarter) regardless of HL.
        Minimum train length is 252 bars (one year) or 6 × HL.

        If fewer than 3 folds can be constructed, a fallback with a smaller
        gap/test length is used to ensure at least 2 folds for meaningful CV.
        """
        n         = len(df)
        HL        = self._half_life if self._half_life else 252
        min_train = max(252, int(HL * 6))
        test_len  = max(189, int(HL * 2.5))
        gap       = max(5, int(round(HL)))   # purge gap >= avg holding period proxy

        def _make_folds(test_len, gap, max_folds=None):
            out  = []
            tr_e = min_train
            while True:
                te_s = tr_e + gap
                te_e = te_s + test_len
                if te_e > n or (max_folds and len(out) >= max_folds):
                    break
                out.append((0, tr_e, te_s, te_e))
                tr_e = te_e   # expanding window absorbs this fold's test+gap
            return out

        folds = _make_folds(test_len, gap)

        # Fallback: fewer than 3 folds — shrink gap/test length for coverage
        if len(folds) < 3:
            folds = _make_folds(max(40, test_len // 2), max(3, gap // 2), max_folds=6)

        return folds

    # ══════════════════════════════════════════════════════════════════════
    # TOP-K TRIALS  (per-seed inspection)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _print_top_trials(top_trials):
        """
        Print the top-K trials of one seed's study — their score and the
        searched parameters — so the consensus isn't a black box built only
        from each seed's single best (often a lucky draw). Seeing the spread
        of high-scoring configs tells you whether the top region is a tight
        basin (trustworthy) or a scatter of unrelated points (fragile).
        """
        if not top_trials:
            print("   (no completed trials)")
            return

        keys = ['z_entry', 'z_exit', 'z_stop', 'slow_window']
        header = "   ".join(f"{k:>11}" for k in keys)
        print(f"   Top-{len(top_trials)} trials:")
        print(f"     {'rank':>4}  {'score':>7}   {header}")
        for rank, t in enumerate(top_trials, 1):
            cells = []
            for k in keys:
                v = t.params.get(k)
                if v is None:
                    cells.append(f"{'—':>11}")
                elif isinstance(v, (int, np.integer)):
                    cells.append(f"{v:>11d}")
                else:
                    cells.append(f"{v:>11.3f}")
            print(f"     {rank:>4}  {t.value:>7.3f}   " + "   ".join(cells))

    # ══════════════════════════════════════════════════════════════════════
    # PARAMETER IMPORTANCE  (fANOVA over each study, averaged across seeds)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _param_importances(study):
        """
        Quantitative importance of each searched parameter for one study,
        via Optuna's fANOVA evaluator. fANOVA decomposes the variance of the
        objective across the search space and attributes each fraction to a
        parameter (and its interactions); the per-parameter values are
        non-negative and sum to 1. Intuitively: "what share of the variation
        in the score is explained by moving this knob?"

        Only COMPLETE trials with finite values are usable, and fANOVA needs
        a handful of them spanning more than one distinct value per param —
        so this returns {} (gracefully skipped) when the study is too sparse.
        """
        try:
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE
                         and t.value is not None and np.isfinite(t.value)]
            if len(completed) < 8:
                return {}
            return optuna.importance.get_param_importances(
                study,
                evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0),
            )
        except Exception:
            return {}

    def _build_importance(self, all_importances):
        """
        Aggregate per-seed importance dicts into a mean ± std ranking.

        Averaging across independent seeds matters: a single study's fANOVA
        can over-attribute to whichever knob that seed's sampler happened to
        explore. A parameter that's consistently important across seeds
        (high mean, low std) is a real driver of the score; one that's
        important in only one seed (high std) is likely seed-specific noise.
        """
        if not all_importances:
            print("\n  Parameter importance : (insufficient trials to compute)")
            return {}

        keys = sorted({k for d in all_importances for k in d})
        importance = {}
        for k in keys:
            vals = np.array([d.get(k, 0.0) for d in all_importances], dtype=float)
            importance[k] = {'mean': float(vals.mean()),
                             'std':  float(vals.std()),
                             'n':    int((vals > 0).sum())}

        ranked = sorted(importance.items(), key=lambda kv: kv[1]['mean'], reverse=True)

        W = 60
        print(f"\n╔{'═'*W}╗")
        print(f"║  PARAMETER IMPORTANCE  (fANOVA, mean over {len(all_importances)} seeds)")
        print(f"║  Share of objective variance explained by each parameter.")
        print(f"╠{'═'*W}╣")
        print(f"  {'Param':<16} {'Importance':>11} {'±Std':>8}   {'':<20}")
        print(f"  {'─'*W}")
        for k, v in ranked:
            bar = '█' * int(round(v['mean'] * 30))
            print(f"  {k:<16} {v['mean']:>10.1%} {v['std']:>8.1%}   {bar}")
        print(f"╚{'═'*W}╝")

        return importance

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

        # Propagate symmetric params to the long/short keys that
        # backtest.py / features.py actually read (kept for compatibility
        # with the asymmetric param interface, but always equal now).
        if 'z_entry' in consensus:
            consensus['z_entry_long']  = consensus['z_entry_short']  = consensus['z_entry']
        if 'z_exit' in consensus:
            consensus['z_exit_long']   = consensus['z_exit_short']   = consensus['z_exit']
        if 'z_stop' in consensus:
            consensus['z_stop_long']   = consensus['z_stop_short']   = consensus['z_stop']

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