"""
Per-pair slow_window / medium_window optimisation.

Why this exists
----------------
multi_pair_backtest.py currently reuses ONE global (slow_window,
medium_window) — tuned on DABUR/HINDUNILVR — across every pair. Different
spreads mean-revert at different speeds, so a window size that's right for
one pair is probably too fast or too slow for another. This script finds a
per-pair slow_window (medium_window is locked to the current ratio) without
touching entry/exit/stop, which stay at their intuitive, untuned values —
mirrors the project's existing stance that those shouldn't be over-fit
per pair (see diagnosis.py's Bayesian search, which has been unreliable).

Half-life vs. window-search beta — two different betas, two different jobs
-----------------------------------------------------------------------------
The window grid is centred on each pair's OU half-life, but the half-life
must NOT be measured off the same Kalman-filtered spread used for trading.
The Kalman filter models beta as a random walk and re-fits it every bar to
minimize that bar's residual — so when the spread genuinely drifts from its
mean, part of that drift gets absorbed into a beta update rather than
showing up as spread autocorrelation. AR(1) on that residual spread is
therefore biased toward shorter, noisier half-lives that reflect the
filter's responsiveness as much as the pair's real reversion speed.

Fix: estimate beta once with static full-sample OLS (stable, deterministic,
the same method diagnosis.py already uses for its global half-life), build
the spread with THAT beta purely to measure half-life. The Kalman beta is
still used for every actual feature/signal/backtest computation below —
this script only swaps in the static beta for the one-line half-life calc.

Evaluation
----------
Each candidate slow_window is scored via walk-forward CV using
diagnosis.py's own fold construction (expanding train, half-life-sized
purge gap, non-overlapping test windows) — the same anti-overfitting
scaffolding already trusted elsewhere in this project. Score =
mean(fold Sharpe) / (1 + std(fold Sharpe)), so a window that's merely
lucky in one fold can't beat one that's modestly good everywhere.

Search: Optuna TPE (single parameter, narrow pre-bounded range, fixed
seed) instead of a fixed grid — samples adaptively within [0.75, 1.25]xHL
rather than only the grid's handful of preset fractions.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import optuna

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from features import features
from backtest import backtest
from diagnosis import diagnosis
from data import data_loader
from params import PARAMS

optuna.logging.set_verbosity(optuna.logging.WARNING)

PAIRS = [
    ('DABUR.NS',      'HINDUNILVR.NS'),
    ('HAL.NS',        'BDL.NS'),
    ('SHREECEM.NS',   'HEIDELBERG.NS'),
    ('NHPC.NS',       'POWERGRID.NS'),
    ('BAJFINANCE.NS', 'KOTAKBANK.NS'),
    ('HDFCBANK.NS',   'KOTAKBANK.NS'),
    ('OIL.NS',        'ONGC.NS'),
]

START              = PARAMS['start']
KALMAN_DELTA       = 1e-6
KALMAN_INIT_WINDOW = 1000

# slow_window search range = [RANGE_FRACTIONS[0], RANGE_FRACTIONS[1]] x HL.
RANGE_FRACTIONS = (0.75, 1.25)
MIN_SLOW_WINDOW = 15
SLOW_MEDIUM_RATIO = PARAMS['medium_window'] / PARAMS['slow_window']  # preserve current 32/55 relationship

N_TRIALS  = 300
TPE_SEED  = 42  # fixed for reproducibility — single narrow parameter, no multi-seed consensus needed

CAPITAL  = 200_000
LEVERAGE = 4

# ─────────────────────────────────────────────
#  DATA LOADING (reuses data.py's data_loader — cache-first, see
#  download_data.py / data.py's price_data/ cache)
# ─────────────────────────────────────────────
_loader = data_loader()


def load_pair(t1, t2, start):
    _loader.T1, _loader.T2, _loader.START = t1, t2, start
    return _loader.load_data()


# ─────────────────────────────────────────────
#  STATIC-BETA HALF-LIFE  (window sizing only — never used for trading)
# ─────────────────────────────────────────────
def static_half_life(df, t1, t2):
    beta = features.estimate_hedge_ratio(df, t1, t2, lookback=None)
    spread = np.log(df[t1]) - beta * np.log(df[t2])

    s_lag = spread.shift(1).dropna()
    s_cur = spread.iloc[1:]
    res   = sm.OLS(s_cur, sm.add_constant(s_lag)).fit()
    phi   = float(np.clip(res.params.iloc[1], 1e-8, 1 - 1e-8))
    kappa = -np.log(phi) * 252
    hl    = np.log(2) / kappa * 252
    return hl, beta


# ─────────────────────────────────────────────
#  PER-CANDIDATE WALK-FORWARD EVALUATION
# ─────────────────────────────────────────────
def evaluate_window(df_valid, kalman_beta, t1, t2, slow_window, medium_window,
                     folds, feature_builder, backtest_engine):
    p = PARAMS.copy()
    p['T1'], p['T2'] = t1, t2
    p['slow_window']   = slow_window
    p['medium_window']  = medium_window
    feature_builder.T1, feature_builder.T2 = t1, t2

    warmup = int(max(medium_window * 3,
                      p.get('vol_z_window', p.get('vol_window', 60) * 2) * 2,
                      60))

    fold_results = []
    for (tr_s, tr_e, te_s, te_e) in folds:
        try:
            lr_train = (np.log(df_valid[t1].iloc[tr_s:tr_e]) -
                        kalman_beta.iloc[tr_s:tr_e] * np.log(df_valid[t2].iloc[tr_s:tr_e]))
            ou_mean = float(lr_train.mean())

            ctx_s  = max(0, te_s - warmup)
            df_ctx = df_valid.iloc[ctx_s:te_e]
            beta_ctx = kalman_beta.iloc[ctx_s:te_e]

            feat_ctx, _ = feature_builder.build_features(df_ctx, p, ou_mean=ou_mean, beta=beta_ctx)
            sig_ctx      = feature_builder.generate_signals(feat_ctx, p)

            te_index = df_valid.index[te_s:te_e]
            feat = feat_ctx.loc[te_index]
            sig  = sig_ctx.loc[te_index]

            pnl, eq, tr = backtest_engine.backtest(feat, sig, p, cost=True)

            n_tr = len(tr)
            if n_tr == 0 or pnl.std() == 0:
                fold_results.append(None)
                continue

            ann_ret = pnl.mean() * 252
            ann_vol = pnl.std() * np.sqrt(252)
            sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

            fold_results.append(dict(n=n_tr, sharpe=sharpe, weight=np.sqrt(max(n_tr, 1))))
        except Exception:
            fold_results.append(None)

    valid = [f for f in fold_results if f is not None]
    if len(valid) < max(2, len(folds) // 2):
        return dict(score=-99.0, avg_n=0.0, avg_sharpe=float('nan'), std_sharpe=float('nan'), n_valid=len(valid))

    total_w = sum(f['weight'] for f in valid)
    avg_n      = sum(f['n'] * f['weight'] for f in valid) / total_w
    avg_sharpe = sum(f['sharpe'] * f['weight'] for f in valid) / total_w
    var_sharpe = sum(f['weight'] * (f['sharpe'] - avg_sharpe) ** 2 for f in valid) / total_w
    std_sharpe = float(np.sqrt(max(var_sharpe, 0.0)))
    cv_sharpe  = avg_sharpe / (1.0 + std_sharpe)

    return dict(score=cv_sharpe, avg_n=avg_n, avg_sharpe=avg_sharpe,
                std_sharpe=std_sharpe, n_valid=len(valid))


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def optimize_pair(t1, t2, diag, feature_builder, backtest_engine):
    name = f"{t1.split('.')[0]}/{t2.split('.')[0]}"
    df = load_pair(t1, t2, START)

    if len(df) < KALMAN_INIT_WINDOW + 200:
        print(f"  {name:24s}: skipped — only {len(df)} bars (need > {KALMAN_INIT_WINDOW + 200})")
        return None

    HL, static_beta = static_half_life(df, t1, t2)

    feature_builder.T1, feature_builder.T2 = t1, t2
    kalman_beta = feature_builder.kalman_hedge_ratio(
        df, t1, t2, delta=KALMAN_DELTA, init_window=KALMAN_INIT_WINDOW)

    valid_idx   = kalman_beta.dropna().index
    df_valid    = df.loc[valid_idx]
    kalman_beta = kalman_beta.loc[valid_idx]

    diag._half_life = HL
    folds = diag._build_folds(df_valid)

    lo = max(MIN_SLOW_WINDOW, int(round(HL * RANGE_FRACTIONS[0])))
    hi = max(lo + 1, int(round(HL * RANGE_FRACTIONS[1])))

    print(f"\n  {name}  (static beta={static_beta:.3f}, HL={HL:.1f}d, "
          f"{len(folds)} folds, slow_window range=[{lo}, {hi}], {N_TRIALS} TPE trials)")

    rows = []

    def objective(trial):
        sw = trial.suggest_int('slow_window', lo, hi)
        mw = max(5, int(round(sw * SLOW_MEDIUM_RATIO)))
        res = evaluate_window(df_valid, kalman_beta, t1, t2, sw, mw,
                               folds, feature_builder, backtest_engine)
        rows.append(dict(slow_window=sw, medium_window=mw, **res))
        #print(f"    slow={sw:>4d}  medium={mw:>4d}  "
              #f"score={res['score']:>7.3f}  avg_sharpe={res['avg_sharpe']:>+6.2f}"
              #f"±{res['std_sharpe']:.2f}  avg_n/fold={res['avg_n']:>5.1f}  "
              #f"valid_folds={res['n_valid']}/{len(folds)}")
        return res['score']

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=TPE_SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    best = max(rows, key=lambda r: r['score'])
    print(f"    -> best: slow_window={best['slow_window']}  medium_window={best['medium_window']}  "
          f"score={best['score']:.3f}")

    return dict(pair=name, T1=t1, T2=t2, half_life=HL,
                slow_window=best['slow_window'], medium_window=best['medium_window'],
                score=best['score'], trials=rows)


if __name__ == '__main__':
    diag             = diagnosis()
    feature_builder  = features()
    backtest_engine  = backtest(trade_capital=CAPITAL, leverage=LEVERAGE)

    print("═" * 78)
    print("  PER-PAIR WINDOW OPTIMISATION  (slow_window centred on static-beta half-life)")
    print("═" * 78)

    results = []
    for t1, t2 in PAIRS:
        r = optimize_pair(t1, t2, diag, feature_builder, backtest_engine)
        if r is not None:
            results.append(r)

    print("\n" + "═" * 78)
    print("  SUMMARY")
    print("═" * 78)
    print(f"  {'Pair':<24}{'HL(d)':>8}{'slow_window':>14}{'medium_window':>16}{'CV-Sharpe':>12}")
    for r in results:
        print(f"  {r['pair']:<24}{r['half_life']:>8.1f}{r['slow_window']:>14d}"
              f"{r['medium_window']:>16d}{r['score']:>12.3f}")

    print("\n  Per-pair PARAMS overrides:")
    for r in results:
        print(f"    '{r['pair']}': dict(slow_window={r['slow_window']}, "
              f"medium_window={r['medium_window']}),")
