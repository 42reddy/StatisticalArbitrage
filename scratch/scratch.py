"""
beta_estimation.py
==================
Hedge ratio estimation for stat-arb spreads.

Replaces rolling OLS with two superior estimators:

  1. Johansen MLE  — theoretically correct cointegration vector,
                     used for the full-sample / fold-level β.

  2. Kalman Filter — smooth, online β that updates daily without
                     window-boundary jumps; also yields a posterior
                     variance usable as a stability/risk signal.

Both operate on log-prices, same as the original code.
No look-ahead: every quantity at time t uses only data up to t.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ── optional: keep the old OLS estimator for comparison ──────────────────────
from statsmodels.api import OLS, add_constant


# =============================================================================
# 1.  JOHANSEN ESTIMATOR  (replaces OLS for fold-level β)
# =============================================================================

def estimate_beta_johansen(df: pd.DataFrame,
                            T1: str,
                            T2: str,
                            lookback: int | None = None,
                            det_order: int = 0) -> tuple[float, np.ndarray]:
    """
    Estimate the cointegration hedge ratio via Johansen MLE.

    Parameters
    ----------
    df        : DataFrame with at least columns [T1, T2].
    T1        : Name of the *dependent* asset (e.g. 'HAL.NS').
                Spread = log(T1) − β·log(T2).
    T2        : Name of the *hedge* asset  (e.g. 'BDL.NS').
    lookback  : If given, use only the last `lookback` rows.
    det_order : Deterministic term in the VECM.
                  -1 → no constant, 0 → restricted constant (default),
                   1 → unrestricted constant.
                For equity pairs a restricted constant (0) is standard.

    Returns
    -------
    beta      : float  — hedge ratio, normalised so T1 coefficient = 1.
    evec      : ndarray shape (2,) — full eigenvector [1, -beta] for
                reference / diagnostics.

    Notes
    -----
    Johansen is superconsistent (O(T⁻¹) vs OLS O(T^{-½})) and does not
    suffer from the spurious-regression bias of OLS on I(1) levels.
    The eigenvector corresponding to the *largest* eigenvalue is the
    cointegration vector; we normalise it so the T1 coefficient = 1.
    """
    p1 = np.log(df[T1])
    p2 = np.log(df[T2])

    if lookback is not None:
        p1 = p1.iloc[-lookback:]
        p2 = p2.iloc[-lookback:]

    mask = np.isfinite(p1) & np.isfinite(p2)
    p1, p2 = p1[mask], p2[mask]

    if len(p1) < 30:
        raise ValueError(f"Too few clean bars for Johansen: {len(p1)}")

    data = np.column_stack([p1.values, p2.values])   # shape (T, 2)

    # coint_johansen: k_ar_diff=1 → one lag difference (standard for daily)
    result = coint_johansen(data, det_order=det_order, k_ar_diff=1)

    # Eigenvectors are columns of result.evec, ordered by eigenvalue (desc)
    evec = result.evec[:, 0]          # cointegration vector for r=1 relation

    # Normalise: force T1 (first column) coefficient = 1
    evec_norm = evec / evec[0]        # → [1, -beta]
    beta = -float(evec_norm[1])       # positive for a typical long/short pair

    return beta, evec_norm


# =============================================================================
# 2.  KALMAN FILTER ESTIMATOR  (replaces rolling OLS for dynamic β)
# =============================================================================

def kalman_beta(df: pd.DataFrame,
                T1: str,
                T2: str,
                beta_init: float | None = None,
                P_init: float = 1.0,
                Q: float = 1e-5,
                R: float | None = None) -> pd.DataFrame:
    """
    Track the hedge ratio β(t) dynamically using a linear Kalman filter.

    State model   :  β(t) = β(t-1) + w(t),      w ~ N(0, Q)
    Observation   :  log(T1)(t) = α + β(t)·log(T2)(t) + v(t),  v ~ N(0, R)

    The observation noise R is estimated from the data if not supplied
    (variance of the full-sample OLS residual — a reasonable initialisation).

    Parameters
    ----------
    df        : DataFrame with columns [T1, T2].
    T1, T2    : Asset column names.
    beta_init : Starting value for β.  If None, uses full-sample OLS as
                a warm-start (not used in live trading; only for backtesting
                initialisation).
    P_init    : Initial state covariance (uncertainty in beta_init).
                Large value → filter adapts quickly at the start.
    Q         : Process noise — how much β is allowed to drift per day.
                Smaller Q → smoother β, slower to adapt.
                Larger Q → β tracks faster, noisier.
                Rule of thumb: Q = 1e-5 implies β half-life ~ 200 days.
                               Q = 1e-4 implies β half-life ~  60 days.
    R         : Observation noise variance.  If None, estimated from data.

    Returns
    -------
    pd.DataFrame with columns:
        beta        — filtered hedge ratio at each date
        beta_var    — posterior variance P(t|t)  [stability signal]
        beta_std    — sqrt(beta_var)             [± 1σ band width]
        spread      — log(T1) − β(t)·log(T2)    [dynamically hedged spread]
        innovation  — observation innovation y(t) − ŷ(t)
        innov_std   — innovation std dev S(t)    [for z-score normalisation]

    Usage in trading
    ----------------
    Entry gate  :  only enter if beta_std < threshold  (β is well-estimated)
    Drift alert :  flag if |beta[t] - beta[t-20]| > 2 * beta_std[t]
    """
    p1 = np.log(df[T1]).values
    p2 = np.log(df[T2]).values
    idx = df.index
    T = len(p1)

    # ── initialise ────────────────────────────────────────────────────────────
    if beta_init is None:
        # warm-start: OLS on first min(120, T//4) observations
        warm = min(120, T // 4)
        mask = np.isfinite(p1[:warm]) & np.isfinite(p2[:warm])
        if mask.sum() >= 20:
            res = OLS(p1[:warm][mask],
                      add_constant(p2[:warm][mask])).fit()
            beta_init = float(res.params[1])
        else:
            beta_init = 1.0

    if R is None:
        # estimate R from full-sample OLS residual variance
        mask = np.isfinite(p1) & np.isfinite(p2)
        res_full = OLS(p1[mask], add_constant(p2[mask])).fit()
        R = float(np.var(res_full.resid))

    # ── storage ───────────────────────────────────────────────────────────────
    betas      = np.full(T, np.nan)
    beta_vars  = np.full(T, np.nan)
    spreads    = np.full(T, np.nan)
    innovations= np.full(T, np.nan)
    innov_stds = np.full(T, np.nan)

    # ── filter loop ───────────────────────────────────────────────────────────
    beta_t = beta_init
    P_t    = P_init           # posterior variance

    for t in range(T):
        if not (np.isfinite(p1[t]) and np.isfinite(p2[t])):
            betas[t]      = beta_t
            beta_vars[t]  = P_t
            continue

        x_t = p2[t]           # regressor (log BDL)
        y_t = p1[t]           # observation (log HAL)

        # ── Predict step ──────────────────────────────────────────────────────
        # State: β_{t|t-1} = β_{t-1|t-1}   (random walk prior)
        # Cov  : P_{t|t-1} = P_{t-1|t-1} + Q
        P_pred = P_t + Q

        # ── Update step ───────────────────────────────────────────────────────
        # Innovation: y_t - x_t * β_{t|t-1}   (ignore α; absorb into R)
        y_hat  = x_t * beta_t
        innov  = y_t - y_hat

        # Innovation covariance: S = x_t² * P_pred + R
        S      = x_t ** 2 * P_pred + R

        # Kalman gain: K = P_pred * x_t / S
        K      = P_pred * x_t / S

        # Posterior update
        beta_t = beta_t + K * innov
        P_t    = (1 - K * x_t) * P_pred      # Joseph form for stability

        # ── Store ─────────────────────────────────────────────────────────────
        betas[t]       = beta_t
        beta_vars[t]   = P_t
        spreads[t]     = y_t - beta_t * x_t
        innovations[t] = innov
        innov_stds[t]  = np.sqrt(S)

    return pd.DataFrame({
        'beta'      : betas,
        'beta_var'  : beta_vars,
        'beta_std'  : np.sqrt(beta_vars),
        'spread'    : spreads,
        'innovation': innovations,
        'innov_std' : innov_stds,
    }, index=idx)


# =============================================================================
# 3.  STABILITY GATE  (entry filter derived from Kalman posterior)
# =============================================================================

def beta_is_stable(kf_row: pd.Series,
                   kf_history: pd.DataFrame,
                   lookback_days: int = 20,
                   max_std: float = 0.08,
                   max_drift: float = 0.06) -> tuple[bool, str]:
    """
    Return (stable: bool, reason: str) for a single bar's entry decision.

    Conditions (both must pass):
      1. beta_std < max_std       — Kalman is confident in current β
      2. |Δβ over lookback| < max_drift — β has not drifted recently

    Parameters
    ----------
    kf_row      : Single row of the Kalman DataFrame at the candidate entry.
    kf_history  : Full Kalman DataFrame up to and including kf_row.
    lookback_days: Window over which to measure β drift.
    max_std     : Maximum tolerated posterior std dev.
    max_drift   : Maximum tolerated absolute change in β over lookback_days.

    Returns
    -------
    (True, 'ok') if both conditions pass; (False, reason_string) otherwise.
    """
    # Condition 1 — uncertainty
    if kf_row['beta_std'] > max_std:
        return False, f"beta_std={kf_row['beta_std']:.4f} > {max_std}"

    # Condition 2 — drift
    if len(kf_history) >= lookback_days:
        beta_now  = kf_row['beta']
        beta_then = kf_history['beta'].iloc[-lookback_days]
        drift     = abs(beta_now - beta_then)
        if drift > max_drift:
            return False, f"beta drift={drift:.4f} > {max_drift} over {lookback_days}d"

    return True, 'ok'


# =============================================================================
# 4.  ROLLING JOHANSEN  (optional: full fold-level re-estimation)
# =============================================================================

def rolling_beta_johansen(df: pd.DataFrame,
                           T1: str,
                           T2: str,
                           window: int = 120) -> pd.Series:
    """
    Rolling Johansen β — same interface as the original rolling_beta().
    Slower than Kalman but gives a theoretically clean snapshot β
    at each point, useful for diagnostics and fold initialisation.

    window: minimum 60 recommended for Johansen stability.
    """
    betas = np.full(len(df), np.nan)
    for i in range(len(df)):
        if i < window - 1:
            continue
        chunk = df.iloc[i - window + 1: i + 1]
        try:
            beta, _ = estimate_beta_johansen(chunk, T1, T2)
            betas[i] = beta
        except Exception:
            pass
    return pd.Series(betas, index=df.index, name='rolling_beta_johansen')


# =============================================================================
# 5.  RECOMMENDED USAGE  (drop-in replacement for original workflow)
# =============================================================================

def build_spread(df: pd.DataFrame,
                 T1: str,
                 T2: str,
                 johansen_lookback: int = None,
                 kalman_Q: float = 1e-5,
                 kalman_R: float = None) -> dict:
    """
    Full pipeline: Johansen for the static β, Kalman for dynamic tracking.

    Returns a dict with:
        beta_static  : float — Johansen β for the full sample / fold
        kf           : DataFrame — full Kalman output (beta, spread, etc.)
        spread_static: Series — log(T1) − beta_static·log(T2)
        spread_dynamic: Series — kf['spread']  (uses time-varying β)

    In your backtester, use:
        spread_static  for training-fold spread (fixed β known at fold start)
        spread_dynamic for live/paper trading   (β updates each day)
        kf['beta_std'] as the stability gate signal before each entry
    """
    # ── Static β from Johansen ────────────────────────────────────────────────
    beta_static, evec = estimate_beta_johansen(
        df, T1, T2, lookback=johansen_lookback
    )

    log_p1 = np.log(df[T1])
    log_p2 = np.log(df[T2])
    spread_static = log_p1 - beta_static * log_p2

    # ── Dynamic β from Kalman ─────────────────────────────────────────────────
    kf = kalman_beta(df, T1, T2,
                     beta_init=beta_static,   # warm-start from Johansen
                     Q=kalman_Q,
                     R=kalman_R)

    return dict(
        beta_static   = beta_static,
        evec          = evec,
        kf            = kf,
        spread_static = spread_static,
        spread_dynamic= kf['spread'],
    )


# =============================================================================
# 6.  QUICK DIAGNOSTIC PLOT
# =============================================================================

def plot_beta_comparison(df: pd.DataFrame,
                         T1: str,
                         T2: str,
                         window: int = 120,
                         kalman_Q: float = 1e-5) -> None:
    """
    Side-by-side comparison of OLS rolling β vs Kalman β.
    Use this once to calibrate your Q choice visually.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # OLS rolling (original method — kept for comparison only)
    ols_betas = []
    for i in range(len(df)):
        if i < window - 1:
            ols_betas.append(np.nan)
            continue
        chunk = df.iloc[i - window + 1: i + 1]
        try:
            res = OLS(np.log(chunk[T1]),
                      add_constant(np.log(chunk[T2]))).fit()
            ols_betas.append(float(res.params[1]))
        except Exception:
            ols_betas.append(np.nan)
    ols_series = pd.Series(ols_betas, index=df.index)

    # Kalman
    kf = kalman_beta(df, T1, T2, Q=kalman_Q)

    # Johansen static
    beta_joh, _ = estimate_beta_johansen(df, T1, T2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 7))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Beta panel
    ax1.plot(ols_series,      color='#9e9e9e', lw=1,   label=f'OLS rolling ({window}d)', alpha=0.8)
    ax1.plot(kf['beta'],      color='#1565C0', lw=1.5, label=f'Kalman (Q={kalman_Q:.0e})')
    ax1.fill_between(kf.index,
                     kf['beta'] - 2 * kf['beta_std'],
                     kf['beta'] + 2 * kf['beta_std'],
                     alpha=0.15, color='#1565C0', label='Kalman ±2σ')
    ax1.axhline(beta_joh, color='#C62828', lw=1.2, ls='--',
                label=f'Johansen full-sample β={beta_joh:.4f}')
    ax1.set_title(f'Hedge ratio β — {T1} / {T2}', fontsize=12)
    ax1.set_ylabel('β')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    # Posterior std panel (stability signal)
    ax2.plot(kf['beta_std'], color='#6A1B9A', lw=1.2, label='Kalman β posterior std')
    ax2.set_ylabel('β posterior std')
    ax2.set_title('β uncertainty  (use as entry stability gate)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 7.  MAIN  — drop-in replacement demo
# =============================================================================

if __name__ == '__main__':
    from data import data_loader
    from params import PARAMS

    data_loader_obj = data_loader()
    df = data_loader_obj.load_data(needs_fx=False)

    T1 = PARAMS['T1']
    T2 = PARAMS['T2']

    # ── Full pipeline ─────────────────────────────────────────────────────────
    result = build_spread(df, T1, T2, kalman_Q=1e-5)

    print(f"\nJohansen full-sample β : {result['beta_static']:.4f}")
    print(f"\nKalman β  (last 10 bars):")
    print(result['kf'][['beta', 'beta_std', 'spread']].tail(10).round(4))

    # ── Stability check at last bar ───────────────────────────────────────────
    kf      = result['kf']
    stable, reason = beta_is_stable(kf.iloc[-1], kf)
    print(f"\nEntry gate — stable: {stable}  ({reason})")

    # ── Visual comparison ─────────────────────────────────────────────────────
    plot_beta_comparison(df, T1, T2, window=120, kalman_Q=1e-5)
