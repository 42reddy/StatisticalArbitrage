import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from params import PARAMS


class features():

    def __init__(self):
        self.T1 = PARAMS['T1']
        self.T2 = PARAMS["T2"]

    # ══════════════════════════════════════════════════════
    # HEDGE RATIO ESTIMATION
    #
    # OLS regression of log(T1) on log(T2) gives the
    # cointegrating coefficient β (hedge ratio).
    #
    # Spread = log(T1) - β * log(T2)
    #
    # This is the Engle-Granger first step. β ≠ 1 for
    # nearly all commodity pairs — assuming 1:1 distorts
    # every z-score, entry threshold, stop, and PnL calc.
    #
    # We estimate β on the first `lookback` bars of df
    # (in-sample window passed in), then apply it to all
    # bars in df. This prevents lookahead on a rolling
    # basis — the caller controls the estimation window.
    #
    # Returns: beta (scalar float)
    # ══════════════════════════════════════════════════════
    @staticmethod
    def estimate_hedge_ratio(df, T1, T2, lookback=None):
        p1 = np.log(df[T1])
        p2 = np.log(df[T2])
        if lookback is not None:
            p1 = p1.iloc[:lookback]
            p2 = p2.iloc[:lookback]

        # Drop any rows where either series has NaN or inf
        # (bad ticks, zero prices, missing data from yfinance)
        mask = np.isfinite(p1) & np.isfinite(p2)
        p1 = p1[mask]
        p2 = p2[mask]

        if len(p1) < 30:
            raise ValueError(
                f"Too few clean bars for OLS hedge ratio estimation: {len(p1)}. "
                f"Check your price data for zeros or NaNs.")

        p2_c = add_constant(p2)
        res = OLS(p1, p2_c).fit()
        beta = float(res.params.iloc[1])
        return beta

    # ══════════════════════════════════════════════════════
    # TIME-VARYING HEDGE RATIO — KALMAN FILTER
    #
    # Static OLS beta assumes a constant cointegrating relationship over
    # the whole estimation window. In practice short-window re-estimates
    # are very noisy (see rolling_beta_stability.py: std comparable to
    # the mean), but that noise is mostly an estimation artefact, not a
    # real regime change — log(T1)/log(T2) are both non-stationary, so a
    # short window can't pin down the cointegrating vector (alpha and
    # beta trade off against each other almost freely until the window
    # spans enough of the series' range to break that collinearity).
    # Empirically this pair needs >= ~1000 bars before the level
    # regression is identified at all (shorter seed windows produce a
    # beta whose implied spread fails an ADF stationarity test).
    #
    # This models beta as a hidden random walk (state-space form, the
    # standard pairs-trading approach — Chan, "Algorithmic Trading"):
    #   state:        theta_t = [beta_t, alpha_t]
    #   transition:   theta_t = theta_{t-1} + w_t,   w_t ~ N(0, Q)
    #   observation:  log(T1)_t = beta_t * log(T2)_t + alpha_t + v_t
    #
    # Q is parametrised via `delta` (Q = delta/(1-delta) * I): delta -> 0
    # freezes beta at the seed (recovers static OLS); larger delta lets
    # it drift. R is the observation-noise variance — it MUST be the
    # regression's residual variance, not the variance of price
    # differences (using return-variance there was the original bug:
    # it made the filter wildly overconfident, since residual variance
    # is roughly an order of magnitude larger, and it froze beta near
    # the seed's local, mis-identified value).
    #
    # `init_window` bars are used only to seed theta via plain OLS and
    # are returned as NaN — the filter only becomes meaningfully causal
    # (uses no information beyond the current bar) from bar
    # `init_window` onward, so callers must treat this exactly like any
    # other rolling feature with a warm-up period.
    # ══════════════════════════════════════════════════════
    @staticmethod
    def kalman_hedge_ratio(df, T1, T2, delta=1e-6, init_window=1000):
        p1 = np.log(df[T1])
        p2 = np.log(df[T2])
        mask = np.isfinite(p1) & np.isfinite(p2)
        p1 = p1[mask]
        p2 = p2[mask]

        n = len(p1)
        if n < init_window:
            raise ValueError(
                f"Too few clean bars for Kalman hedge ratio: {n} "
                f"(need >= init_window={init_window}).")

        y = p1.values
        x = p2.values

        x0_c = add_constant(x[:init_window])
        seed = OLS(y[:init_window], x0_c).fit()
        theta = np.array([float(seed.params[1]), float(seed.params[0])])
        R = float(np.var(seed.resid))
        R = R if R > 0 else 1e-8

        Q = (delta / (1 - delta)) * np.eye(2)
        P = np.eye(2) * 0.01

        betas = np.empty(n)
        for t in range(n):
            H = np.array([x[t], 1.0])

            # predict (random walk — no drift)
            P = P + Q

            # update
            e = y[t] - H @ theta
            S = H @ P @ H + R
            K = (P @ H) / S
            theta = theta + K * e
            P = P - np.outer(K, H) @ P

            betas[t] = theta[0]

        betas[:init_window] = np.nan
        return pd.Series(betas, index=p1.index, name='beta_kalman')

    # ══════════════════════════════════════════════════════
    def build_features(self, df, p, ou_mean=None, beta=None):
        """
        Parameters
        ----------
        df      : price DataFrame with T1 and T2 columns
        p       : PARAMS dict
        ou_mean : pre-computed equilibrium mean of the spread
                  (pass the training-window mean for holdout)
        beta    : pre-computed hedge ratio — scalar (static OLS) or a
                  pd.Series aligned to df.index (e.g. from
                  kalman_hedge_ratio, time-varying). If None, estimated
                  from the first slow_window*2 bars of df (in-sample
                  within the slice passed in)

        Returns
        -------
        feat    : DataFrame of features (includes 'beta' column)
        ou_mean : float — equilibrium mean used
        """

        """df = df[(df[self.T1] > 0) & (df[self.T2] > 0)].copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) < p.get('slow_window', 60):
            raise ValueError(
                f"Too few clean bars after filtering: {len(df)}. "
                f"Check price data for this fold.")"""

        # ── Hedge ratio ──────────────────────────────────
        if beta is None:
            lookback = min(len(df), p.get('slow_window', 60) * 4)
            beta = self.estimate_hedge_ratio(df, self.T1, self.T2, lookback)

        # ── Hedge-ratio-adjusted spread ───────────────────
        # lr = log(T1) - β * log(T2)
        # This is the mean-reverting combination whose
        # z-scores and OU dynamics are actually stationary.
        lr = np.log(df[self.T1]) - beta * np.log(df[self.T2])

        # ── Z-scores ──────────────────────────────────────
        mu_slow = lr.rolling(p['slow_window'],
                             min_periods=p['slow_window'] // 2).mean()
        mu_med  = lr.rolling(p['medium_window'],
                             min_periods=p['medium_window'] // 2).mean()
        vol_slow = lr.rolling(p['vol_window'],
                              min_periods=p['vol_window'] // 2).std()

        z_slow = (lr - mu_slow) / vol_slow.replace(0, np.nan)
        z_med  = (lr - mu_med)  / vol_slow.replace(0, np.nan)

        mu_fast = lr.ewm(span=p['fast_span'], adjust=False).mean()
        z_fast  = (lr - mu_fast) / vol_slow.replace(0, np.nan)

        # ── OU z-score with adaptive drift ────────────────
        if ou_mean is None:
            ou_mean = float(lr.iloc[:504].mean())
        ou_mean_series  = lr.ewm(span=p.get('ou_adapt_span', 252), adjust=False).mean()
        ou_mean_blended = 0.7 * ou_mean + 0.3 * ou_mean_series
        z_ou = (lr - ou_mean_blended) / vol_slow.replace(0, np.nan)

        # ── Vol regime ────────────────────────────────────
        vol_ratio = (lr.rolling(10).std() /
                     lr.rolling(60).std().replace(0, np.nan)).fillna(1.0)

        # ── Autocorrelation regime filter ─────────────────
        autocorr = lr.diff().rolling(p.get('autocorr_window', 20)).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 5 else 0,
            raw=False
        ).fillna(0)
        mr_regime = autocorr < p.get('autocorr_threshold', 0.1)

        # ── ATR ───────────────────────────────────────────
        atr = lr.diff().abs().ewm(span=14, adjust=False).mean()

        # ── Rolling vol z-score (of spread volatility) ────
        # Use existing slow vol as the base series, then
        # standardise it versus a slower rolling window.
        vol_z_window = int(p.get('vol_z_window', p.get('vol_window', 60) * 2))
        vol_mean = vol_slow.rolling(vol_z_window,
                                    min_periods=vol_z_window // 2).mean()
        vol_std = vol_slow.rolling(vol_z_window,
                                   min_periods=vol_z_window // 2).std()
        vol_zscore = (vol_slow - vol_mean) / vol_std.replace(0, np.nan)

        # ── RSI of the spread (Wilder-style) ───────────────
        rsi_period = int(p.get('rsi_period', 14))
        lr_delta = lr.diff()
        up = lr_delta.clip(lower=0.0)
        down = -lr_delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1 / rsi_period, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / rsi_period, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        # ── Simple momentum of the spread ─────────────────
        mom_window = int(p.get('mom_window', 5))
        lr_mom = lr.diff(mom_window)

        # ── "Falling knife" diagnostics ────────────────────
        # z_fast_slope:  rate of change of the fast-mean deviation —
        #   how quickly price is moving away from (or back
        #   toward) the mean right now.
        # z_fast_accel:  rate of change of that slope. A trade
        #   entered while |z_fast_accel| is large and same-sign
        #   as the move (still accelerating) is catching a
        #   falling knife — the dislocation may be event-driven,
        #   not yet exhausted. A small/decelerating accel means
        #   the move is losing steam and more likely to revert.
        # z_fast_accel_z: accel standardised against its own
        #   rolling distribution so it's comparable across pairs
        #   and regimes (raw accel has no natural scale).
        #
        # IMPORTANT: these are built from the raw fast-mean gap
        # (lr - mu_fast), NOT from z_fast itself. z_fast divides that
        # gap by vol_slow (a 20d rolling std) — during a genuinely
        # sharp, accelerating move, vol_slow inflates *faster* than
        # the gap widens (it's measuring vol over a window that
        # contains the very shock being measured), so z_fast's
        # magnitude mechanically SHRINKS at the height of a violent
        # move. That made z_fast_slope/accel report a still-violent
        # plunge as "decelerating" — confirmed by feeding a synthetic
        # accelerating-then-flat plunge through the formula: the raw
        # gap widened every bar while z_fast shrank from -3.7 to -1.7
        # over the same stretch, purely from the denominator catching
        # up. Normalizing by a slower background vol (accel_vol_mult x
        # longer window, so it doesn't react to the move it's supposed
        # to be measuring against) recovers the intended sign.
        accel_window = int(p.get('accel_window', 3))
        accel_vol_mult = int(p.get('accel_vol_mult', 3))
        gap = lr - mu_fast
        vol_bg = lr.rolling(p['vol_window'] * accel_vol_mult,
                             min_periods=p['vol_window']).std()
        z_fast_slope = gap.diff(accel_window) / vol_bg.replace(0, np.nan)
        z_fast_accel = gap.diff(accel_window).diff(accel_window) / vol_bg.replace(0, np.nan)
        accel_z_window = vol_z_window
        z_fast_accel_z = (
            (z_fast_accel - z_fast_accel.rolling(
                accel_z_window, min_periods=accel_z_window // 2).mean()) /
            z_fast_accel.rolling(
                accel_z_window, min_periods=accel_z_window // 2).std().replace(0, np.nan)
        )

        # ── Slopes of RSI / momentum (same accelerating-vs-
        #    decelerating logic applied to the other oscillators) ──
        rsi_slope = rsi.diff(accel_window)
        lr_mom_slope = lr_mom.diff(accel_window)
        lr_mom_slope_z = (
            (lr_mom_slope - lr_mom_slope.rolling(
                accel_z_window, min_periods=accel_z_window // 2).mean()) /
            lr_mom_slope.rolling(
                accel_z_window, min_periods=accel_z_window // 2).std().replace(0, np.nan)
        )

        # ── Confirmation score (asymmetric thresholds) ────
        CONFIRM_THR = 1.0

        def confirm(z_series, thr):
            return np.where(z_series > thr,  1,
                   np.where(z_series < -thr, -1, 0))

        agreement = pd.Series(
            confirm(z_slow, CONFIRM_THR) +
            confirm(z_med,  CONFIRM_THR) +
            confirm(z_fast, CONFIRM_THR * 0.8) +
            confirm(z_ou,   CONFIRM_THR * 0.8),
            index=lr.index
        )

        # ── Raw crosses (needed by signal_diagnostics) ────
        ze_long  = p.get('z_entry_long',  p.get('z_entry', 1.25))
        ze_short = p.get('z_entry_short', p.get('z_entry', 1.25))
        cross_down = (z_slow < -ze_long)  & (z_slow.shift(1) >= -ze_long)
        cross_up   = (z_slow >  ze_short) & (z_slow.shift(1) <= ze_short)

        # ── Composite regime score ─────────────────────────
        # agreement (±4) plus signed mean-reversion regime and
        # vol-ok flags (±1 each) — a single scalar summarising
        # how favourable the regime was at any given bar, for
        # post-hoc bucketing of trade outcomes against PnL.
        vol_ok_flag = vol_ratio < p['vol_cap']
        mr_signed   = pd.Series(np.where(mr_regime, 1, -1), index=lr.index)
        vol_signed  = pd.Series(np.where(vol_ok_flag, 1, -1), index=lr.index)
        regime_score = agreement + mr_signed + vol_signed

        return pd.DataFrame({
            'lr':        lr,
            'beta':      beta,           # scalar broadcast — constant column
            'mu_slow':   mu_slow,
            'mu_med':    mu_med,
            'mu_fast':   mu_fast,
            'vol':       vol_slow,
            'vol_ratio': vol_ratio,
            'z_slow':    z_slow,
            'z_med':     z_med,
            'z_fast':    z_fast,
            'z_ou':      z_ou,
            'z':         z_slow,
            'agreement': agreement,
            'mr_regime': mr_regime,
            'autocorr':  autocorr,
            'atr':       atr,
            'vol_z':     vol_zscore,
            'rsi_lr':    rsi,
            'rsi_slope': rsi_slope,
            'lr_mom':    lr_mom,
            'lr_mom_slope':   lr_mom_slope,
            'lr_mom_slope_z': lr_mom_slope_z,
            'z_fast_slope':   z_fast_slope,
            'z_fast_accel':   z_fast_accel,
            'z_fast_accel_z': z_fast_accel_z,
            'regime_score':   regime_score,
            'cross_down': cross_down,
            'cross_up':   cross_up,
        }, index=df.index), float(ou_mean)

    # ══════════════════════════════════════════════════════
    def generate_signals(self, feat, p):
        z      = feat['z']
        z_slow = feat['z_slow']
        vr     = feat['vol_ratio']
        agree  = feat['agreement']
        mr     = feat['mr_regime']

        ze_long  = p.get('z_entry_long',  p.get('z_entry', 1.25))
        ze_short = p.get('z_entry_short', p.get('z_entry', 1.25))
        zx_long  = p.get('z_exit_long',   p.get('z_exit',  0.30))
        zx_short = p.get('z_exit_short',  p.get('z_exit',  0.30))
        zs_long  = p.get('z_stop_long',   p.get('z_stop',  3.00))
        zs_short = p.get('z_stop_short',  p.get('z_stop',  3.00))

        vol_ok = vr < p['vol_cap']

        # ── Confirmation gate (fixed symmetric threshold) ──
        # Decoupled from ze so Bayesian can't game it by
        # lowering one entry threshold to inflate agreement.
        conf_ok_long  = agree <= -2   # ≥2 z-scores below −1σ
        conf_ok_short = agree >=  2   # ≥2 z-scores above +1σ

        # ── Asymmetric crosses on z_slow (PRIMARY) ────────
        cross_long  = (z_slow < -ze_long)  & (z_slow.shift(1) >= -ze_long)
        cross_short = (z_slow >  ze_short) & (z_slow.shift(1) <= ze_short)

        # ── Falling-knife guard (long only) ───────────────
        # Pooled across 7 pairs (multi_pair_backtest.py, guards disabled,
        # unconstrained capital, n=297 / 59 losses) and checked pair-by-pair
        # (scratch2.py MODE='multipair') — not single-pair-tuned like the
        # version this replaced. Three features sign-stable in ALL 4 pairs
        # with enough long losses to read (BAJFINANCE, NHPC/NTPC, OIL/ONGC,
        # PETRONET/MGL):
        #   rsi_slope     -0.667 pooled — winning longs had a MORE negative
        #                 RSI slope (a deeper, more decisive move) than
        #                 losing longs, in every pair checked.
        #   lr_mom_slope  -0.523 pooled — same pattern: losing longs had a
        #                 shallower/still-building 5-day momentum slope —
        #                 the z-cross caught noise, not a real move.
        #   z_fast_slope  -0.388 pooled — losing longs had a weaker (less
        #                 negative) fast-mean-gap slope at entry.
        # All three point the same way: a STRONGER (more negative) reading
        # at entry predicts a win, not a loss — i.e. shallow/weak crosses
        # are the false positives, not violently-accelerating ones. This is
        # the opposite mechanism from the guard's original name ("falling
        # knife" = too violent), but it's what 7 pairs of pooled, cross-pair-
        # stable data show; keep the name, the logic now filters out WEAK
        # moves instead. 2-of-3 vote, same consensus rationale as before:
        # no single noisy reading can veto a trade alone.
        if p.get('fk_enabled', False):
            fk_rsi_slope_max    = p.get('fk_rsi_slope_max',    -8.410)  # do be < this (deep, decisive move)
            fk_lr_mom_slope_max = p.get('fk_lr_mom_slope_max', -0.029)  # do be < this (momentum already building)
            fk_z_fast_slope_max = p.get('fk_z_fast_slope_max', -0.784) # do be < this (real, not a shallow cross)

            fk_votes_long = (
                (feat['rsi_slope']    < fk_rsi_slope_max).astype(int) +
                (feat['lr_mom_slope'] < fk_lr_mom_slope_max).astype(int) +
                (feat['z_fast_slope'] < fk_z_fast_slope_max).astype(int)
            )
            not_falling_knife_long = fk_votes_long >= p.get('fk_votes_min', 2)
        else:
            not_falling_knife_long = pd.Series(True, index=feat.index)

        # ── Exhaustion guard (short only) ──────────────────
        # Same pooled 7-pair methodology as the long-side guard above (n=297
        # / 59 losses, checked pair-by-pair). Features stable in 4/5 pairs
        # with enough short losses to read (BAJFINANCE, HDFCBANK, NHPC/NTPC,
        # OIL/ONGC, PETRONET/MGL):
        #   z_fast_slope  +0.523 pooled — winning shorts had a HIGHER (still
        #                 rising) fast-mean-gap slope than losing shorts.
        #   z_fast_accel  +0.416 pooled — same pattern for acceleration.
        #   rsi_slope     +0.541 pooled, 4/5 — winning shorts had a higher
        #                 RSI slope (still climbing) than losing shorts.
        # This FLIPS the sign of the guard's previous (single-pair-tuned)
        # version, which assumed winners had LOWER readings ("cresting").
        # Pooled across pairs, it's the opposite: a stronger, still-rising
        # move at entry predicts a win — shallow/stalling crosses are the
        # false positives. Same 2-of-3 consensus voting as the long side.
        if p.get('se_enabled', False):
            se_z_fast_slope_min = p.get('se_z_fast_slope_min', 0.840)  # don't be < this (move still real)
            se_z_fast_accel_min = p.get('se_z_fast_accel_min', 0.670)  # don't be < this (still accelerating)
            se_rsi_slope_min    = p.get('se_rsi_slope_min',    9.277)  # don't be < this (RSI still climbing)

            se_votes_short = (
                (feat['z_fast_slope'] > se_z_fast_slope_min).astype(int) +
                (feat['z_fast_accel'] > se_z_fast_accel_min).astype(int) +
                (feat['rsi_slope']    > se_rsi_slope_min).astype(int)
            )
            not_exhausted_short = se_votes_short >= p.get('se_votes_min', 2)
        else:
            not_exhausted_short = pd.Series(True, index=feat.index)

        # ── Entry: cross + confirmation + regime ──────────
        long_entry  = cross_long  & conf_ok_long  & mr & vol_ok & not_falling_knife_long
        short_entry = cross_short & conf_ok_short & mr & vol_ok & not_exhausted_short

        # ── Size multiplier (soft vol scaling pre-hard-cap) ──
        size_mult = np.where(vr < p['vol_cap'] * 0.75, 1.0,
                    np.where(vr < p['vol_cap'],         0.75, 0.5))

        # ── Exits ─────────────────────────────────────────
        # Stop: spread moves FURTHER against position
        # Long stop  → spread fell below -zs_long  (more negative = more extreme)
        # Short stop → spread rose above +zs_short (more positive = more extreme)
        exit_stop_long  = z_slow < -zs_long
        exit_stop_short = z_slow >  zs_short
        exit_stop       = exit_stop_long | exit_stop_short

        # Mean revert: spread reverts back through exit threshold
        # Long exits when spread is no longer depressed (z > -zx_long)
        # Short exits when spread is no longer elevated (z < +zx_short)
        exit_mean_long  = z_slow > -zx_long
        exit_mean_short = z_slow <  zx_short
        exit_mean       = exit_mean_long | exit_mean_short

        # Zero cross: spread crosses equilibrium (clean exit)
        exit_cross = ((z > 0) & (z.shift(1) < 0)) | ((z < 0) & (z.shift(1) > 0))

        exit_any = exit_stop | exit_mean | exit_cross

        return pd.DataFrame({
            'long_entry':       long_entry,
            'short_entry':      short_entry,
            'exit_mean_long':   exit_mean_long,
            'exit_mean_short':  exit_mean_short,
            'exit_stop_long':   exit_stop_long,
            'exit_stop_short':  exit_stop_short,
            'exit_mean':        exit_mean,
            'exit_cross':       exit_cross,
            'exit_stop':        exit_stop,
            'exit_any':         exit_any,
            'exit_priority':    (exit_stop.astype(int) * 3 +
                                 exit_mean.astype(int) * 2 +
                                 exit_cross.astype(int) * 1),
            'size_mult':        pd.Series(size_mult, index=feat.index),
            'z':                z,
            'z_slow':           z_slow,
            'z_fast':           feat['z_fast'],
            'z_ou':             feat['z_ou'],
            'z_med':            feat['z_med'],
            'agreement':        agree,
            'vol_ratio':        vr,
            'mr_regime':        mr,
        }, index=feat.index)

    # ══════════════════════════════════════════════════════
    # SIGNAL DIAGNOSTICS  (v3 — fixed cross_down/cross_up)
    # ══════════════════════════════════════════════════════
    def signal_diagnostics(self, feat, sig, p):
        z     = feat['z'].dropna()
        total = len(z)
        th    = p.get('z_entry_long', p.get('z_entry', 1.25))

        n_long  = sig['long_entry'].sum()
        n_short = sig['short_entry'].sum()

        above = (z.abs() > th).sum()

        # Regime stats
        mr_pct  = feat['mr_regime'].mean() * 100
        vol_ok  = (feat['vol_ratio'] < p['vol_cap']).mean() * 100
        conf_l  = (feat['agreement'] <= -2).mean() * 100
        conf_s  = (feat['agreement'] >=  2).mean() * 100

        # Raw crosses from features (not recomputed here)
        raw_cross_down = feat['cross_down'].sum()
        raw_cross_up   = feat['cross_up'].sum()
        filtered_l     = raw_cross_down - n_long
        filtered_s     = raw_cross_up   - n_short

        print(f"""
── Signal Diagnostics v3 ──
Bars with data              : {total}
|z| > {th:.2f} (sustained)    : {above:>5}  ({above/total*100:.1f}% of bars)

Raw crosses (z_slow only)   : {raw_cross_down} long / {raw_cross_up} short
Filtered by regime/confirm  : {filtered_l} long / {filtered_s} short
Final entries               : {n_long} long / {n_short} short

Regime conditions (% bars):
  Mean-reverting regime     : {mr_pct:.1f}%
  Vol OK (< cap)            : {vol_ok:.1f}%
  Confirmation long (≥2)   : {conf_l:.1f}%
  Confirmation short (≥2)  : {conf_s:.1f}%

Z-score coverage:
  z_slow : min={feat['z_slow'].min():.2f}  max={feat['z_slow'].max():.2f}  std={feat['z_slow'].std():.2f}
  z_med  : min={feat['z_med'].min():.2f}  max={feat['z_med'].max():.2f}  std={feat['z_med'].std():.2f}
  z_fast : min={feat['z_fast'].min():.2f}  max={feat['z_fast'].max():.2f}  std={feat['z_fast'].std():.2f}
  z_ou   : min={feat['z_ou'].min():.2f}  max={feat['z_ou'].max():.2f}  std={feat['z_ou'].std():.2f}
  agree  : min={feat['agreement'].min():.0f}  max={feat['agreement'].max():.0f}  mean={feat['agreement'].mean():.2f}
""")