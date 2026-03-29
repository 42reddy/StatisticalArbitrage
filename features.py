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
    def build_features(self, df, p, ou_mean=None, beta=None):
        """
        Parameters
        ----------
        df      : price DataFrame with T1 and T2 columns
        p       : PARAMS dict
        ou_mean : pre-computed equilibrium mean of the spread
                  (pass the training-window mean for holdout)
        beta    : pre-computed hedge ratio
                  If None, estimated from the first slow_window*2
                  bars of df (in-sample within the slice passed in)

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

        # ── Entry: cross + confirmation + regime ──────────
        long_entry  = cross_long  & conf_ok_long  & mr & vol_ok
        short_entry = cross_short & conf_ok_short & mr & vol_ok

        # ── Size multiplier (soft vol scaling pre-hard-cap) ──
        size_mult = np.where(vr < p['vol_cap'] * 0.75, 1.0,
                    np.where(vr < p['vol_cap'],         0.75, 0.5))

        # ── Pyramid guard: no add within 2 bars of entry ──
        entry_occurred = long_entry | short_entry
        recent_entry   = (entry_occurred |
                          entry_occurred.shift(1).fillna(False) |
                          entry_occurred.shift(2).fillna(False))

        long_add  = ((z_slow < -p['z_add']) & (z_slow.shift(1) >= -p['z_add']) &
                     mr & vol_ok & ~recent_entry)
        short_add = ((z_slow >  p['z_add']) & (z_slow.shift(1) <=  p['z_add']) &
                     mr & vol_ok & ~recent_entry)

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
            'long_add':         long_add,
            'short_add':        short_add,
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
        n_add_l = sig['long_add'].sum()
        n_add_s = sig['short_add'].sum()

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
Pyramid signals             : {n_add_l + n_add_s}  ({n_add_l} long, {n_add_s} short)

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