import numpy as np
import pandas as pd


class features():

    def __init__(self):
        self.T1 = "BZ=F"
        self.T2 = "CL=F"

    # ══════════════════════════════════════════════════════
    # FEATURES  (v2 — robust rewrite)
    #
    # Key changes vs v1:
    #   - z_slow is PRIMARY signal; z_fast and z_ou are CONFIRMATION gates
    #     (not averaged in — averaging dilutes strong signals)
    #   - Regime filter added: blocks entries when spread is trending
    #     (Hurst-proxy via vol_ratio + rolling autocorrelation)
    #   - Vol regime is a HARD block above vol_cap (not just size reduction)
    #   - OU mean updated with slow EWM to track equilibrium drift
    #   - medium_window z-score retained and used as confirmation tier
    # ══════════════════════════════════════════════════════

    def build_features(self, df, p, ou_mean=None):
        lr = np.log(df[self.T1] / df[self.T2])

        # ── Z-scores ──
        mu_slow = lr.rolling(p['slow_window'], min_periods=p['slow_window'] // 2).mean()
        mu_med = lr.rolling(p['medium_window'], min_periods=p['medium_window'] // 2).mean()
        vol_slow = lr.rolling(p['vol_window'], min_periods=p['vol_window'] // 2).std()

        z_slow = (lr - mu_slow) / vol_slow.replace(0, np.nan)
        z_med = (lr - mu_med) / vol_slow.replace(0, np.nan)

        mu_fast = lr.ewm(span=p['fast_span'], adjust=False).mean()
        z_fast = (lr - mu_fast) / vol_slow.replace(0, np.nan)

        # ── OU z-score with adaptive drift ──
        if ou_mean is None:
            ou_mean = float(lr.iloc[:504].mean())
        ou_mean_series = lr.ewm(span=p.get('ou_adapt_span', 252), adjust=False).mean()
        ou_mean_blended = 0.7 * ou_mean + 0.3 * ou_mean_series
        z_ou = (lr - ou_mean_blended) / vol_slow.replace(0, np.nan)

        # ── Vol regime ──
        vol_ratio = (lr.rolling(10).std() /
                     lr.rolling(60).std().replace(0, np.nan)).fillna(1.0)

        # ── Autocorrelation regime filter ──
        autocorr = lr.diff().rolling(p.get('autocorr_window', 20)).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 5 else 0,
            raw=False
        ).fillna(0)
        mr_regime = autocorr < p.get('autocorr_threshold', 0.1)

        # ── ATR ──
        atr = lr.diff().abs().ewm(span=14, adjust=False).mean()

        # ── Confirmation score (asymmetric thresholds) ──
        ze_long = p.get('z_entry_long', p.get('z_entry', 1.25))
        ze_short = p.get('z_entry_short', p.get('z_entry', 1.25))

        CONFIRM_THR = 1.0

        def confirm(z_series, thr):
            return np.where(z_series > thr, 1,
                            np.where(z_series < -thr, -1, 0))

        agreement = pd.Series(
            confirm(z_slow, CONFIRM_THR) +
            confirm(z_med, CONFIRM_THR) +
            confirm(z_fast, CONFIRM_THR * 0.8) +
            confirm(z_ou, CONFIRM_THR * 0.8),
            index=lr.index
        )

        # NOTE: cross detection removed from here — handled in generate_signals
        #       using asymmetric thresholds per direction

        return pd.DataFrame({
            'lr': lr,
            'mu_slow': mu_slow,
            'mu_med': mu_med,
            'mu_fast': mu_fast,
            'vol': vol_slow,
            'vol_ratio': vol_ratio,
            'z_slow': z_slow,
            'z_med': z_med,
            'z_fast': z_fast,
            'z_ou': z_ou,
            'z': z_slow,
            'agreement': agreement,
            'mr_regime': mr_regime,
            'autocorr': autocorr,
            'atr': atr,
        }, index=df.index), float(ou_mean)


    # ══════════════════════════════════════════════════════
    # SIGNAL GENERATION  (v2)
    #
    # Key changes vs v1:
    #   - Entry requires PRIMARY cross + CONFIRMATION (agreement >= 2)
    #   - Entry blocked in trending regime (mr_regime filter)
    #   - Entry hard-blocked above vol_cap (not just size reduction)
    #   - Pyramid guard: long_add can't fire within 2 bars of entry
    #   - Exit priority: stop > time > mean (explicit hierarchy)
    # ══════════════════════════════════════════════════════
    def generate_signals(self, feat, p):
        z = feat['z']
        z_slow = feat['z_slow']
        vr = feat['vol_ratio']
        agree = feat['agreement']
        mr = feat['mr_regime']

        ze_long = p.get('z_entry_long', p.get('z_entry', 1.25))
        ze_short = p.get('z_entry_short', p.get('z_entry', 1.25))
        zx_long = p.get('z_exit_long', p.get('z_exit', 0.30))
        zx_short = p.get('z_exit_short', p.get('z_exit', 0.30))
        zs_long = p.get('z_stop_long', p.get('z_stop', 3.00))
        zs_short = p.get('z_stop_short', p.get('z_stop', 3.00))

        vol_ok = vr < p['vol_cap']

        # ── Confirmation uses FIXED symmetric threshold ──
        # Decoupled from ze_long/ze_short so Bayesian can't game
        # the confirmation gate by lowering one entry threshold.
        # 1.0σ is a reasonable fixed confirmation bar — z_slow,
        # z_med, z_fast, z_ou must each exceed this to vote.
        CONFIRM_THR = 1.0
        conf_ok_long = agree <= -2  # ≥2 z-scores below -1.0σ
        conf_ok_short = agree >= 2  # ≥2 z-scores above +1.0σ

        # ── Asymmetric crosses on z_slow (PRIMARY entry signal) ──
        cross_long = (z_slow < -ze_long) & (z_slow.shift(1) >= -ze_long)
        cross_short = (z_slow > ze_short) & (z_slow.shift(1) <= ze_short)

        # ── Entry: cross + confirmation + regime ──
        long_entry = cross_long & conf_ok_long & mr & vol_ok
        short_entry = cross_short & conf_ok_short & mr & vol_ok

        size_mult = np.where(vr < p['vol_cap'] * 0.75, 1.0,
                             np.where(vr < p['vol_cap'], 0.75, 0.5))

        entry_occurred = long_entry | short_entry
        recent_entry = (entry_occurred |
                        entry_occurred.shift(1).fillna(False) |
                        entry_occurred.shift(2).fillna(False))

        long_add = ((z_slow < -p['z_add']) & (z_slow.shift(1) >= -p['z_add']) &
                    mr & vol_ok & ~recent_entry)
        short_add = ((z_slow > p['z_add']) & (z_slow.shift(1) <= p['z_add']) &
                     mr & vol_ok & ~recent_entry)

        # ── Exits ──
        exit_stop_long = z_slow < -zs_long
        exit_stop_short = z_slow > zs_short
        exit_stop = exit_stop_long | exit_stop_short

        exit_mean_long = z_slow > -zx_long
        exit_mean_short = z_slow < zx_short
        exit_mean = exit_mean_long | exit_mean_short

        exit_cross = ((z > 0) & (z.shift(1) < 0)) | ((z < 0) & (z.shift(1) > 0))
        exit_any = exit_stop | exit_mean | exit_cross

        return pd.DataFrame({
            'long_entry': long_entry,
            'short_entry': short_entry,
            'long_add': long_add,
            'short_add': short_add,
            'exit_mean_long': exit_mean_long,
            'exit_mean_short': exit_mean_short,
            'exit_stop_long': exit_stop_long,
            'exit_stop_short': exit_stop_short,
            'exit_mean': exit_mean,
            'exit_cross': exit_cross,
            'exit_stop': exit_stop,
            'exit_any': exit_any,
            'exit_priority': (exit_stop.astype(int) * 3 +
                              exit_mean.astype(int) * 2 +
                              exit_cross.astype(int) * 1),
            'size_mult': pd.Series(size_mult, index=feat.index),
            'z': z,
            'z_slow': z_slow,
            'z_fast': feat['z_fast'],
            'z_ou': feat['z_ou'],
            'z_med': feat['z_med'],
            'agreement': agree,
            'vol_ratio': vr,
            'mr_regime': mr,
        }, index=feat.index)


    # ══════════════════════════════════════════════════════
    # SIGNAL DIAGNOSTICS  (v2 — extended)
    # ══════════════════════════════════════════════════════
    def signal_diagnostics(self, feat, sig, p):
        z     = feat['z'].dropna()
        total = len(z)
        th    = p['z_entry']

        n_long  = sig['long_entry'].sum()
        n_short = sig['short_entry'].sum()
        n_add_l = sig['long_add'].sum()
        n_add_s = sig['short_add'].sum()

        above      = (z.abs() > th).sum()
        crossings  = n_long + n_short

        # Regime stats
        mr_pct   = feat['mr_regime'].mean() * 100
        vol_ok   = (feat['vol_ratio'] < p['vol_cap']).mean() * 100
        conf_l   = (feat['agreement'] <= -2).mean() * 100
        conf_s   = (feat['agreement'] >=  2).mean() * 100

        # How many raw crosses were filtered out by regime/confirmation
        raw_cross_down = feat['cross_down'].sum()
        raw_cross_up   = feat['cross_up'].sum()
        filtered_l = raw_cross_down - n_long
        filtered_s = raw_cross_up   - n_short

        print(f"""
    ── Signal Diagnostics v2 ──
    Bars with data              : {total}
    |z| > {th} (sustained)     : {above:>5}  ({above/total*100:.1f}% of bars)

    Raw crosses (z_slow only)   : {raw_cross_down} long / {raw_cross_up} short
    Filtered by regime/confirm  : {filtered_l} long / {filtered_s} short  ← regime quality
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



