"""
Sliding-window hedge-ratio (beta) stability test.

Compares two ways of tracking a time-varying hedge ratio:
  1. Rolling-window OLS — re-fit from scratch on a fixed-size window,
     stepped forward. Each window is an independent estimate, so it's
     as noisy as the window is short, and jumps whenever an old
     observation drops off the back of the window.
  2. Kalman filter — beta as a hidden random walk, updated one bar at
     a time using all history (with exponentially decaying memory).
     No window-edge effects, and adapts continuously instead of in
     discrete jumps.

Both are plotted together to see whether the Kalman version actually
captures a more fundamental, slower-moving relationship instead of
window-resampling noise.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data import data_loader
from features import features
from params import PARAMS

RESULTS_FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'figures')

WINDOW = 252   # ~1 trading year per window
STEP   = 21    # ~1 month between windows

# Kalman: small delta -> beta barely drifts from the seed (recovers
# static OLS); init_window must be long enough that the seed OLS fit
# is actually identified (empirically >= ~1000 bars for this pair —
# shorter seeds produce a beta whose spread fails the ADF test below).
KALMAN_DELTA       = 1e-6
KALMAN_INIT_WINDOW = 1000

T1 = PARAMS['T1']
T2 = PARAMS['T2']

df = data_loader().load_data(needs_fx=False)
full_beta = features().estimate_hedge_ratio(df, T1, T2, lookback=None)

# ── Rolling OLS ────────────────────────────────────────────
dates, betas = [], []
for start in range(0, len(df) - WINDOW + 1, STEP):
    window = df.iloc[start:start + WINDOW]
    b = features().estimate_hedge_ratio(window, T1, T2, lookback=None)
    dates.append(window.index[-1])
    betas.append(b)

rolling_ols = pd.Series(betas, index=pd.DatetimeIndex(dates), name='rolling_ols_beta')

# ── Kalman filter (causal, full series) ─────────────────────
kalman = features().kalman_hedge_ratio(df, T1, T2, delta=KALMAN_DELTA,
                                        init_window=KALMAN_INIT_WINDOW)

# Sample Kalman at the same checkpoints as the rolling-OLS windows so
# the two are comparable apples-to-apples.
kalman_at_checkpoints = kalman.reindex(rolling_ols.index, method='ffill')

print(f"Full-sample beta        : {full_beta:.4f}")
print(f"Windows                 : {len(rolling_ols)}  (size={WINDOW}d, step={STEP}d)")
print(f"Kalman warm-up          : {KALMAN_INIT_WINDOW}d  (beta NaN before this)")
print()
print(f"{'':<20}{'Rolling OLS':>15}{'Kalman':>15}")
print(f"{'mean':<20}{rolling_ols.mean():>15.4f}{kalman_at_checkpoints.mean():>15.4f}")
print(f"{'median':<20}{rolling_ols.median():>15.4f}{kalman_at_checkpoints.median():>15.4f}")
print(f"{'std':<20}{rolling_ols.std():>15.4f}{kalman_at_checkpoints.std():>15.4f}")
print(f"{'min':<20}{rolling_ols.min():>15.4f}{kalman_at_checkpoints.min():>15.4f}")
print(f"{'max':<20}{rolling_ols.max():>15.4f}{kalman_at_checkpoints.max():>15.4f}")

# Bar-to-bar "jumpiness" — average absolute step size at the same
# checkpoint cadence. Lower means smoother / less noisy.
ols_jump    = rolling_ols.diff().abs().mean()
kalman_jump = kalman_at_checkpoints.diff().abs().mean()
print()
print(f"{'avg |step|':<20}{ols_jump:>15.4f}{kalman_jump:>15.4f}")
print(f"std reduction           : {(1 - kalman_at_checkpoints.std() / rolling_ols.std()) * 100:.1f}%")

# ── Stationarity check ───────────────────────────────────────
# A smoother beta is worthless if the resulting spread stops being
# mean-reverting. ADF on the Kalman-adjusted spread (post warm-up)
# confirms the filter is still tracking the cointegrating relationship,
# not just damping it into something arbitrary.
valid = kalman.dropna()
spread_kalman = (np.log(df[T1]) - kalman * np.log(df[T2])).loc[valid.index]
spread_static = np.log(df[T1]) - full_beta * np.log(df[T2])
adf_kalman = adfuller(spread_kalman, autolag='AIC')
adf_static = adfuller(spread_static.loc[valid.index], autolag='AIC')
print()
print(f"ADF p-value (Kalman spread)        : {adf_kalman[1]:.4f}")
print(f"ADF p-value (static full-beta spread, same window) : {adf_static[1]:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(df.index, np.log(df[T1]), lw=0.8, color='steelblue', label=T1)
axes[0].plot(df.index, np.log(df[T2]), lw=0.8, color='firebrick', label=T2)
axes[0].set_title(f"Log prices  ({T1} vs {T2})")
axes[0].set_ylabel("log price")
axes[0].legend()

axes[1].plot(rolling_ols.index, rolling_ols.values, marker='o', ms=3, lw=1,
             color='darkorange', label=f"rolling OLS (window={WINDOW}d)")
axes[1].plot(kalman.index, kalman.values, lw=1.3, color='seagreen',
             label=f"Kalman (delta={KALMAN_DELTA:g})")
axes[1].axhline(full_beta, color='gray', ls='--', lw=1, label=f"full-sample OLS = {full_beta:.3f}")
axes[1].set_title("Hedge ratio: rolling OLS vs Kalman filter")
axes[1].set_ylabel("beta")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FIGURES, "rolling_beta_stability.png"), dpi=120)
plt.show()
