"""
Sliding-window OU half-life test.

Re-estimates the OU half-life of the beta-adjusted spread on a rolling
window across the full sample, to check whether mean-reversion speed is
stable over time or drifts/breaks down in parts of the data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data import data_loader
from features import features
from params import PARAMS

RESULTS_FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'figures')

WINDOW = 252   # ~1 trading year per window
STEP   = 21    # ~1 month between windows

T1 = PARAMS['T1']
T2 = PARAMS['T2']


def ou_half_life(spread):
    y     = spread.values
    dy    = np.diff(y)
    y_lag = y[:-1]
    slope, *_ = scipy_stats.linregress(y_lag, dy)
    lam = -slope
    return np.log(2) / lam if lam > 0 else np.nan


df = data_loader().load_data(needs_fx=False)
beta = features().estimate_hedge_ratio(df, T1, T2, lookback=None)
spread = np.log(df[T1]) - beta * np.log(df[T2])

dates, hls = [], []
for start in range(0, len(spread) - WINDOW + 1, STEP):
    window = spread.iloc[start:start + WINDOW]
    dates.append(window.index[-1])
    hls.append(ou_half_life(window))

result = pd.Series(hls, index=pd.DatetimeIndex(dates), name='ou_half_life')

print(f"Beta              : {beta:.4f}")
print(f"Windows           : {len(result)}  (size={WINDOW}d, step={STEP}d)")
print(f"Half-life mean    : {result.mean():.1f}d")
print(f"Half-life median  : {result.median():.1f}d")
print(f"Half-life std     : {result.std():.1f}d")
print(f"Half-life min/max : {result.min():.1f}d / {result.max():.1f}d")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(spread.index, spread.values, lw=0.8, color='steelblue')
axes[0].set_title(f"Beta-adjusted spread  ({T1} vs {T2})")
axes[0].set_ylabel("spread")

axes[1].plot(result.index, result.values, marker='o', ms=3, lw=1, color='darkorange')
axes[1].axhline(result.median(), color='gray', ls='--', lw=1, label=f"median = {result.median():.0f}d")
axes[1].set_title(f"Rolling OU half-life  (window={WINDOW}d, step={STEP}d)")
axes[1].set_ylabel("half-life (days)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FIGURES, "halflife_stability.png"), dpi=120)
plt.show()
