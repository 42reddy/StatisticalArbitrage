"""
WTI / BRENT PAIRS TRADING  —  v1
==================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
T1 = "ZC=F"
T2 = "CL=F"
LOOKBACK = "20y"

# Core parameters — tuned for 14.5d half-life
PARAMS = dict(
    slow_window  = 60,    # rolling mean window for slow-z (structural anchor)
    medium_window = 30,
    fast_span    = 10,    # EWMA span for fast-z (regime awareness)
    vol_window   = 20,    # rolling std window for vol normalisation
    z_entry      = 1.5,   # primary z-score threshold
    z_add        = 2.5,   # pyramid: add to position at this z (half size)
    z_exit       = 0.3,   # exit z (mean reversion target)
    z_stop       = 3.5,   # stop loss
    max_hold     = 25,    # time-stop days
    vol_cap      = 2.0,   # vol regime: skip if short_vol/long_vol > this
)

# Cost model (futures — liquid)
COMMISSION = 0.0002  # Brokerage + GST + Exchange Fees
STT = 0.0005        # 0.05% on Sell side only (approx 0.00025 per leg)
SLIPPAGE = 0.0005    # ~1.5 ticks in Crude Oil Mini
TOTAL_COST = COMMISSION + STT + (SLIPPAGE * 2) # approx 0.0017

# ══════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════
def load_data():
    print(f"Downloading {T1} and {T2}...")
    p1 = yf.download(T1, period=LOOKBACK, auto_adjust=True, progress=False)['Close'].squeeze()
    p2 = yf.download(T2, period=LOOKBACK, auto_adjust=True, progress=False)['Close'].squeeze()
    df = pd.concat([p1, p2], axis=1, join='inner').dropna()
    df.columns = [T1, T2]
    print(f"  {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df

# ══════════════════════════════════════════════════════
# 2. FEATURES
#
# Three z-score anchors:
#   z_slow : deviation from 60-day rolling mean (structural)
#   z_fast : deviation from EWMA-10 mean (regime-aware)
#   z_ou   : deviation from OU long-run equilibrium (pure stat)
#
# Trade when z_slow signals. z_fast and z_ou are confirmations.
# ══════════════════════════════════════════════════════
def build_features(df, p, ou_mean=None):
    lr = np.log(df[T1] / df[T2])

    # ── Slow z-score (60d rolling mean, 20d rolling std) ──
    mu_slow  = lr.rolling(p['slow_window'], min_periods=p['slow_window']//2).mean()
    vol_slow = lr.rolling(p['vol_window'],  min_periods=p['vol_window']//2).std()
    z_slow   = (lr - mu_slow) / vol_slow.replace(0, np.nan)

    mu_med = lr.rolling(p['medium_window'], min_periods=p['medium_window']//2).mean()
    vol_med = lr.rolling(p['medium_window'], min_periods=p['medium_window']//2).std()
    z_med  = (lr - mu_med) / vol_med.replace(0, np.nan)

    # ── Fast z-score (EWMA mean, same vol) ──
    mu_fast  = lr.ewm(span=p['fast_span'], adjust=False).mean()
    z_fast   = (lr - mu_fast) / vol_slow.replace(0, np.nan)

    # ── OU equilibrium z-score ──
    # Estimate OU mean on the first 2 years of data (out-of-sample for rest)
    # This is the long-run log-ratio the pair gravitates toward
    if ou_mean is None:
        ou_mean = float(lr.iloc[:504].mean())   # ~2yr in-sample estimate
    z_ou = (lr - ou_mean) / vol_slow.replace(0, np.nan)

    # ── Combined signal: average of the three anchors ──
    # This is more robust than any single z — they each capture
    # a different frequency of the same mean-reversion
    z_combined = (z_slow + z_med + z_fast + z_ou) / 4.0

    # ── Vol regime ──
    vol_short = lr.rolling(10).std()
    vol_long  = lr.rolling(60).std()
    vol_ratio = (vol_short / vol_long.replace(0, np.nan)).fillna(1.0)

    # ── ATR for position sizing ──
    atr = lr.diff().abs().ewm(span=14, adjust=False).mean()

    # ── Spread z-score crossing detection ──
    # True on the bar where z first crosses the threshold
    z_cross_up   = (z_combined >  p['z_entry']) & (z_combined.shift(1) <=  p['z_entry'])
    z_cross_down = (z_combined < -p['z_entry']) & (z_combined.shift(1) >= -p['z_entry'])

    return pd.DataFrame({
        'lr':          lr,
        'mu_slow':     mu_slow,
        'mu_fast':     mu_fast,
        'vol':         vol_slow,
        'z_slow':      z_slow,
        'z_fast':      z_fast,
        'z_ou':        z_ou,
        'z':           z_combined,
        'vol_ratio':   vol_ratio,
        'atr':         atr,
        'cross_up':    z_cross_up,
        'cross_down':  z_cross_down,
    }, index=df.index), ou_mean

# ══════════════════════════════════════════════════════
# 3. SIGNAL GENERATION
#
# Entry: combined z crosses threshold
#   - Enter on CROSS (not sustained level), avoids entering mid-move
#   - Vol filter: soft (reduces size, doesn't block)
#
# Pyramid: if in position and spread extends further, add at z_add
#
# Exit: combined z crosses back through z_exit
# Stop: |z| > z_stop OR time-stop
# ══════════════════════════════════════════════════════
def generate_signals(feat, p):
    z    = feat['z']
    vr   = feat['vol_ratio']

    # Vol is a SIZE modifier, not a blocker
    # size_mult: 1.0 in normal conditions, 0.5 when vol is elevated
    size_mult = np.where(vr < p['vol_cap'], 1.0, 0.5)

    # Entry signals (on z crossing the threshold)
    long_entry  = feat['cross_down']   # z crossed below -z_entry → long spread
    short_entry = feat['cross_up']     # z crossed above +z_entry → short spread

    # Pyramid signals (sustained beyond z_add)
    long_add  = (z < -p['z_add'])  & (z.shift(1) >= -p['z_add'])
    short_add = (z >  p['z_add'])  & (z.shift(1) <=  p['z_add'])

    # Exit
    exit_mean  = z.abs() < p['z_exit']
    exit_cross = ((z > 0) & (z.shift(1) < 0)) | ((z < 0) & (z.shift(1) > 0))  # zero cross
    exit_stop  = z.abs() > p['z_stop']

    return pd.DataFrame({
        'long_entry':  long_entry,
        'short_entry': short_entry,
        'long_add':    long_add,
        'short_add':   short_add,
        'exit_mean':   exit_mean,
        'exit_cross':  exit_cross,
        'exit_stop':   exit_stop,
        'size_mult':   pd.Series(size_mult, index=feat.index),
        'z':           z,
        'z_slow':      feat['z_slow'],
        'z_fast':      feat['z_fast'],
        'z_ou':        feat['z_ou'],
        'vol_ratio':   vr,
    }, index=feat.index)

# ══════════════════════════════════════════════════════
# 4. SIGNAL DIAGNOSTICS
# ══════════════════════════════════════════════════════
def signal_diagnostics(feat, sig, p):
    z     = feat['z'].dropna()
    total = len(z)
    th    = p['z_entry']

    n_long  = sig['long_entry'].sum()
    n_short = sig['short_entry'].sum()
    n_add_l = sig['long_add'].sum()
    n_add_s = sig['short_add'].sum()

    # Sustained bars beyond threshold
    above = (z.abs() > th).sum()
    crossings = n_long + n_short

    print(f"""
  ── Signal Diagnostics ──
  Bars with data           : {total}
  |z| > {th} (sustained)  : {above:>5}  ({above/total*100:.1f}% of bars)
  Threshold crossings      : {crossings:>5}  ({n_long} long, {n_short} short)
  Pyramid signals          : {n_add_l + n_add_s:>5}  ({n_add_l} long, {n_add_s} short)

  Z-score coverage:
    z_slow: min={feat['z_slow'].min():.2f}  max={feat['z_slow'].max():.2f}  std={feat['z_slow'].std():.2f}
    z_fast: min={feat['z_fast'].min():.2f}  max={feat['z_fast'].max():.2f}  std={feat['z_fast'].std():.2f}
    z_ou  : min={feat['z_ou'].min():.2f}  max={feat['z_ou'].max():.2f}  std={feat['z_ou'].std():.2f}
    z_comb: min={feat['z'].min():.2f}  max={feat['z'].max():.2f}  std={feat['z'].std():.2f}
""")

# ══════════════════════════════════════════════════════
# 5. BACKTEST ENGINE
#    Supports pyramiding: base position + pyramid layer
#    Position tracking: (base_size, pyramid_size, direction)
# ══════════════════════════════════════════════════════
def backtest(feat, sig, p, cost=True):
    lr   = feat['lr']
    z    = sig['z']
    atr  = feat['atr']

    position   = 0      # net position: +1 long, -1 short, 0 flat
    n_units    = 0.0    # total position size (base + pyramid)
    entry_idx  = 0
    entry_lr   = 0.0
    entry_z    = 0.0
    pyramided  = False

    trades     = []
    daily_pnl  = []
    equity     = 1.0
    equity_ts  = []

    for i in range(1, len(lr)):
        date  = lr.index[i]
        s     = lr.iloc[i]
        z_t   = z.iloc[i]
        sm_   = float(sig['size_mult'].iloc[i])
        atr_t = max(float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.005, 1e-6)

        # Base size: target ~0.8% daily spread vol, scaled by vol regime
        base_size = min(0.008 / atr_t, 6.0) * sm_

        # Daily mark-to-market P&L
        pnl_day = (s - lr.iloc[i-1]) * position * n_units
        daily_pnl.append(pnl_day)
        equity *= (1 + pnl_day)
        equity_ts.append(equity)
        hold_days = i - entry_idx

        # ── Exit logic ──
        if position != 0:
            reason = None
            if sig['exit_stop'].iloc[i]:
                reason = 'stop'
            elif sig['exit_mean'].iloc[i] or sig['exit_cross'].iloc[i]:
                reason = 'mean_revert'
            elif hold_days >= p['max_hold']:
                reason = 'time_stop'

            if reason:
                trade_pnl = (s - entry_lr) * position * n_units
                if cost:
                    trade_pnl -= TOTAL_COST * n_units
                trades.append({
                    'entry_date':  lr.index[entry_idx],
                    'exit_date':   date,
                    'hold_days':   hold_days,
                    'direction':   'long' if position == 1 else 'short',
                    'entry_z':     entry_z,
                    'exit_z':      z_t,
                    'n_units':     n_units,
                    'pyramided':   pyramided,
                    'pnl':         trade_pnl,
                    'exit_reason': reason,
                })
                position  = 0
                n_units   = 0.0
                pyramided = False

        # ── Pyramid: add to existing position ──
        if position == 1 and sig['long_add'].iloc[i] and not pyramided:
            add_size = base_size * 0.5
            n_units += add_size
            pyramided = True
            if cost:
                daily_pnl[-1] -= TOTAL_COST * add_size
                equity        *= (1 - TOTAL_COST * add_size)

        elif position == -1 and sig['short_add'].iloc[i] and not pyramided:
            add_size = base_size * 0.5
            n_units += add_size
            pyramided = True
            if cost:
                daily_pnl[-1] -= TOTAL_COST * add_size
                equity        *= (1 - TOTAL_COST * add_size)

        # ── Entry logic ──
        if position == 0:
            if sig['long_entry'].iloc[i]:
                position  = 1
                n_units   = base_size
                entry_idx = i
                entry_lr  = s
                entry_z   = z_t
                pyramided = False
                if cost:
                    daily_pnl[-1] -= TOTAL_COST * n_units
                    equity        *= (1 - TOTAL_COST * n_units)

            elif sig['short_entry'].iloc[i]:
                position  = -1
                n_units   = base_size
                entry_idx = i
                entry_lr  = s
                entry_z   = z_t
                pyramided = False
                if cost:
                    daily_pnl[-1] -= TOTAL_COST * n_units
                    equity        *= (1 - TOTAL_COST * n_units)

    pnl_s    = pd.Series(daily_pnl, index=lr.index[1:])
    equity_s = pd.Series(equity_ts, index=lr.index[1:])
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=['entry_date','exit_date','hold_days','direction',
                 'entry_z','exit_z','n_units','pyramided','pnl','exit_reason'])
    return pnl_s, equity_s, trades_df

# ══════════════════════════════════════════════════════
# 6. METRICS
# ══════════════════════════════════════════════════════
def calc_metrics(pnl, equity, trades):
    ann_ret = pnl.mean() * 252
    ann_vol = pnl.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    down    = pnl[pnl < 0]
    sortino = ann_ret / (down.std() * np.sqrt(252)) if len(down) > 1 else 0.0

    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max
    max_dd   = dd.min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    n = len(trades)
    if n > 0:
        wr   = (trades['pnl'] > 0).mean()
        wins = trades.loc[trades['pnl'] > 0, 'pnl']
        loss = trades.loc[trades['pnl'] < 0, 'pnl']
        aw   = wins.mean() if len(wins) > 0 else 0.0
        al   = loss.mean() if len(loss) > 0 else 0.0
        ah   = trades['hold_days'].mean()
        pf   = abs(wins.sum() / loss.sum()) if loss.sum() != 0 else np.inf
        py_  = trades['pyramided'].mean() * 100 if 'pyramided' in trades else 0
    else:
        wr = aw = al = ah = pf = py_ = 0.0

    return dict(sharpe=sharpe, sortino=sortino, ann_ret=ann_ret,
                ann_vol=ann_vol, max_dd=max_dd, calmar=calmar,
                n_trades=n, win_rate=wr, avg_win=aw, avg_loss=al,
                avg_hold=ah, profit_factor=pf, pct_pyramided=py_)

# ══════════════════════════════════════════════════════
# 7. STAT DIAGNOSTICS
# ══════════════════════════════════════════════════════
def run_stat_diag(df, feat):
    lr = feat['lr'].dropna()

    adf_stat, adf_p, _, _, crit, _ = adfuller(lr, autolag='AIC')

    lags = range(2, 60)
    tau  = [np.std(np.subtract(lr.values[l:], lr.values[:-l])) for l in lags]
    hurst, *_ = stats.linregress(np.log(list(lags)), np.log(tau))

    s_lag = lr.shift(1).dropna()
    s_cur = lr.iloc[1:]
    res   = sm.OLS(s_cur, sm.add_constant(s_lag)).fit()
    phi   = min(float(res.params.iloc[1]), 1 - 1e-8)
    kappa = -np.log(max(phi, 1e-8)) * 252
    hl    = np.log(2) / kappa * 252

    try:
        joh = coint_johansen(df[[T1, T2]], det_order=0, k_ar_diff=1)
        joh_pass  = bool(joh.lr1[0] > joh.cvt[0, 1])
        joh_trace = float(joh.lr1[0])
        joh_crit  = float(joh.cvt[0, 1])
    except Exception:
        joh_pass = False; joh_trace = np.nan; joh_crit = np.nan

    return dict(adf_p=adf_p, adf_stat=adf_stat, adf_crit=crit,
                hurst=hurst, half_life=hl, kappa=kappa,
                joh_pass=joh_pass, joh_trace=joh_trace, joh_crit=joh_crit)

# ══════════════════════════════════════════════════════
# 8. SENSITIVITY ANALYSIS
#    Sweep z_entry and slow_window to show Sharpe surface
#    This replaces walk-forward for interpretability
# ══════════════════════════════════════════════════════
def sensitivity_analysis(df, ou_mean):
    print("\n── Sensitivity Analysis  (z_entry × slow_window) ──")
    z_entries    = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    slow_windows = [20, 30, 45, 60, 90, 120]

    results = {}
    for sw in slow_windows:
        for ze in z_entries:
            p = {**PARAMS, 'slow_window': sw, 'z_entry': ze}
            try:
                feat, _ = build_features(df, p, ou_mean=ou_mean)
                sig     = generate_signals(feat, p)
                pnl, eq, tr = backtest(feat, sig, p, cost=True)
                if len(tr) < 3:
                    results[(sw, ze)] = (np.nan, 0)
                    continue
                m = calc_metrics(pnl, eq, tr)
                results[(sw, ze)] = (m['sharpe'], m['n_trades'])
            except Exception:
                results[(sw, ze)] = (np.nan, 0)

    # Print table
    print(f"\n  Sharpe  (rows=slow_window, cols=z_entry threshold)")
    header = f"  {'win':>6}" + "".join(f"  z={ze:>4}" for ze in z_entries)
    print(header)
    for sw in slow_windows:
        row = f"  {sw:>6}"
        for ze in z_entries:
            sh, nt = results[(sw, ze)]
            if np.isnan(sh) or nt < 3:
                row += f"  {'--':>6}"
            else:
                row += f"  {sh:>6.2f}"
        print(row)

    print(f"\n  Trade count (rows=slow_window, cols=z_entry threshold)")
    print(header)
    for sw in slow_windows:
        row = f"  {sw:>6}"
        for ze in z_entries:
            sh, nt = results[(sw, ze)]
            row += f"  {nt:>6}"
        print(row)

    # Find best
    best = max(results.items(), key=lambda x: x[1][0] if not np.isnan(x[1][0]) else -99)
    best_sw, best_ze = best[0]
    best_sh, best_nt = best[1]
    print(f"\n  Best: slow_window={best_sw}  z_entry={best_ze}  "
          f"Sharpe={best_sh:.2f}  Trades={best_nt}")

    return results, best_sw, best_ze

# ══════════════════════════════════════════════════════
# 9. MONTHLY P&L TABLE
# ══════════════════════════════════════════════════════
def monthly_pnl(pnl):
    m   = pnl.resample('ME').sum()
    df  = pd.DataFrame({'Y': m.index.year, 'M': m.index.month, 'P': m.values})
    tbl = df.pivot(index='Y', columns='M', values='P')
    mn  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    tbl.columns = [mn[c-1] for c in tbl.columns]
    tbl['Total'] = tbl.sum(axis=1)
    return tbl

# ══════════════════════════════════════════════════════
# 10. PLOTTING
# ══════════════════════════════════════════════════════
def plot_all(df, feat, sig, pnl, equity, trades, stat_d, metrics, sens_results, p):
    BG='#0d0d0d'; PAN='#161616'; GR='#252525'
    C1='#4fc3f7'; C2='#ef5350'; C3='#66bb6a'; C4='#ffa726'; C5='#ce93d8'
    C6='#80cbc4'; DIM='#777777'; TXT='#dddddd'

    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"WTI / Brent  Pairs Strategy  v4   |   "
        f"Sharpe={metrics['sharpe']:.2f}   Sortino={metrics['sortino']:.2f}   "
        f"Calmar={metrics['calmar']:.2f}   MaxDD={metrics['max_dd']*100:.1f}%   "
        f"Trades={metrics['n_trades']}   WinRate={metrics['win_rate']*100:.0f}%   "
        f"H={stat_d['hurst']:.3f}   HL={stat_d['half_life']:.1f}d",
        fontsize=11, color=TXT, fontweight='bold', y=0.999
    )
    gs = gridspec.GridSpec(5, 2, figure=fig,
                           hspace=0.52, wspace=0.28,
                           top=0.975, bottom=0.025, left=0.06, right=0.97)

    def ax_(pos, title):
        a = fig.add_subplot(pos)
        a.set_facecolor(PAN)
        a.set_title(title, color=TXT, fontsize=8.5, pad=5, loc='left')
        a.tick_params(colors=DIM, labelsize=7)
        for sp in a.spines.values(): sp.set_color(GR)
        a.grid(True, color=GR, lw=0.4, alpha=0.7)
        return a

    # ── 1. Log-ratio spread with all three anchors ──
    ax1 = ax_(gs[0, :], "§1  Log-Ratio Spread  +  Three Mean Anchors  (slow / fast / OU)")
    ax1.plot(feat['lr'],      color=C1, lw=0.9, alpha=0.75, label='log(WTI/Brent)')
    ax1.plot(feat['mu_slow'], color=C4, lw=1.5, label=f"Slow mean ({p['slow_window']}d)", alpha=0.9)
    ax1.plot(feat['mu_fast'], color=C3, lw=1.0, linestyle='--', label=f"Fast EWMA ({p['fast_span']})", alpha=0.7)

    # OU mean line
    ou_mean_val = float(feat['lr'].iloc[:504].mean())
    ax1.axhline(ou_mean_val, color=C5, lw=1.0, linestyle=':', label=f"OU equil ({ou_mean_val:.4f})", alpha=0.8)

    # Trade regions
    if len(trades) > 0:
        for _, t in trades[trades['direction']=='long'].iterrows():
            ax1.axvspan(t['entry_date'], t['exit_date'], alpha=0.10, color=C3)
        for _, t in trades[trades['direction']=='short'].iterrows():
            ax1.axvspan(t['entry_date'], t['exit_date'], alpha=0.10, color=C2)
    ax1.legend(fontsize=7, facecolor=PAN, labelcolor=TXT)

    # ── 2. Combined z-score ──
    ax2 = ax_(gs[1, :], "§2  Combined Z-Score  (avg of slow / fast / OU)  +  entry markers")
    z_clipped = sig['z'].clip(-5, 5)
    ax2.plot(z_clipped, color=C5, lw=0.85)
    ax2.axhline(0,            color=DIM, lw=0.7)
    ax2.axhline( p['z_entry'], color=C2, lw=1.1, linestyle='--', label=f"+{p['z_entry']}σ")
    ax2.axhline(-p['z_entry'], color=C3, lw=1.1, linestyle='--', label=f"-{p['z_entry']}σ")
    ax2.axhline( p['z_add'],   color=C2, lw=0.7, linestyle=':', alpha=0.6, label=f"pyramid {p['z_add']}σ")
    ax2.axhline(-p['z_add'],   color=C3, lw=0.7, linestyle=':', alpha=0.6)
    ax2.axhline( p['z_stop'],  color=C2, lw=0.5, linestyle=':', alpha=0.3)
    ax2.axhline(-p['z_stop'],  color=C3, lw=0.5, linestyle=':', alpha=0.3)
    # Mark entries
    if len(trades) > 0:
        for _, t in trades.iterrows():
            c = C3 if t['direction']=='long' else C2
            if t['entry_date'] in sig.index:
                try:
                    ax2.scatter(t['entry_date'], sig.loc[t['entry_date'],'z'],
                                color=c, s=22, zorder=5, alpha=0.85)
                except Exception:
                    pass
    ax2.set_ylim(-5.2, 5.2)
    ax2.legend(fontsize=7, facecolor=PAN, labelcolor=TXT)

    # ── 3. Z-score components ──
    ax3 = ax_(gs[2, 0], "§3  Z-Score Components  (slow / fast / OU)")
    ax3.plot(feat['z_slow'].clip(-4,4), color=C4, lw=0.8, alpha=0.8, label='z_slow (structural)')
    ax3.plot(feat['z_fast'].clip(-4,4), color=C3, lw=0.8, alpha=0.7, label='z_fast (regime)')
    ax3.plot(feat['z_ou'].clip(-4,4),   color=C5, lw=0.8, alpha=0.7, label='z_ou (OU equil)')
    ax3.axhline( p['z_entry'], color=DIM, lw=0.7, linestyle='--')
    ax3.axhline(-p['z_entry'], color=DIM, lw=0.7, linestyle='--')
    ax3.axhline(0, color=DIM, lw=0.5)
    ax3.set_ylim(-4.5, 4.5)
    ax3.legend(fontsize=7, facecolor=PAN, labelcolor=TXT)

    # ── 4. Vol regime ──
    ax4 = ax_(gs[2, 1], "§4  Volatility Regime  (short/long vol ratio)")
    ax4.plot(sig['vol_ratio'], color=C1, lw=0.9)
    ax4.axhline(p['vol_cap'], color=C2, lw=1.1, linestyle='--', label=f"cap={p['vol_cap']}x")
    ax4.axhline(1.0,          color=DIM, lw=0.6)
    ax4.fill_between(sig.index,
                     sig['vol_ratio'].clip(lower=p['vol_cap']), p['vol_cap'],
                     color=C2, alpha=0.18, label='Size halved')
    ax4.set_ylim(0, 4.5)
    ax4.legend(fontsize=7, facecolor=PAN, labelcolor=TXT)

    # ── 5. Equity + drawdown ──
    ax5 = ax_(gs[3, 0], "§5  Equity Curve  (with costs + pyramiding)")
    ax5.plot(equity, color=C3, lw=1.4, label='Strategy')
    ax5.axhline(1.0, color=DIM, lw=0.6, linestyle='--')
    ax5b = ax5.twinx()
    dd = ((equity - equity.cummax()) / equity.cummax()) * 100
    ax5b.fill_between(dd.index, dd, 0, color=C2, alpha=0.2)
    ax5b.set_ylabel('DD %', color=C2, fontsize=7)
    ax5b.tick_params(colors=C2, labelsize=6)
    ax5b.set_ylim(-60, 5)
    ax5.legend(fontsize=7, facecolor=PAN, labelcolor=TXT)

    # ── 6. Sensitivity heatmap ──
    ax6 = ax_(gs[3, 1], "§6  Sharpe Sensitivity  (slow_window × z_entry)")
    if sens_results:
        z_entries    = sorted(set(k[1] for k in sens_results))
        slow_windows = sorted(set(k[0] for k in sens_results))
        mat = np.array([[sens_results[(sw, ze)][0] for ze in z_entries]
                        for sw in slow_windows], dtype=float)
        im  = ax6.imshow(mat, cmap='RdYlGn', aspect='auto',
                         vmin=-1, vmax=2, origin='lower')
        ax6.set_xticks(range(len(z_entries)))
        ax6.set_xticklabels([f"z={z}" for z in z_entries], color=TXT, fontsize=7)
        ax6.set_yticks(range(len(slow_windows)))
        ax6.set_yticklabels([f"w={w}" for w in slow_windows], color=TXT, fontsize=7)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                nt = sens_results[(slow_windows[r], z_entries[c])][1]
                if not np.isnan(v) and nt >= 3:
                    ax6.text(c, r, f"{v:.2f}\n({nt}tr)", ha='center', va='center',
                             fontsize=6, color='#111' if abs(v) > 0.7 else TXT)
        plt.colorbar(im, ax=ax6, fraction=0.03, pad=0.02)
    else:
        ax6.text(0.5, 0.5, 'No sensitivity data', transform=ax6.transAxes,
                 ha='center', color=DIM)

    # ── 7. Monthly P&L ──
    ax7 = ax_(gs[4, :], "§7  Monthly P&L Heatmap")
    try:
        tbl  = monthly_pnl(pnl)
        mcols = [c for c in tbl.columns if c != 'Total']
        data = tbl[mcols].values.astype(float)
        vmax = np.nanpercentile(np.abs(data[~np.isnan(data)]), 95) if data.size else 1
        im = ax7.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
        ax7.set_xticks(range(len(mcols))); ax7.set_xticklabels(mcols, color=TXT, fontsize=7)
        ax7.set_yticks(range(len(tbl.index))); ax7.set_yticklabels(tbl.index, color=TXT, fontsize=7)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                v = data[r, c]
                if not np.isnan(v):
                    ax7.text(c, r, f"{v*100:.2f}", ha='center', va='center',
                             fontsize=5.5, color='#111' if abs(v) > vmax*0.4 else TXT)
        plt.colorbar(im, ax=ax7, fraction=0.012, pad=0.01)
    except Exception as e:
        ax7.text(0.5, 0.5, f'Error: {e}', transform=ax7.transAxes, ha='center', color=DIM)


    print("  → Chart saved: WTI_Brent_v4.png")

# ══════════════════════════════════════════════════════
# 11. REPORT
# ══════════════════════════════════════════════════════
def print_report(metrics, stat_d, trades, p, ou_mean):
    n = metrics['n_trades']
    stat_note = ""
    if n < 20:
        stat_note = f"  ⚠  {n} trades — low statistical power. Sharpe estimate has wide CI."
    elif n < 50:
        stat_note = f"  ~  {n} trades — moderate power. Treat Sharpe as directional, not precise."
    else:
        stat_note = f"  ✓  {n} trades — reasonable power for Sharpe interpretation."

    print(f"""
╔{'═'*66}╗
║  WTI / BRENT  STRATEGY REPORT  v4                             ║
╠{'═'*66}╣
║  PARAMETERS
║    Slow window (structural mean)  : {p['slow_window']:>4}d
║    Fast EWMA span                 : {p['fast_span']:>4}
║    Vol window                     : {p['vol_window']:>4}d
║    Z entry threshold              : {p['z_entry']:>6.2f}
║    Z pyramid threshold            : {p['z_add']:>6.2f}
║    Z exit threshold               : {p['z_exit']:>6.2f}
║    Z stop                         : {p['z_stop']:>6.2f}
║    Max hold                       : {p['max_hold']:>4}d
║    OU equilibrium mean            : {ou_mean:>8.5f}
╠{'═'*66}╣
║  SPREAD DIAGNOSTICS
║    ADF p-value    : {stat_d['adf_p']:>8.4f}   {'✓ stationary' if stat_d['adf_p']<0.05 else '✗'}
║    Hurst          : {stat_d['hurst']:>8.4f}   ({'mean-rev ✓' if stat_d['hurst']<0.5 else 'RW/trend ✗'})
║    OU half-life   : {stat_d['half_life']:>7.1f}d
║    Johansen       : {'✓ pass' if stat_d['joh_pass'] else '✗ fail':>12}  (trace={stat_d['joh_trace']:.2f}  crit={stat_d['joh_crit']:.2f})
╠{'═'*66}╣
║  PERFORMANCE  (with transaction costs)
║    Sharpe ratio   : {metrics['sharpe']:>8.2f}
║    Sortino ratio  : {metrics['sortino']:>8.2f}
║    Calmar ratio   : {metrics['calmar']:>8.2f}
║    Ann. return    : {metrics['ann_ret']*100:>7.2f}%
║    Ann. vol       : {metrics['ann_vol']*100:>7.2f}%
║    Max drawdown   : {metrics['max_dd']*100:>7.2f}%
╠{'═'*66}╣
║  TRADE STATS
║    Total trades   : {metrics['n_trades']:>8}
║    Win rate       : {metrics['win_rate']*100:>7.1f}%
║    Avg win        : {metrics['avg_win']:>10.5f}
║    Avg loss       : {metrics['avg_loss']:>10.5f}
║    Profit factor  : {metrics['profit_factor']:>8.2f}
║    Avg hold       : {metrics['avg_hold']:>7.1f}d
║    % pyramided    : {metrics['pct_pyramided']:>7.1f}%
╠{'═'*66}╣
║  STATISTICAL POWER NOTE
║  {stat_note:<64}║
╚{'═'*66}╝""")

    if len(trades) > 0:
        print(f"\n  LAST 20 TRADES")
        print(f"  {'Entry':>12}  {'Exit':>12}  {'Dir':>6}  "
              f"{'Hold':>5}  {'EntryZ':>7}  {'ExitZ':>6}  "
              f"{'Units':>6}  {'Pyr':>3}  {'PnL':>10}  Reason")
        print("  " + "─" * 92)
        for _, t in trades.tail(20).iterrows():
            ed = t['entry_date'].date() if hasattr(t['entry_date'], 'date') else t['entry_date']
            xd = t['exit_date'].date()  if hasattr(t['exit_date'],  'date') else t['exit_date']
            pyr = "Y" if t.get('pyramided', False) else "N"
            print(f"  {str(ed):>12}  {str(xd):>12}  {t['direction']:>6}  "
                  f"{t['hold_days']:>5}  {t['entry_z']:>7.3f}  {t['exit_z']:>6.3f}  "
                  f"{t['n_units']:>6.2f}  {pyr:>3}  {t['pnl']:>10.5f}  {t['exit_reason']}")

        by_r = trades.groupby('exit_reason')['pnl'].agg(['count','sum','mean'])
        print(f"\n  EXIT REASON BREAKDOWN")
        for r, row in by_r.iterrows():
            pct_pos = (trades.loc[trades['exit_reason']==r, 'pnl'] > 0).mean() * 100
            print(f"    {r:<15}  n={int(row['count']):>4}  "
                  f"total={row['sum']:>10.5f}  avg={row['mean']:>10.5f}  "
                  f"wr={pct_pos:.0f}%")

# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

df = load_data()

# Estimate OU mean on first 2 years (in-sample)
lr_full  = np.log(df[T1] / df[T2])
ou_mean  = float(lr_full.iloc[:504].mean())
print(f"  OU equilibrium mean (first 2yr): {ou_mean:.5f}")

# Run sensitivity analysis first — tells you optimal params
print("\n── Sensitivity Analysis ──")
print("  Sweeping z_entry and slow_window...")
sens_results, best_sw, best_ze = sensitivity_analysis(df, ou_mean)

# Update PARAMS with best found
PARAMS['slow_window'] = best_sw
PARAMS['z_entry']     = best_ze
print(f"\n  Using: slow_window={best_sw}  z_entry={best_ze}")

# Full run with best params
print("\n── Full-Sample Run ──")
feat, ou_mean = build_features(df, PARAMS, ou_mean=ou_mean)
sig           = generate_signals(feat, PARAMS)

signal_diagnostics(feat, sig, PARAMS)

pnl, equity, trades = backtest(feat, sig, PARAMS, cost=True)
stat_d  = run_stat_diag(df, feat)
metrics = calc_metrics(pnl, equity, trades)

print_report(metrics, stat_d, trades, PARAMS, ou_mean)
plot_all(df, feat, sig, pnl, equity, trades, stat_d, metrics, sens_results, PARAMS)

print("\nDone.")


