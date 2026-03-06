import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import pandas as pd


class plotting():

    def __init__(self):
        # ── Palette ──
        self.BG  = '#0a0a0f'
        self.PAN = '#12121a'
        self.GR  = '#1e1e2e'
        self.C1  = '#4fc3f7'   # ice blue   — spread / neutral
        self.C2  = '#ef5350'   # red        — short / loss / stop
        self.C3  = '#66bb6a'   # green      — long / win
        self.C4  = '#ffa726'   # amber      — slow mean
        self.C5  = '#ce93d8'   # lavender   — OU / z_ou
        self.C6  = '#80cbc4'   # teal       — fast mean
        self.C7  = '#fff176'   # yellow     — equity / P&L
        self.DIM = '#555566'
        self.TXT = '#e0e0f0'

    # ─────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ─────────────────────────────────────────────────────
    def _ax(self, fig, pos, title):
        a = fig.add_subplot(pos)
        a.set_facecolor(self.PAN)
        a.set_title(title, color=self.TXT, fontsize=8, pad=5, loc='left', fontweight='bold')
        a.tick_params(colors=self.DIM, labelsize=7)
        for sp in a.spines.values():
            sp.set_color(self.GR)
        a.grid(True, color=self.GR, lw=0.4, alpha=0.6)
        return a

    def _shade_trades(self, ax, trades):
        """Shade long/short regions on any axis."""
        if trades is None or len(trades) == 0:
            return
        for _, t in trades[trades['direction'] == 'long'].iterrows():
            ax.axvspan(t['entry_date'], t['exit_date'], alpha=0.08, color=self.C3, lw=0)
        for _, t in trades[trades['direction'] == 'short'].iterrows():
            ax.axvspan(t['entry_date'], t['exit_date'], alpha=0.08, color=self.C2, lw=0)

    def _hline(self, ax, y, color, lw=1.0, ls='--', label=None, alpha=1.0):
        ax.axhline(y, color=color, lw=lw, linestyle=ls, alpha=alpha,
                   label=label if label else None)

    # ─────────────────────────────────────────────────────
    #  MAIN PLOT
    #
    #  Signature matches main.py call:
    #    plot_all(df, feat, sig, pnl, equity, trades,
    #             stat_d, metrics, sens_results, p,
    #             wf_pnl=None, wf_equity=None, wf_params_df=None)
    #
    #  sens_results accepted but unused (Bayes replaced grid search)
    #  wf_* used for OOS equity overlay if provided
    # ─────────────────────────────────────────────────────
    def plot_all(self, df, feat, sig, pnl, equity, trades, stat_d, metrics,
                 sens_results, p,
                 wf_pnl=None, wf_equity=None, wf_params_df=None,
                 label=""):

        CAPITAL = 200_000   # must match backtest.CAPITAL

        fig = plt.figure(figsize=(22, 30))
        fig.patch.set_facecolor(self.BG)

        title_label = f"  [{label}]" if label else ""
        fig.suptitle(
            f"WTI / Brent  Stat-Arb  v5{title_label}   |   "
            f"Sharpe {metrics['sharpe']:.2f}   "
            f"Calmar {metrics['calmar']:.2f}   "
            f"Sortino {metrics['sortino']:.2f}   "
            f"MaxDD {metrics['max_dd']*100:.1f}%   "
            f"Trades {metrics['n_trades']}   "
            f"WR {metrics['win_rate']*100:.0f}%   "
            f"H={stat_d['hurst']:.3f}   HL={stat_d['half_life']:.1f}d",
            fontsize=10, color=self.TXT, fontweight='bold', y=0.999
        )

        gs = gridspec.GridSpec(
            6, 2, figure=fig,
            hspace=0.55, wspace=0.30,
            top=0.975, bottom=0.02, left=0.06, right=0.97
        )

        # ── §1  Log-ratio spread + mean anchors ─────────────
        ax1 = self._ax(fig, gs[0, :],
                       "§1  Log-Ratio Spread  (ZC=F / CL=F)  +  Slow / Fast / OU anchors")
        ax1.plot(feat['lr'], color=self.C1, lw=0.8, alpha=0.7, label='log(Brent/WTI)')
        ax1.plot(feat['mu_slow'], color=self.C4, lw=1.6,
                 label=f"Slow {p['slow_window']}d", alpha=0.95)
        ax1.plot(feat['mu_fast'], color=self.C6, lw=1.0, ls='--',
                 label=f"Fast EWMA {p['fast_span']}", alpha=0.8)

        # OU equilibrium — use feat value not recomputed
        ou_val = float(feat['z_ou'].mean() * feat['vol'].mean() + feat['mu_slow'].mean())
        if 'z_ou' in feat.columns and 'vol' in feat.columns:
            ou_series = feat['mu_slow'] + feat['z_ou'] * feat['vol']
            ou_flat   = ou_series.mean()
        else:
            ou_flat = float(feat['lr'].iloc[:504].mean())
        ax1.axhline(ou_flat, color=self.C5, lw=1.0, ls=':', alpha=0.8,
                    label=f"OU equil ≈{ou_flat:.4f}")

        self._shade_trades(ax1, trades)
        ax1.legend(fontsize=7, facecolor=self.PAN, labelcolor=self.TXT, loc='upper left')
        ax1.set_ylabel('log ratio', color=self.DIM, fontsize=7)

        # ── §2  Asymmetric z-score entries/exits ─────────────
        ax2 = self._ax(fig, gs[1, :],
                       "§2  Z-Slow (primary)  +  Asymmetric entry/exit thresholds  +  trade markers")

        z_slow = sig['z_slow'].clip(-5, 5)
        ax2.plot(z_slow, color=self.C5, lw=0.85, alpha=0.9, label='z_slow')
        ax2.axhline(0, color=self.DIM, lw=0.6)

        # Asymmetric entry lines
        ze_long  = p.get('z_entry_long',  p.get('z_entry', 1.25))
        ze_short = p.get('z_entry_short', p.get('z_entry', 1.25))
        zx_long  = p.get('z_exit_long',   p.get('z_exit',  0.30))
        zx_short = p.get('z_exit_short',  p.get('z_exit',  0.30))
        zs_long  = p.get('z_stop_long',   p.get('z_stop',  3.00))
        zs_short = p.get('z_stop_short',  p.get('z_stop',  3.00))

        self._hline(ax2, -ze_long,  self.C3, lw=1.3, ls='--', label=f"entry long  −{ze_long:.2f}σ")
        self._hline(ax2,  ze_short, self.C2, lw=1.3, ls='--', label=f"entry short +{ze_short:.2f}σ")
        self._hline(ax2, -zx_long,  self.C3, lw=0.8, ls=':',  label=f"exit long   −{zx_long:.2f}σ", alpha=0.7)
        self._hline(ax2,  zx_short, self.C2, lw=0.8, ls=':',  label=f"exit short  +{zx_short:.2f}σ", alpha=0.7)
        self._hline(ax2, -zs_long,  self.C2, lw=0.5, ls=':',  alpha=0.35)
        self._hline(ax2,  zs_short, self.C2, lw=0.5, ls=':',  alpha=0.35)

        # Entry scatter markers
        if trades is not None and len(trades) > 0:
            for _, t in trades.iterrows():
                c   = self.C3 if t['direction'] == 'long' else self.C2
                edt = t['entry_date']
                if edt in sig.index:
                    try:
                        yv = float(sig.loc[edt, 'z_slow'])
                        ax2.scatter(edt, np.clip(yv, -5, 5), color=c, s=25, zorder=5, alpha=0.9)
                    except Exception:
                        pass

        ax2.set_ylim(-5.3, 5.3)
        ax2.legend(fontsize=6.5, facecolor=self.PAN, labelcolor=self.TXT,
                   loc='upper left', ncol=3)
        ax2.set_ylabel('z-score', color=self.DIM, fontsize=7)

        # ── §3  Z-score components ────────────────────────────
        ax3 = self._ax(fig, gs[2, 0],
                       "§3  Z-Score Components  (slow / med / fast / OU)")
        for col, color, lbl in [
            ('z_slow', self.C4,  'z_slow (structural)'),
            ('z_med',  self.C6,  'z_med'),
            ('z_fast', self.C3,  'z_fast (EWMA)'),
            ('z_ou',   self.C5,  'z_ou (OU)'),
        ]:
            if col in feat.columns:
                ax3.plot(feat[col].clip(-4.5, 4.5), lw=0.7, alpha=0.75,
                         color=color, label=lbl)
        ax3.axhline(0, color=self.DIM, lw=0.5)
        ax3.set_ylim(-4.8, 4.8)
        ax3.legend(fontsize=6.5, facecolor=self.PAN, labelcolor=self.TXT)
        ax3.set_ylabel('z-score', color=self.DIM, fontsize=7)

        # ── §4  Regime: vol ratio + autocorr ──────────────────
        ax4  = self._ax(fig, gs[2, 1],
                        "§4  Regime Filters  (vol ratio  |  autocorrelation)")
        ax4b = ax4.twinx()

        if 'vol_ratio' in sig.columns:
            ax4.plot(sig['vol_ratio'].clip(0, 5), color=self.C1, lw=0.9,
                     alpha=0.85, label='vol ratio')
            ax4.axhline(p.get('vol_cap', 1.5), color=self.C2, lw=1.1, ls='--',
                        label=f"vol_cap={p.get('vol_cap',1.5):.2f}")
            ax4.fill_between(sig.index,
                             sig['vol_ratio'].clip(0, 5).clip(lower=p.get('vol_cap', 1.5)),
                             p.get('vol_cap', 1.5),
                             color=self.C2, alpha=0.15)
            ax4.set_ylim(0, 5)
            ax4.set_ylabel('vol ratio', color=self.C1, fontsize=7)
            ax4.tick_params(axis='y', colors=self.C1, labelsize=6)

        if 'autocorr' in feat.columns:
            ax4b.plot(feat['autocorr'].clip(-1, 1), color=self.C5, lw=0.7,
                      alpha=0.7, label='autocorr')
            ax4b.axhline(p.get('autocorr_threshold', 0.1), color=self.C5,
                         lw=0.8, ls=':', alpha=0.6)
            ax4b.axhline(0, color=self.DIM, lw=0.4)
            ax4b.set_ylim(-1.2, 1.2)
            ax4b.set_ylabel('autocorr', color=self.C5, fontsize=7)
            ax4b.tick_params(axis='y', colors=self.C5, labelsize=6)

        lines1, labs1 = ax4.get_legend_handles_labels()
        lines2, labs2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labs1 + labs2, fontsize=6.5,
                   facecolor=self.PAN, labelcolor=self.TXT)

        # ── §5  Accumulated $ P&L  (PRIMARY performance chart) ─
        ax5 = self._ax(fig, gs[3, :],
                       "§5  Accumulated P&L  ($)  |  capital deployed per bar  |  drawdown")

        # Accumulated dollar P&L — this is what we care about
        cum_pnl_dol = (pnl * CAPITAL).cumsum()
        ax5.plot(cum_pnl_dol, color=self.C7, lw=1.6, label='Cumulative P&L ($)', zorder=4)
        ax5.axhline(0, color=self.DIM, lw=0.7, ls='--')
        ax5.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'${x:,.0f}'))
        ax5.set_ylabel('Cumulative P&L ($)', color=self.C7, fontsize=7)
        ax5.tick_params(axis='y', colors=self.C7, labelsize=7)

        # Drawdown overlay (right axis)
        ax5b = ax5.twinx()
        dd = ((equity - equity.cummax()) / equity.cummax().replace(0, np.nan)) * 100
        ax5b.fill_between(dd.index, dd, 0, color=self.C2, alpha=0.25, label='Drawdown %')
        ax5b.set_ylabel('Drawdown %', color=self.C2, fontsize=7)
        ax5b.tick_params(axis='y', colors=self.C2, labelsize=6)
        ax5b.set_ylim(-60, 5)

        # WFV OOS overlay if available
        if wf_pnl is not None:
            cum_wf = (wf_pnl * CAPITAL).cumsum()
            ax5.plot(cum_wf, color=self.C6, lw=1.0, ls='--',
                     alpha=0.7, label='WFV OOS P&L ($)', zorder=3)

        lines5, labs5   = ax5.get_legend_handles_labels()
        lines5b, labs5b = ax5b.get_legend_handles_labels()
        ax5.legend(lines5 + lines5b, labs5 + labs5b, fontsize=7,
                   facecolor=self.PAN, labelcolor=self.TXT, loc='upper left')

        # ── §6  Per-trade P&L waterfall ───────────────────────
        ax6 = self._ax(fig, gs[4, 0],
                       "§6  Per-Trade P&L  ($)  +  cumulative")
        if trades is not None and len(trades) > 0:
            trade_pnl_dol = trades['pnl'].values * CAPITAL
            colors_bar    = [self.C3 if v >= 0 else self.C2 for v in trade_pnl_dol]
            x_idx         = range(len(trade_pnl_dol))
            ax6.bar(x_idx, trade_pnl_dol, color=colors_bar, alpha=0.75, width=0.8)

            ax6b = ax6.twinx()
            ax6b.plot(x_idx, np.cumsum(trade_pnl_dol), color=self.C7,
                      lw=1.4, label='Cum P&L ($)')
            ax6b.axhline(0, color=self.DIM, lw=0.4)
            ax6b.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f'${x:,.0f}'))
            ax6b.set_ylabel('Cum P&L ($)', color=self.C7, fontsize=7)
            ax6b.tick_params(axis='y', colors=self.C7, labelsize=6)

            ax6.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f'${x:,.0f}'))
            ax6.set_ylabel('Trade P&L ($)', color=self.TXT, fontsize=7)
            ax6.set_xlabel('Trade #', color=self.DIM, fontsize=7)

        # ── §7  Exit reason + direction breakdown ─────────────
        ax7 = self._ax(fig, gs[4, 1],
                       "§7  Exit Reason × Direction  (avg P&L $  |  count)")
        if trades is not None and len(trades) > 0:
            breakdown = (trades.groupby(['exit_reason', 'direction'])['pnl']
                         .agg(['mean', 'count'])
                         .reset_index())
            breakdown['mean_dol'] = breakdown['mean'] * CAPITAL

            reasons   = breakdown['exit_reason'].unique()
            dirs      = ['long', 'short']
            x         = np.arange(len(reasons))
            w         = 0.35

            for i, d in enumerate(dirs):
                sub = breakdown[breakdown['direction'] == d]
                sub = sub.set_index('exit_reason').reindex(reasons).fillna(0)
                clr = self.C3 if d == 'long' else self.C2
                bars = ax7.bar(x + i * w, sub['mean_dol'], w,
                               color=clr, alpha=0.8, label=d)
                for bar, cnt in zip(bars, sub['count']):
                    if cnt > 0:
                        ax7.text(bar.get_x() + bar.get_width() / 2,
                                 bar.get_height() + 5,
                                 f'n={int(cnt)}', ha='center', va='bottom',
                                 color=self.TXT, fontsize=6)

            ax7.set_xticks(x + w / 2)
            ax7.set_xticklabels(reasons, color=self.TXT, fontsize=7, rotation=15)
            ax7.axhline(0, color=self.DIM, lw=0.6)
            ax7.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f'${x:,.0f}'))
            ax7.set_ylabel('Avg P&L ($)', color=self.TXT, fontsize=7)
            ax7.legend(fontsize=7, facecolor=self.PAN, labelcolor=self.TXT)

        # ── §8  Monthly P&L heatmap ───────────────────────────
        ax8 = self._ax(fig, gs[5, :],
                       "§8  Monthly P&L  ($)  —  accumulated profit per calendar month")
        try:
            m    = (pnl * CAPITAL).resample('ME').sum()
            df_m = pd.DataFrame({'Y': m.index.year, 'M': m.index.month, 'P': m.values})
            tbl  = df_m.pivot(index='Y', columns='M', values='P')
            mn   = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
            tbl.columns = [mn[c-1] for c in tbl.columns]

            data = tbl.values.astype(float)
            vmax = np.nanpercentile(np.abs(data[~np.isnan(data)]), 95) if data.size else 1

            im = ax8.imshow(data, cmap='RdYlGn', aspect='auto',
                            vmin=-vmax, vmax=vmax, origin='upper')
            ax8.set_xticks(range(len(tbl.columns)))
            ax8.set_xticklabels(tbl.columns, color=self.TXT, fontsize=7)
            ax8.set_yticks(range(len(tbl.index)))
            ax8.set_yticklabels(tbl.index, color=self.TXT, fontsize=7)

            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    v = data[r, c]
                    if not np.isnan(v):
                        ax8.text(c, r, f'${v:,.0f}', ha='center', va='center',
                                 fontsize=5.5,
                                 color='#111111' if abs(v) > vmax * 0.4 else self.TXT)

            cbar = plt.colorbar(im, ax=ax8, fraction=0.012, pad=0.01)
            cbar.ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
            cbar.ax.tick_params(colors=self.TXT, labelsize=6)

        except Exception as e:
            ax8.text(0.5, 0.5, f'Monthly P&L error: {e}',
                     transform=ax8.transAxes, ha='center', color=self.DIM, fontsize=8)

        plt.savefig('WTI_Brent_v5.png', dpi=150, bbox_inches='tight',
                    facecolor=self.BG)
        plt.show()
        print("  → Chart saved: WTI_Brent_v5.png")

    # ─────────────────────────────────────────────────────
    #  PRINT REPORT
    #  Accepts optional label= for TRAIN / HOLDOUT tagging
    # ─────────────────────────────────────────────────────
    def print_report(self, metrics, stat_d, trades, p, ou_mean, label=""):

        CAPITAL = 200_000
        n       = metrics['n_trades']

        # Statistical power note
        if n < 20:
            stat_note = f"⚠  {n} trades — low power. Sharpe CI is very wide."
        elif n < 50:
            stat_note = f"~  {n} trades — moderate power. Sharpe is directional only."
        else:
            stat_note = f"✓  {n} trades — reasonable power for Sharpe interpretation."

        # Asymmetric param display
        ze_long  = p.get('z_entry_long',  p.get('z_entry', '—'))
        ze_short = p.get('z_entry_short', p.get('z_entry', '—'))
        zx_long  = p.get('z_exit_long',   p.get('z_exit',  '—'))
        zx_short = p.get('z_exit_short',  p.get('z_exit',  '—'))
        zs_long  = p.get('z_stop_long',   p.get('z_stop',  '—'))
        zs_short = p.get('z_stop_short',  p.get('z_stop',  '—'))

        tag = f"  [{label}]" if label else ""

        print(f"""
╔{'═'*66}╗
║  WTI / BRENT  STAT-ARB  v5{tag:<38}║
╠{'═'*66}╣
║  PARAMETERS  (asymmetric)
║    Slow window          : {p['slow_window']:>4}d     Fast EWMA span : {p['fast_span']:>4}
║    Vol window           : {p['vol_window']:>4}d     Vol cap        : {p.get('vol_cap',0):>6.3f}
║    z_entry  long/short  : {ze_long if isinstance(ze_long,str) else f'{ze_long:.3f}':>7} / {ze_short if isinstance(ze_short,str) else f'{ze_short:.3f}':>7}
║    z_exit   long/short  : {zx_long if isinstance(zx_long,str) else f'{zx_long:.3f}':>7} / {zx_short if isinstance(zx_short,str) else f'{zx_short:.3f}':>7}
║    z_stop   long/short  : {zs_long if isinstance(zs_long,str) else f'{zs_long:.3f}':>7} / {zs_short if isinstance(zs_short,str) else f'{zs_short:.3f}':>7}
║    z_add                : {p.get('z_add',0):>6.3f}     Max hold       : {p['max_hold']:>4}d
║    OU equilibrium mean  : {ou_mean:>10.5f}
╠{'═'*66}╣
║  SPREAD DIAGNOSTICS
║    ADF p-value  : {stat_d['adf_p']:>8.4f}  {'✓ stationary' if stat_d['adf_p']<0.05 else '✗ non-stationary'}
║    Hurst        : {stat_d['hurst']:>8.4f}  {'mean-rev ✓' if stat_d['hurst']<0.5 else 'random walk ✗'}
║    OU half-life : {stat_d['half_life']:>7.1f}d
║    Johansen     : {'✓ pass' if stat_d['joh_pass'] else '✗ fail':>8}  (trace={stat_d['joh_trace']:.2f}  crit={stat_d['joh_crit']:.2f})
╠{'═'*66}╣
║  PERFORMANCE  (with transaction costs, ${CAPITAL:,} capital)
║    Sharpe ratio   : {metrics['sharpe']:>8.2f}
║    Sortino ratio  : {metrics['sortino']:>8.2f}
║    Calmar ratio   : {metrics['calmar']:>8.2f}
║    Ann. return    : {metrics['ann_ret']*100:>7.2f}%   (${metrics['ann_ret']*CAPITAL:>10,.0f} / yr)
║    Ann. vol       : {metrics['ann_vol']*100:>7.2f}%   (${metrics['ann_vol']*CAPITAL:>10,.0f} / yr)
║    Max drawdown   : {metrics['max_dd']*100:>7.2f}%   (${metrics['max_dd']*CAPITAL:>10,.0f})
╠{'═'*66}╣
║  TRADE STATS
║    Total trades   : {n:>8}
║    Win rate       : {metrics['win_rate']*100:>7.1f}%
║    Avg win        : ${metrics['avg_win']*CAPITAL:>10,.2f}   ({metrics['avg_win']*100:.4f}%)
║    Avg loss       : ${metrics['avg_loss']*CAPITAL:>10,.2f}   ({metrics['avg_loss']*100:.4f}%)
║    Profit factor  : {metrics['profit_factor']:>8.2f}
║    Avg hold       : {metrics['avg_hold']:>7.1f}d
║    % pyramided    : {metrics['pct_pyramided']:>7.1f}%
╠{'═'*66}╣
║  STATISTICAL POWER
║    {stat_note:<64}║
╚{'═'*66}╝""")

        # Trade log
        if trades is not None and len(trades) > 0:
            print(f"\n  LAST 20 TRADES  (P&L in $, capital=${CAPITAL:,})")
            print(f"  {'Entry':>12}  {'Exit':>12}  {'Dir':>6}  {'Hold':>5}  "
                  f"{'EntryZ':>7}  {'ExitZ':>7}  {'P&L ($)':>10}  {'Reason'}")
            print("  " + "─" * 90)
            for _, t in trades.tail(20).iterrows():
                ed  = t['entry_date'].date() if hasattr(t['entry_date'], 'date') else t['entry_date']
                xd  = t['exit_date'].date()  if hasattr(t['exit_date'],  'date') else t['exit_date']
                pnl_dol = t['pnl'] * CAPITAL
                print(f"  {str(ed):>12}  {str(xd):>12}  {t['direction']:>6}  "
                      f"{t['hold_days']:>5}  {t['entry_z']:>7.3f}  {t['exit_z']:>7.3f}  "
                      f"${pnl_dol:>9,.2f}  {t['exit_reason']}")

            # Exit breakdown in $
            print(f"\n  EXIT REASON BREAKDOWN  ($ terms)")
            by_r = trades.groupby('exit_reason')['pnl'].agg(['count','sum','mean'])
            for r, row in by_r.iterrows():
                wr = (trades.loc[trades['exit_reason']==r, 'pnl'] > 0).mean() * 100
                print(f"    {r:<16} n={int(row['count']):>4}  "
                      f"total=${row['sum']*CAPITAL:>10,.2f}  "
                      f"avg=${row['mean']*CAPITAL:>8,.2f}  "
                      f"wr={wr:.0f}%")

            # Direction breakdown
            print(f"\n  DIRECTION BREAKDOWN  ($ terms)")
            by_d = trades.groupby('direction')['pnl'].agg(['count','sum','mean'])
            for d, row in by_d.iterrows():
                wr = (trades.loc[trades['direction']==d, 'pnl'] > 0).mean() * 100
                print(f"    {d:<8} n={int(row['count']):>4}  "
                      f"total=${row['sum']*CAPITAL:>10,.2f}  "
                      f"avg=${row['mean']*CAPITAL:>8,.2f}  "
                      f"wr={wr:.0f}%")


