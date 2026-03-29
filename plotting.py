import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker


class plotting():

    def __init__(self):
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
        a.set_title(title, color=self.TXT, fontsize=8, pad=5,
                    loc='left', fontweight='bold')
        a.tick_params(colors=self.DIM, labelsize=7)
        for sp in a.spines.values():
            sp.set_color(self.GR)
        a.grid(True, color=self.GR, lw=0.4, alpha=0.6)
        return a

    def _shade_trades(self, ax, trades):
        if trades is None or len(trades) == 0:
            return
        for _, t in trades[trades['direction'] == 'long'].iterrows():
            ax.axvspan(t['entry_date'], t['exit_date'],
                       alpha=0.08, color=self.C3, lw=0)
        for _, t in trades[trades['direction'] == 'short'].iterrows():
            ax.axvspan(t['entry_date'], t['exit_date'],
                       alpha=0.08, color=self.C2, lw=0)

    def _hline(self, ax, y, color, lw=1.0, ls='--', label=None, alpha=1.0):
        ax.axhline(y, color=color, lw=lw, linestyle=ls, alpha=alpha,
                   label=label if label else None)

    # ─────────────────────────────────────────────────────
    #  MAIN PLOT
    # ─────────────────────────────────────────────────────
    def plot_all(self, df, feat, sig, pnl, equity, trades, stat_d, metrics,
                 sens_results, p,
                 wf_pnl=None, wf_equity=None, wf_params_df=None,
                 label=""):

        CAPITAL = 200_000
        beta    = float(feat['beta'].iloc[0])

        fig = plt.figure(figsize=(22, 30))
        fig.patch.set_facecolor(self.BG)

        title_label = f"  [{label}]" if label else ""
        fig.suptitle(
            f"HAL.NS / BDL.NS  Stat-Arb  v3{title_label}   |   "
            f"β={beta:.3f}   "
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

        # ── §1  Beta-adjusted spread + mean anchors ────────
        ax1 = self._ax(
            fig, gs[0, :],
            f"§1  β-Adjusted Spread  [log(NG) − {beta:.3f}×log(CL)]"
            f"  +  Slow / Fast / OU anchors")
        ax1.plot(feat['lr'], color=self.C1, lw=0.8, alpha=0.7,
                 label=f'spread (β={beta:.3f})')
        ax1.plot(feat['mu_slow'], color=self.C4, lw=1.6,
                 label=f"Slow {p['slow_window']}d", alpha=0.95)
        ax1.plot(feat['mu_fast'], color=self.C6, lw=1.0, ls='--',
                 label=f"Fast EWMA {p['fast_span']}", alpha=0.8)

        if 'z_ou' in feat.columns and 'vol' in feat.columns:
            ou_series = feat['mu_slow'] + feat['z_ou'] * feat['vol']
            ou_flat   = ou_series.mean()
        else:
            ou_flat = float(feat['lr'].iloc[:504].mean())
        ax1.axhline(ou_flat, color=self.C5, lw=1.0, ls=':', alpha=0.8,
                    label=f"OU equil ≈{ou_flat:.4f}")

        self._shade_trades(ax1, trades)
        ax1.legend(fontsize=7, facecolor=self.PAN, labelcolor=self.TXT,
                   loc='upper left')
        ax1.set_ylabel('β-adjusted log spread', color=self.DIM, fontsize=7)

        # ── §2  Z-score + asymmetric thresholds ───────────
        ax2 = self._ax(fig, gs[1, :],
                       "§2  Z-Slow (primary)  +  Asymmetric entry/exit/stop  +  trade markers")

        z_slow = sig['z_slow'].clip(-5, 5)
        ax2.plot(z_slow, color=self.C5, lw=0.85, alpha=0.9, label='z_slow')
        ax2.axhline(0, color=self.DIM, lw=0.6)

        ze_long  = p.get('z_entry_long',  p.get('z_entry', 1.25))
        ze_short = p.get('z_entry_short', p.get('z_entry', 1.25))
        zx_long  = p.get('z_exit_long',   p.get('z_exit',  0.30))
        zx_short = p.get('z_exit_short',  p.get('z_exit',  0.30))
        zs_long  = p.get('z_stop_long',   p.get('z_stop',  3.00))
        zs_short = p.get('z_stop_short',  p.get('z_stop',  3.00))

        self._hline(ax2, -ze_long,  self.C3, lw=1.3, ls='--',
                    label=f"entry long  −{ze_long:.2f}σ")
        self._hline(ax2,  ze_short, self.C2, lw=1.3, ls='--',
                    label=f"entry short +{ze_short:.2f}σ")
        self._hline(ax2, -zx_long,  self.C3, lw=0.8, ls=':',
                    label=f"exit long   −{zx_long:.2f}σ", alpha=0.7)
        self._hline(ax2,  zx_short, self.C2, lw=0.8, ls=':',
                    label=f"exit short  +{zx_short:.2f}σ", alpha=0.7)
        self._hline(ax2, -zs_long,  self.C2, lw=0.5, ls=':', alpha=0.35)
        self._hline(ax2,  zs_short, self.C2, lw=0.5, ls=':', alpha=0.35)

        # Trade markers — differentiate pyramided with larger marker
        if trades is not None and len(trades) > 0:
            for _, t in trades.iterrows():
                c   = self.C3 if t['direction'] == 'long' else self.C2
                edt = t['entry_date']
                sz  = 45 if t.get('pyramided', False) else 25
                if edt in sig.index:
                    try:
                        yv = float(sig.loc[edt, 'z_slow'])
                        ax2.scatter(edt, np.clip(yv, -5, 5),
                                    color=c, s=sz, zorder=5, alpha=0.9,
                                    marker='^' if t.get('pyramided', False) else 'o')
                    except Exception:
                        pass

        ax2.set_ylim(-5.3, 5.3)
        ax2.legend(fontsize=6.5, facecolor=self.PAN, labelcolor=self.TXT,
                   loc='upper left', ncol=3)
        ax2.set_ylabel('z-score', color=self.DIM, fontsize=7)

        # ── §3  Z-score components ─────────────────────────
        ax3 = self._ax(fig, gs[2, 0],
                       "§3  Z-Score Components  (slow / med / fast / OU)")
        for col, color, lbl in [
            ('z_slow', self.C4, 'z_slow (structural)'),
            ('z_med',  self.C6, 'z_med'),
            ('z_fast', self.C3, 'z_fast (EWMA)'),
            ('z_ou',   self.C5, 'z_ou (OU)'),
        ]:
            if col in feat.columns:
                ax3.plot(feat[col].clip(-4.5, 4.5), lw=0.7, alpha=0.75,
                         color=color, label=lbl)
        ax3.axhline(0, color=self.DIM, lw=0.5)
        ax3.set_ylim(-4.8, 4.8)
        ax3.legend(fontsize=6.5, facecolor=self.PAN, labelcolor=self.TXT)
        ax3.set_ylabel('z-score', color=self.DIM, fontsize=7)

        # ── §4  Regime filters ────────────────────────────
        ax4  = self._ax(fig, gs[2, 1],
                        "§4  Regime Filters  (vol ratio  |  autocorrelation)")
        ax4b = ax4.twinx()

        if 'vol_ratio' in sig.columns:
            ax4.plot(sig['vol_ratio'].clip(0, 5), color=self.C1, lw=0.9,
                     alpha=0.85, label='vol ratio')
            ax4.axhline(p.get('vol_cap', 1.5), color=self.C2, lw=1.1, ls='--',
                        label=f"vol_cap={p.get('vol_cap',1.5):.2f}")
            ax4.fill_between(
                sig.index,
                sig['vol_ratio'].clip(0, 5).clip(lower=p.get('vol_cap', 1.5)),
                p.get('vol_cap', 1.5),
                color=self.C2, alpha=0.15)
            ax4.set_ylim(0, 5)
            ax4.set_ylabel('vol ratio', color=self.C1, fontsize=7)
            ax4.tick_params(axis='y', colors=self.C1, labelsize=6)

        if 'autocorr' in feat.columns:
            ax4b.plot(feat['autocorr'].clip(-1, 1), color=self.C5, lw=0.7,
                      alpha=0.7, label='autocorr')
            ax4b.axhline(p.get('autocorr_threshold', 0.1),
                         color=self.C5, lw=0.8, ls=':', alpha=0.6)
            ax4b.axhline(0, color=self.DIM, lw=0.4)
            ax4b.set_ylim(-1.2, 1.2)
            ax4b.set_ylabel('autocorr', color=self.C5, fontsize=7)
            ax4b.tick_params(axis='y', colors=self.C5, labelsize=6)

        lines1, labs1 = ax4.get_legend_handles_labels()
        lines2, labs2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labs1 + labs2, fontsize=6.5,
                   facecolor=self.PAN, labelcolor=self.TXT)

        # ── §5  Accumulated  P&L ─────────────────────────
        ax5 = self._ax(fig, gs[3, :],
                       "§5  Accumulated P&L  ()  |  drawdown  |  notional cap: 1M")

        cum_pnl_dol = (pnl * CAPITAL).cumsum()
        ax5.plot(cum_pnl_dol, color=self.C7, lw=1.6,
                 label='Cumulative P&L ()', zorder=4)
        ax5.axhline(0, color=self.DIM, lw=0.7, ls='--')
        ax5.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'{x:,.0f}'))
        ax5.set_ylabel('Cumulative P&L ()', color=self.C7, fontsize=7)
        ax5.tick_params(axis='y', colors=self.C7, labelsize=7)

        ax5b = ax5.twinx()
        dd = ((equity - equity.cummax()) /
               equity.cummax().replace(0, np.nan)) * 100
        ax5b.fill_between(dd.index, dd, 0,
                          color=self.C2, alpha=0.25, label='Drawdown %')
        ax5b.set_ylabel('Drawdown %', color=self.C2, fontsize=7)
        ax5b.tick_params(axis='y', colors=self.C2, labelsize=6)
        ax5b.set_ylim(-60, 5)

        if wf_pnl is not None:
            cum_wf = (wf_pnl * CAPITAL).cumsum()
            ax5.plot(cum_wf, color=self.C6, lw=1.0, ls='--',
                     alpha=0.7, label='WFV OOS P&L ()', zorder=3)

        lines5, labs5   = ax5.get_legend_handles_labels()
        lines5b, labs5b = ax5b.get_legend_handles_labels()
        ax5.legend(lines5 + lines5b, labs5 + labs5b, fontsize=7,
                   facecolor=self.PAN, labelcolor=self.TXT, loc='upper left')

        # ── §6  Per-trade P&L waterfall ───────────────────
        ax6 = self._ax(fig, gs[4, 0],
                       "§6  Per-Trade P&L ()  +  cumulative  (▲ = pyramided)")
        if trades is not None and len(trades) > 0:
            trade_pnl_dol = trades['pnl'].values * CAPITAL
            colors_bar    = [self.C3 if v >= 0 else self.C2 for v in trade_pnl_dol]
            x_idx         = range(len(trade_pnl_dol))
            ax6.bar(x_idx, trade_pnl_dol, color=colors_bar, alpha=0.75, width=0.8)

            # Mark pyramided trades with a triangle on top
            if 'pyramided' in trades.columns:
                for xi, (_, t) in enumerate(trades.iterrows()):
                    if t.get('pyramided', False):
                        ypos = t['pnl'] * CAPITAL
                        ax6.scatter(xi, ypos + (200 if ypos >= 0 else -200),
                                    marker='^', color=self.C7, s=30, zorder=6)

            ax6b = ax6.twinx()
            ax6b.plot(x_idx, np.cumsum(trade_pnl_dol),
                      color=self.C7, lw=1.4, label='Cum P&L ()')
            ax6b.axhline(0, color=self.DIM, lw=0.4)
            ax6b.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f'{x:,.0f}'))
            ax6b.set_ylabel('Cum P&L ()', color=self.C7, fontsize=7)
            ax6b.tick_params(axis='y', colors=self.C7, labelsize=6)

            ax6.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f'{x:,.0f}'))
            ax6.set_ylabel('Trade P&L ()', color=self.TXT, fontsize=7)
            ax6.set_xlabel('Trade #', color=self.DIM, fontsize=7)

        # ── §7  Exit reason × direction ───────────────────
        ax7 = self._ax(fig, gs[4, 1],
                       "§7  Exit Reason × Direction  (avg P&L   |  count)")
        if trades is not None and len(trades) > 0:
            breakdown = (trades.groupby(['exit_reason', 'direction'])['pnl']
                         .agg(['mean', 'count']).reset_index())
            breakdown['mean_dol'] = breakdown['mean'] * CAPITAL

            reasons = breakdown['exit_reason'].unique()
            dirs    = ['long', 'short']
            x       = np.arange(len(reasons))
            w       = 0.35

            for i_d, d in enumerate(dirs):
                sub  = breakdown[breakdown['direction'] == d]
                sub  = sub.set_index('exit_reason').reindex(reasons).fillna(0)
                clr  = self.C3 if d == 'long' else self.C2
                bars = ax7.bar(x + i_d * w, sub['mean_dol'], w,
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
                lambda x, _: f'{x:,.0f}'))
            ax7.set_ylabel('Avg P&L ()', color=self.TXT, fontsize=7)
            ax7.legend(fontsize=7, facecolor=self.PAN, labelcolor=self.TXT)

        # ── §8  Monthly P&L heatmap ───────────────────────
        ax8 = self._ax(fig, gs[5, :],
                       "§8  Monthly P&L ()  —  accumulated profit per calendar month")
        try:
            m    = (pnl * CAPITAL).resample('ME').sum()
            df_m = pd.DataFrame({
                'Y': m.index.year, 'M': m.index.month, 'P': m.values})
            tbl  = df_m.pivot(index='Y', columns='M', values='P')
            mn   = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
            tbl.columns = [mn[c-1] for c in tbl.columns]

            data = tbl.values.astype(float)
            vmax = (np.nanpercentile(np.abs(data[~np.isnan(data)]), 95)
                    if data.size else 1)

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
                        ax8.text(c, r, f'{v:,.0f}', ha='center', va='center',
                                 fontsize=5.5,
                                 color='#111111' if abs(v) > vmax * 0.4 else self.TXT)

            cbar = plt.colorbar(im, ax=ax8, fraction=0.012, pad=0.01)
            cbar.ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
            cbar.ax.tick_params(colors=self.TXT, labelsize=6)

        except Exception as e:
            ax8.text(0.5, 0.5, f'Monthly P&L error: {e}',
                     transform=ax8.transAxes, ha='center',
                     color=self.DIM, fontsize=8)

        plt.savefig('NG_CL_statarb_v3.png', dpi=150, bbox_inches='tight',
                    facecolor=self.BG)
        plt.show()
        print("  → Chart saved: NG_CL_statarb_v3.png")

    def print_trade_audit(self, trades, df):
        """
        Audit every trade against the backtest trade log.

        Notionals, beta, leverage and capital are all read from the trade log
        itself — never recomputed from caller arguments — so there is no risk
        of a parameter mismatch inflating or deflating gross P&L in the audit.

        Columns expected in trades DataFrame (all written by backtest()):
            entry_date, exit_date, hold_days, direction, entry_lr, exit_lr,
            entry_z, exit_z, units, pyramided, beta, leverage,
            t1_notional, t2_notional, total_notional,
            entry_cost_dol, exit_cost_dol, gross_pnl_dol, pnl_dol,
            exit_reason
        """
        if trades is None or len(trades) == 0:
            print("No trades to audit.")
            return

        # ── Read authoritative values from first trade row ─────────────────
        first = trades.iloc[0]
        beta = float(first['beta'])
        leverage = float(first['leverage'])
        total_notional = 200_000  # per-trade notional
        t1_notional = total_notional/2
        t2_notional = total_notional/2
        # Infer capital from total_notional and leverage
        capital_inferred = total_notional / leverage

        # ── Identify price columns in df ───────────────────────────────────
        price_cols = [c for c in df.columns
                      if c not in ('Open', 'High', 'Low', 'Volume')]
        t1_col = price_cols[0] if len(price_cols) >= 1 else df.columns[0]
        t2_col = price_cols[1] if len(price_cols) >= 2 else df.columns[1]

        # ── Header ────────────────────────────────────────────────────────
        W = 200
        print(f"\n╔{'═' * W}╗")
        print(f"║  TRADE AUDIT LOG  —  prices from df  +  spread/PnL from trade log")
        print(f"║  β={beta:.4f}  |  capital≈{capital_inferred:,.0f}  |  leverage={leverage:.0f}×")
        print(f"║  t1_notional={t1_notional:,.0f}   t2_notional={t2_notional:,.0f}"
              f"   total_notional={total_notional:,.0f}  (read from trade log)")
        print(f"║")
        print(f"║  Gross_log  = (exit_lr − entry_lr) × sign × |units| × total_notional")
        print(f"║               uses LOGGED spreads — blended entry for pyramided trades ▲")
        print(f"║  Gross_px   = spread derived from actual T1/T2 prices (verification only)")
        print(f"║  Net_calc   = Gross_log − entry_cost − exit_cost  (should equal Net_rec)")
        print(f"║  Sprd_chk   ≈ 0 for non-pyramided; non-zero [blend] expected for ▲")
        print(f"╠{'═' * W}╣")

        col_hdr = (
            f"  {'#':>3}  {'Dir':>6}  {'Entry':>12}  {'Exit':>12}  {'Hold':>4}  "
            f"{'T1_in':>10}  {'T1_out':>10}  "
            f"{'T2_in':>10}  {'T2_out':>10}  "
            f"{'Sprd_px_in':>10}  {'Sprd_px_out':>11}  "
            f"{'ΔSprd_pos':>10}  {'Units':>5}  "
            f"{'Gross_log':>11}  {'Costs':>8}  "
            f"{'Net_calc':>11}  {'Net_rec':>11}  {'Δ':>9}  {'OK':>4}  Reason"
        )
        print(col_hdr)
        print("  " + "─" * W)

        mismatches = []
        running_pnl = 0.0

        for idx, (_, t) in enumerate(trades.iterrows()):
            ed = t['entry_date']
            xd = t['exit_date']
            direction = t['direction']
            sign = 1 if direction == 'long' else -1
            n_units = abs(int(t['units']))
            pyramided = bool(t.get('pyramided', False))

            # Per-trade notional (handles future variable-sizing cleanly)
            tn = float(total_notional)

            # ── Actual prices at entry / exit ──────────────────────────────
            def _price(col, date):
                try:
                    return float(df.loc[date, col])
                except KeyError:
                    try:
                        return float(df[col].asof(date))
                    except Exception:
                        return float('nan')

            t1_in = _price(t1_col, ed)
            t2_in = _price(t2_col, ed)
            t1_out = _price(t1_col, xd)
            t2_out = _price(t2_col, xd)

            # ── Price-derived spreads (verification) ───────────────────────
            sprd_px_in = np.log(t1_in) - beta * np.log(t2_in)
            sprd_px_out = np.log(t1_out) - beta * np.log(t2_out)

            # ── Log spreads (authoritative for PnL) ───────────────────────
            sprd_log_in = float(t['entry_lr'])
            sprd_log_out = float(t['exit_lr'])

            # Sanity: price-derived vs logged entry spread
            # Non-pyramided → should be ~0.  Pyramided → will differ [blend].
            sprd_chk = sprd_px_in - sprd_log_in

            # ── P&L calculations ───────────────────────────────────────────
            delta_sprd_pos = (sprd_log_out - sprd_log_in) * sign  # positive = profit

            gross_log = delta_sprd_pos * n_units * tn
            gross_px = (sprd_px_out - sprd_px_in) * sign * n_units * tn  # sanity

            entry_cost = float(t['entry_cost_dol'])
            exit_cost = float(t['exit_cost_dol'])
            costs = entry_cost + exit_cost

            net_calc = gross_log - costs
            net_rec = float(t['pnl_dol'])
            diff = net_calc - net_rec

            # Also cross-check against the gross stored in the trade log
            gross_rec = float(t['gross_pnl_dol'])
            gross_diff = gross_log - gross_rec  # should be ~0

            ok = abs(diff) < 1.0 and abs(gross_diff) < 1.0
            match_str = '  ✓' if ok else ' ✗✗'
            if not ok:
                mismatches.append(idx + 1)

            running_pnl += net_rec
            pyr_flag = '▲' if pyramided else ' '

            print(
                f"  {idx + 1:>3}  {direction:>5}{pyr_flag}"
                f"  {str(ed.date() if hasattr(ed, 'date') else ed):>12}"
                f"  {str(xd.date() if hasattr(xd, 'date') else xd):>12}"
                f"  {int(t['hold_days']):>4}"
                f"  {t1_in:>10.3f}  {t1_out:>10.3f}"
                f"  {t2_in:>10.3f}  {t2_out:>10.3f}"
                f"  {sprd_px_in:>10.5f}  {sprd_px_out:>11.5f}"
                f"  {delta_sprd_pos:>10.5f}"
                f"  {n_units:>5}"
                f"  {gross_log:>11,.2f}"
                f"  {costs:>8,.2f}"
                f"  {net_calc:>11,.2f}  {net_rec:>11,.2f}"
                f"  {diff:>9.2f}"
                f"  {match_str}"
                f"  {t['exit_reason']}"
            )

        # ── Running total ──────────────────────────────────────────────────
        print("  " + "─" * W)
        print(f"  {'':>3}  {'TOTAL':>6}  {'':>12}  {'':>12}  {'':>4}  "
              f"{'':>10}  {'':>10}  {'':>10}  {'':>10}  "
              f"{'':>10}  {'':>11}  {'':>11}  {'':>9}  {'':>10}  {'':>5}  "
              f"{'':>11}  {'':>10}  {'':>8}  "
              f"{'':>11}  {running_pnl:>11,.2f}")

        # ── Direction summary ──────────────────────────────────────────────
        print(f"\n  {'─' * 90}")
        print(f"  DIRECTION SUMMARY  (pnl_dol from trade log)")
        print(f"  {'Dir':>6}  {'n':>4}  {'WR%':>6}  {'AvgPnL':>11}  "
              f"{'TotalPnL':>12}  {'AvgHold':>8}  {'Payoff':>7}  "
              f"{'AvgEntry|z|':>12}  {'AvgExit|z|':>11}")
        print(f"  {'─' * 90}")

        for d in ('long', 'short'):
            sub = trades[trades['direction'] == d]
            if len(sub) == 0:
                continue
            wr = (sub['pnl_dol'] > 0).mean()
            avg_pnl = sub['pnl_dol'].mean()
            tot_pnl = sub['pnl_dol'].sum()
            avg_hld = sub['hold_days'].mean()
            avg_ez = sub['entry_z'].abs().mean()
            avg_xz = sub['exit_z'].abs().mean()
            wins = sub.loc[sub['pnl_dol'] > 0, 'pnl_dol'].mean() \
                if (sub['pnl_dol'] > 0).any() else 0.0
            losses = sub.loc[sub['pnl_dol'] < 0, 'pnl_dol'].abs().mean() \
                if (sub['pnl_dol'] < 0).any() else 0.0
            payoff = wins / losses if losses > 0 else float('inf')
            print(f"  {d:>6}  {len(sub):>4}  {wr * 100:>5.1f}%"
                  f"  {avg_pnl:>11,.2f}  {tot_pnl:>12,.2f}"
                  f"  {avg_hld:>7.1f}d  {payoff:>7.2f}"
                  f"  {avg_ez:>12.3f}  {avg_xz:>11.3f}")

        # ── Exit reason breakdown ──────────────────────────────────────────
        print(f"\n  EXIT REASON BREAKDOWN")
        print(f"  {'Reason':>15}  {'n':>4}  {'WR%':>6}  {'AvgPnL':>11}  {'TotalPnL':>12}")
        print(f"  {'─' * 55}")
        for reason, grp in trades.groupby('exit_reason'):
            wr = (grp['pnl_dol'] > 0).mean()
            avg_pnl = grp['pnl_dol'].mean()
            tot_pnl = grp['pnl_dol'].sum()
            print(f"  {reason:>15}  {len(grp):>4}  {wr * 100:>5.1f}%"
                  f"  {avg_pnl:>11,.2f}  {tot_pnl:>12,.2f}")

        # ── Cost summary ───────────────────────────────────────────────────
        total_gross = trades['gross_pnl_dol'].sum()
        total_costs = (trades['entry_cost_dol'] + trades['exit_cost_dol']).sum()
        total_net = trades['pnl_dol'].sum()
        cost_drag_pct = (total_costs / abs(total_gross) * 100) if total_gross != 0 else float('nan')

        print(f"\n  COST SUMMARY")
        print(f"  {'':>4}  Gross P&L :  {total_gross:>12,.2f}")
        print(f"  {'':>4}  Total Costs: {total_costs:>12,.2f}  ({cost_drag_pct:.1f}% of |gross|)")
        print(f"  {'':>4}  Net P&L   :  {total_net:>12,.2f}")

        # ── Mismatch report ────────────────────────────────────────────────
        if mismatches:
            print(f"\n  ⚠  {len(mismatches)} mismatch(es) in trade(s): {mismatches}")
            print(f"     net_calc ≠ net_rec by ≥ 1.00 — check entry_cost_dol /")
            print(f"     exit_cost_dol accumulation for pyramided trades.")
        else:
            print(f"\n  ✓  All {len(trades)} trades reconcile within 1.00")

        print(f"╚{'═' * W}╝\n")

    # ─────────────────────────────────────────────────────
    #  PRINT REPORT  (updated to show beta)
    # ─────────────────────────────────────────────────────
    def print_report(self, metrics, stat_d, trades, p, ou_mean,
                     beta=None, label=""):

        CAPITAL = 200_000
        n       = metrics['n_trades']

        if n < 20:
            stat_note = f"⚠  {n} trades — low power. Sharpe CI is very wide."
        elif n < 50:
            stat_note = f"~  {n} trades — moderate power. Sharpe is directional only."
        else:
            stat_note = f"✓  {n} trades — reasonable power for Sharpe interpretation."

        ze_long  = p.get('z_entry_long',  p.get('z_entry', '—'))
        ze_short = p.get('z_entry_short', p.get('z_entry', '—'))
        zx_long  = p.get('z_exit_long',   p.get('z_exit',  '—'))
        zx_short = p.get('z_exit_short',  p.get('z_exit',  '—'))
        zs_long  = p.get('z_stop_long',   p.get('z_stop',  '—'))
        zs_short = p.get('z_stop_short',  p.get('z_stop',  '—'))

        beta_val  = beta if beta is not None else stat_d.get('beta', float('nan'))
        tag       = f"  [{label}]" if label else ""

        print(f"""
╔{'═'*66}╗
║  HAL.NS / BDL.NS  STAT-ARB  v3{tag:<38}║
╠{'═'*66}╣
║  STRUCTURAL
║    Hedge ratio β        : {beta_val:.4f}  (log(NG) − β×log(CL))
║    Notional cap         : 200,000  (hard ceiling per trade)
║    Leverage             : {p.get('leverage', '—')}×
╠{'═'*66}╣
║  PARAMETERS  (asymmetric)
║    Slow window          : {p['slow_window']:>4}d     Fast EWMA span : {p['fast_span']:>4}
║    Vol window           : {p['vol_window']:>4}d     Vol cap        : {p.get('vol_cap',0):>6.3f}
║    z_entry  long/short  : {ze_long if isinstance(ze_long,str) else f'{ze_long:.3f}':>7} / {ze_short if isinstance(ze_short,str) else f'{ze_short:.3f}':>7}
║    z_exit   long/short  : {zx_long if isinstance(zx_long,str) else f'{zx_long:.3f}':>7} / {zx_short if isinstance(zx_short,str) else f'{zx_short:.3f}':>7}
║    z_stop   long/short  : {zs_long if isinstance(zs_long,str) else f'{zs_long:.3f}':>7} / {zs_short if isinstance(zs_short,str) else f'{zs_short:.3f}':>7}
║    z_add                : {p.get('z_add',0):>6.3f}     Max hold       : {p['max_hold']:>4}d
║    OU equilibrium mean  : {ou_mean:>10.5f}  (β-adjusted)
╠{'═'*66}╣
║  SPREAD DIAGNOSTICS
║    ADF p-value  : {stat_d['adf_p']:>8.4f}  {'✓ stationary' if stat_d['adf_p']<0.05 else '✗ non-stationary'}
║    Hurst        : {stat_d['hurst']:>8.4f}  {'mean-rev ✓' if stat_d['hurst']<0.5 else 'random walk ✗'}
║    OU half-life : {stat_d['half_life']:>7.1f}d
║    Johansen     : {'✓ pass' if stat_d['joh_pass'] else '✗ fail':>8}  (trace={stat_d['joh_trace']:.2f}  crit={stat_d['joh_crit']:.2f})
╠{'═'*66}╣
║  PERFORMANCE  (with transaction costs, {CAPITAL:,} capital)
║    Sharpe ratio   : {metrics['sharpe']:>8.2f}
║    Sortino ratio  : {metrics['sortino']:>8.2f}
║    Calmar ratio   : {metrics['calmar']:>8.2f}
║    Ann. return    : {metrics['ann_ret']*100:>7.2f}%   ({metrics['ann_ret']*CAPITAL:>10,.0f} / yr)
║    Ann. vol       : {metrics['ann_vol']*100:>7.2f}%   ({metrics['ann_vol']*CAPITAL:>10,.0f} / yr)
║    Max drawdown   : {metrics['max_dd']*100:>7.2f}%   ({metrics['max_dd']*CAPITAL:>10,.0f})
╠{'═'*66}╣
║  TRADE STATS
║    Total trades   : {n:>8}
║    Win rate       : {metrics['win_rate']*100:>7.1f}%
║    Avg win        : {metrics['avg_win']*CAPITAL:>10,.2f}
║    Avg loss       : {metrics['avg_loss']*CAPITAL:>10,.2f}
║    Profit factor  : {metrics['profit_factor']:>8.2f}
║    Avg hold       : {metrics['avg_hold']:>7.1f}d
║    % pyramided    : {metrics['pct_pyramided']:>7.1f}%
╠{'═'*66}╣
║  STATISTICAL POWER
║    {stat_note:<64}║
╚{'═'*66}╝""")

        if trades is not None and len(trades) > 0:
            print(f"\n  LAST 20 TRADES  (P&L in , capital={CAPITAL:,})")
            print(f"  {'Entry':>12}  {'Exit':>12}  {'Dir':>6}  {'Hold':>5}  "
                  f"{'EntryZ':>7}  {'ExitZ':>7}  {'Units':>5}  "
                  f"{'P&L ()':>10}  {'Reason'}")
            print("  " + "─" * 100)
            for _, t in trades.tail(20).iterrows():
                ed      = (t['entry_date'].date()
                           if hasattr(t['entry_date'], 'date') else t['entry_date'])
                xd      = (t['exit_date'].date()
                           if hasattr(t['exit_date'],  'date') else t['exit_date'])
                pnl_dol = t['pnl'] * CAPITAL
                units   = int(t.get('units', 1))
                pyr     = '▲' if t.get('pyramided', False) else ' '
                print(f"  {str(ed):>12}  {str(xd):>12}  {t['direction']:>6}  "
                      f"{t['hold_days']:>5}  {t['entry_z']:>7.3f}  {t['exit_z']:>7.3f}  "
                      f"{units:>4}{pyr}  {pnl_dol:>9,.2f}  {t['exit_reason']}")

            print(f"\n  EXIT REASON BREAKDOWN")
            by_r = trades.groupby('exit_reason')['pnl'].agg(['count','sum','mean'])
            for r, row in by_r.iterrows():
                wr = (trades.loc[trades['exit_reason']==r, 'pnl'] > 0).mean() * 100
                print(f"    {r:<16} n={int(row['count']):>4}  "
                      f"total={row['sum']*CAPITAL:>10,.2f}  "
                      f"avg={row['mean']*CAPITAL:>8,.2f}  wr={wr:.0f}%")

            print(f"\n  DIRECTION BREAKDOWN")
            by_d = trades.groupby('direction')['pnl'].agg(['count','sum','mean'])
            for d, row in by_d.iterrows():
                wr = (trades.loc[trades['direction']==d, 'pnl'] > 0).mean() * 100
                print(f"    {d:<8} n={int(row['count']):>4}  "
                      f"total={row['sum']*CAPITAL:>10,.2f}  "
                      f"avg={row['mean']*CAPITAL:>8,.2f}  wr={wr:.0f}%")
