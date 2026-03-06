import numpy as np
import pandas as pd


class backtest():

    def __init__(self, trade_capital=200_000, leverage=1):
        self.T1 = "BZ=F"
        self.T2 = "CL=F"

        # ── 2026 MCX Cost Model ──
        self.COMMISSION = 0.0004
        self.STT = 0.0005
        self.SLIPPAGE = 0.0004
        # COST_RT is the full Round Trip (Entry + Exit)
        self.COST_RT = self.COMMISSION + self.STT + (self.SLIPPAGE * 2)

        self.TRADE_CAPITAL = trade_capital
        self.LEVERAGE = leverage
        self._notional = self.TRADE_CAPITAL * self.LEVERAGE

    def backtest(self, feat, sig, p, cost=True):
        lr = feat['lr']
        z = sig['z_slow']

        # ── Params ──
        ze_long = p.get('z_entry_long', p.get('z_entry', 1.25))
        ze_short = p.get('z_entry_short', p.get('z_entry', 1.25))
        zx_long = p.get('z_exit_long', p.get('z_exit', 0.30))
        zx_short = p.get('z_exit_short', p.get('z_exit', 0.30))
        zs_long = p.get('z_stop_long', p.get('z_stop', 3.00))
        zs_short = p.get('z_stop_short', p.get('z_stop', 3.00))

        notional = self._notional

        # ── State ──
        position = 0
        entry_lr = 0.0
        entry_idx = 0
        entry_z = 0.0
        entry_cost_dol = 0.0

        trades = []
        daily_pnl_dol = []
        daily_pnl_pct = []
        equity_dol = self.TRADE_CAPITAL
        equity_ts = []

        # Half of the round-trip is paid on entry, half on exit
        leg_cost_ratio = self.COST_RT / 2 if cost else 0.0

        for i in range(1, len(lr)):
            date = lr.index[i]
            lr_t = lr.iloc[i]
            lr_p = lr.iloc[i - 1]
            z_t = float(z.iloc[i])

            # 1. Daily MTM move
            lr_move = lr_t - lr_p
            pnl_dol = lr_move * position * notional

            # (Note: We update pnl_dol further down if entry/exit happens)

            hold_days = i - entry_idx

            # ── Exit Logic ──
            if position != 0:
                reason = None

                if position == 1:
                    if sig['exit_stop_long'].iloc[i]:
                        reason = 'stop'
                    elif sig['exit_mean_long'].iloc[i]:
                        reason = 'mean_revert'
                    elif sig['exit_cross'].iloc[i]:
                        reason = 'zero_cross'
                    elif hold_days >= p['max_hold']:
                        reason = 'time_stop'

                elif position == -1:
                    if sig['exit_stop_short'].iloc[i]:
                        reason = 'stop'
                    elif sig['exit_mean_short'].iloc[i]:
                        reason = 'mean_revert'
                    elif sig['exit_cross'].iloc[i]:
                        reason = 'zero_cross'
                    elif hold_days >= p['max_hold']:
                        reason = 'time_stop'

                if reason:
                    # Gross profit from the move itself
                    gross_pnl_dol = (lr_t - entry_lr) * position * notional

                    # Exit leg cost (0.5x of RT)
                    exit_cost_dol = leg_cost_ratio * notional

                    # Apply exit cost to the day's MTM
                    pnl_dol -= exit_cost_dol

                    # Trade P&L = Gross - Entry Leg Cost - Exit Leg Cost
                    trade_pnl_dol = gross_pnl_dol - entry_cost_dol - exit_cost_dol
                    trade_pnl_pct = trade_pnl_dol / self.TRADE_CAPITAL

                    trades.append({
                        'entry_date': lr.index[entry_idx],
                        'exit_date': date,
                        'hold_days': hold_days,
                        'direction': 'long' if position == 1 else 'short',
                        'entry_z': entry_z,
                        'exit_z': z_t,
                        'entry_ze': ze_long if position == 1 else ze_short,
                        'exit_zx': zx_long if position == 1 else zx_short,
                        'stop_zs': zs_long if position == 1 else zs_short,
                        'notional': notional,
                        'leverage': self.LEVERAGE,
                        'pnl': trade_pnl_pct,
                        'pnl_dol': trade_pnl_dol,
                        'exit_reason': reason,
                    })

                    position = 0
                    entry_lr = 0.0
                    entry_cost_dol = 0.0

            # ── Entry Logic ──
            if position == 0:
                is_long = sig['long_entry'].iloc[i]
                is_short = sig['short_entry'].iloc[i]

                if is_long or is_short:
                    position = 1 if is_long else -1
                    entry_idx = i
                    entry_lr = lr_t
                    entry_z = z_t

                    # Entry leg cost (0.5x of RT)
                    entry_cost_dol = leg_cost_ratio * notional

                    # Subtract entry cost from today's P&L
                    pnl_dol -= entry_cost_dol

            # Finalize daily metrics
            daily_pnl_dol.append(pnl_dol)
            daily_pnl_pct.append(pnl_dol / self.TRADE_CAPITAL)
            equity_dol += pnl_dol
            equity_ts.append(equity_dol)

        # ── End-of-Data Force Close ──
        if position != 0:
            gross_pnl_dol = (lr.iloc[-1] - entry_lr) * position * notional
            exit_cost_dol = leg_cost_ratio * notional
            trade_pnl_dol = gross_pnl_dol - entry_cost_dol - exit_cost_dol

            trades.append({
                'entry_date': lr.index[entry_idx],
                'exit_date': lr.index[-1],
                'hold_days': len(lr) - 1 - entry_idx,
                'direction': 'long' if position == 1 else 'short',
                'entry_z': entry_z,
                'exit_z': float(z.iloc[-1]),
                'entry_ze': ze_long if position == 1 else ze_short,
                'exit_zx': zx_long if position == 1 else zx_short,
                'stop_zs': zs_long if position == 1 else zs_short,
                'notional': notional,
                'leverage': self.LEVERAGE,
                'pnl': trade_pnl_dol / self.TRADE_CAPITAL,
                'pnl_dol': trade_pnl_dol,
                'exit_reason': 'end_of_data',
            })

        pnl_s = pd.Series(daily_pnl_pct, index=lr.index[1:])
        equity_s = pd.Series(equity_ts, index=lr.index[1:])
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=['entry_date', 'exit_date', 'hold_days', 'direction',
                     'entry_z', 'exit_z', 'entry_ze', 'exit_zx', 'stop_zs',
                     'notional', 'leverage', 'pnl', 'pnl_dol', 'exit_reason'])

        return pnl_s, equity_s, trades_df

