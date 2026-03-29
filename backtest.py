import numpy as np
import pandas as pd
from params import PARAMS


class backtest():

    def __init__(self, trade_capital=200_000, leverage=1):
        self.T1 = PARAMS['T1']
        self.T2 = PARAMS["T2"]

        # ── 2026 MCX Cost Model ──
        self.COMMISSION = 0.0004
        self.STT        = 0.0005
        self.SLIPPAGE   = 0.0008
        # Full round-trip cost (entry leg + exit leg)
        self.COST_RT    = self.COMMISSION + self.STT + (self.SLIPPAGE * 2)

        self.TRADE_CAPITAL = trade_capital
        self.LEVERAGE      = leverage

    def _get_notional(self):
        return self.TRADE_CAPITAL * self.LEVERAGE

    def backtest(self, feat, sig, p, cost=True, pyramid=False):
        """
        Parameters
        ----------
        feat    : features DataFrame from build_features()
        sig     : signals DataFrame from generate_signals()
        p       : PARAMS dict
        cost    : bool — apply transaction costs (default True)
        pyramid : bool — allow pyramiding into positions (default False)
                  When False, max position size is ±1 unit at all times.
                  When True, allows adding a second unit via long_add /
                  short_add signals, up to ±2 units total.
        """
        lr   = feat['lr']
        z    = sig['z_slow']

        beta = float(feat['beta'].iloc[0])

        # ── Params ────────────────────────────────────────
        ze_long  = p.get('z_entry_long',  p.get('z_entry', 1.25))
        ze_short = p.get('z_entry_short', p.get('z_entry', 1.25))
        zx_long  = p.get('z_exit_long',   p.get('z_exit',  0.30))
        zx_short = p.get('z_exit_short',  p.get('z_exit',  0.30))
        zs_long  = p.get('z_stop_long',   p.get('z_stop',  3.00))
        zs_short = p.get('z_stop_short',  p.get('z_stop',  3.00))

        notional       = self._get_notional()
        leg_cost_ratio = self.COST_RT / 2 if cost else 0.0

        # ── State ─────────────────────────────────────────
        position       = 0
        units          = 0
        entry_lr       = 0.0
        entry_idx      = 0
        entry_z        = 0.0
        entry_cost_dol = 0.0
        pyramided      = False

        trades        = []
        daily_pnl_dol = []
        daily_pnl_pct = []
        equity_dol    = self.TRADE_CAPITAL
        equity_ts     = []

        for i in range(1, len(lr)):
            date  = lr.index[i]
            lr_t  = float(lr.iloc[i])
            lr_p  = float(lr.iloc[i - 1])
            z_t   = float(z.iloc[i])

            # ── Daily MTM P&L ──────────────────────────────
            lr_move = lr_t - lr_p
            pnl_dol = lr_move * units * notional

            hold_days = i - entry_idx

            # ── Exit Logic ────────────────────────────────
            if units != 0:
                reason    = None
                direction = 'long' if units > 0 else 'short'

                if direction == 'long':
                    if sig['exit_stop_long'].iloc[i]:
                        reason = 'stop'
                    elif sig['exit_mean_long'].iloc[i]:
                        reason = 'mean_revert'
                    elif sig['exit_cross'].iloc[i]:
                        reason = 'zero_cross'
                    elif hold_days >= p['max_hold']:
                        reason = 'time_stop'

                elif direction == 'short':
                    if sig['exit_stop_short'].iloc[i]:
                        reason = 'stop'
                    elif sig['exit_mean_short'].iloc[i]:
                        reason = 'mean_revert'
                    elif sig['exit_cross'].iloc[i]:
                        reason = 'zero_cross'
                    elif hold_days >= p['max_hold']:
                        reason = 'time_stop'

                if reason:
                    gross_pnl_dol = (lr_t - entry_lr) * units * notional
                    exit_cost_dol = leg_cost_ratio * notional * abs(units)
                    pnl_dol      -= exit_cost_dol

                    trade_pnl_dol = gross_pnl_dol - entry_cost_dol - exit_cost_dol
                    trade_pnl_pct = trade_pnl_dol / self.TRADE_CAPITAL

                    trades.append({
                        'entry_date':  lr.index[entry_idx],
                        'exit_date':   date,
                        'hold_days':   hold_days,
                        'direction':   direction,
                        'entry_z':     entry_z,
                        'exit_z':      z_t,
                        'entry_ze':    ze_long if direction == 'long' else ze_short,
                        'exit_zx':     zx_long if direction == 'long' else zx_short,
                        'stop_zs':     zs_long if direction == 'long' else zs_short,
                        'notional':    notional,
                        'leverage':    self.LEVERAGE,
                        'beta':        beta,
                        'units':       units,
                        'pyramided':   pyramided,
                        'pnl':         trade_pnl_dol / self.TRADE_CAPITAL,
                        'pnl_dol':     trade_pnl_dol,
                        'exit_reason': reason,
                    })

                    units          = 0
                    position       = 0
                    entry_lr       = 0.0
                    entry_cost_dol = 0.0
                    pyramided      = False

            # ── Entry Logic ───────────────────────────────
            if units == 0:
                is_long  = bool(sig['long_entry'].iloc[i])
                is_short = bool(sig['short_entry'].iloc[i])

                if is_long or is_short:
                    sign           = 1 if is_long else -1
                    units          = sign * 1
                    position       = sign
                    entry_idx      = i
                    entry_lr       = lr_t
                    entry_z        = z_t
                    pyramided      = False
                    entry_cost_dol = leg_cost_ratio * notional
                    pnl_dol       -= entry_cost_dol

            # ── Pyramid Logic ─────────────────────────────
            # Only executes when pyramid=True is explicitly passed.
            # Default is False — position stays at ±1 unit always.
            elif abs(units) == 1 and not pyramided and pyramid:
                do_add = False
                if units > 0 and bool(sig['long_add'].iloc[i]):
                    do_add = True
                elif units < 0 and bool(sig['short_add'].iloc[i]):
                    do_add = True

                if do_add:
                    add_cost        = leg_cost_ratio * notional
                    pnl_dol        -= add_cost
                    entry_cost_dol += add_cost
                    entry_lr        = (entry_lr + lr_t) / 2.0
                    units          *= 2
                    pyramided       = True

            # ── Daily bookkeeping ─────────────────────────
            daily_pnl_dol.append(pnl_dol)
            daily_pnl_pct.append(pnl_dol / self.TRADE_CAPITAL)
            equity_dol += pnl_dol
            equity_ts.append(equity_dol)

        # ── End-of-Data Force Close ───────────────────────
        if units != 0:
            direction     = 'long' if units > 0 else 'short'
            gross_pnl_dol = (float(lr.iloc[-1]) - entry_lr) * units * notional
            exit_cost_dol = leg_cost_ratio * notional * abs(units)
            trade_pnl_dol = gross_pnl_dol - entry_cost_dol - exit_cost_dol

            trades.append({
                'entry_date':  lr.index[entry_idx],
                'exit_date':   lr.index[-1],
                'hold_days':   len(lr) - 1 - entry_idx,
                'direction':   direction,
                'entry_z':     entry_z,
                'exit_z':      float(z.iloc[-1]),
                'entry_ze':    ze_long if direction == 'long' else ze_short,
                'exit_zx':     zx_long if direction == 'long' else zx_short,
                'stop_zs':     zs_long if direction == 'long' else zs_short,
                'notional':    notional,
                'leverage':    self.LEVERAGE,
                'beta':        beta,
                'units':       units,
                'pyramided':   pyramided,
                'pnl':         trade_pnl_dol / self.TRADE_CAPITAL,
                'pnl_dol':     trade_pnl_dol,
                'exit_reason': 'end_of_data',
            })

        pnl_s     = pd.Series(daily_pnl_pct, index=lr.index[1:])
        equity_s  = pd.Series(equity_ts,     index=lr.index[1:])
        trades_df = (pd.DataFrame(trades) if trades
                     else pd.DataFrame(columns=[
                         'entry_date', 'exit_date', 'hold_days', 'direction',
                         'entry_z', 'exit_z', 'entry_ze', 'exit_zx', 'stop_zs',
                         'notional', 'leverage', 'beta', 'units', 'pyramided',
                         'pnl', 'pnl_dol', 'exit_reason']))

        return pnl_s, equity_s, trades_df