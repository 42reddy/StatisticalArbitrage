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

    def backtest(self, feat, sig, p, cost=True):
        """
        Parameters
        ----------
        feat    : features DataFrame from build_features()
        sig     : signals DataFrame from generate_signals()
        p       : PARAMS dict
        cost    : bool — apply transaction costs (default True)
        """
        lr   = feat['lr']
        z    = sig['z_slow']
        mu_slow  = feat['mu_slow']
        vol_slow = feat['vol']

        # ── Diagnostic columns snapshotted at entry ────────
        # Not used for entry/exit logic — recorded purely so
        # scratch2.py can later measure which of these actually
        # discriminate winning from losing trades.
        DIAG_COLS = [
            'z_fast', 'z_fast_slope', 'z_fast_accel', 'z_fast_accel_z',
            'rsi_lr', 'rsi_slope', 'lr_mom', 'lr_mom_slope', 'lr_mom_slope_z',
            'vol_z', 'vol_ratio', 'atr', 'autocorr', 'agreement', 'regime_score',
        ]
        diag_cols = [c for c in DIAG_COLS if c in feat.columns]

        beta_series = feat['beta']   # scalar (static) or time-varying Series

        # ── Params ────────────────────────────────────────
        ze_long  = p.get('z_entry_long',  p.get('z_entry', 1.25))
        ze_short = p.get('z_entry_short', p.get('z_entry', 1.25))
        zx_long  = p.get('z_exit_long',   p.get('z_exit',  0.30))
        zx_short = p.get('z_exit_short',  p.get('z_exit',  0.30))
        zs_long  = p.get('z_stop_long',   p.get('z_stop',  3.00))
        zs_short = p.get('z_stop_short',  p.get('z_stop',  3.00))

        notional       = self._get_notional()
        leg_cost_ratio = self.COST_RT / 2 if cost else 0.0

        # ── Threshold-fill helper ───────────────────────────
        # We only have daily close data, so we almost never observe the
        # exact instant the spread crosses a laid threshold (entry/exit/
        # stop/zero-cross). To mimic real-time execution rather than
        # silently filling at the close (which overstates the move
        # already captured by the time we'd have acted), assume the fill
        # happened exactly at the pre-laid z-level, converted back to a
        # spread price using the same bar's mu_slow/vol_slow.
        def level_to_lr(i, z_level):
            return float(mu_slow.iloc[i] + z_level * vol_slow.iloc[i])

        # ── State ─────────────────────────────────────────
        position       = 0
        units          = 0
        entry_lr       = 0.0
        entry_idx      = 0
        entry_z        = 0.0
        entry_beta     = 0.0
        entry_cost_dol = 0.0
        entry_diag     = {}

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
            exited_this_bar = False

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
                    # Fill at the pre-laid threshold, not the close —
                    # time_stop has no price level, so it fills at the
                    # close (the only price we actually have for it).
                    if reason == 'stop':
                        z_level = -zs_long if direction == 'long' else zs_short
                    elif reason == 'mean_revert':
                        z_level = -zx_long if direction == 'long' else zx_short
                    elif reason == 'zero_cross':
                        z_level = 0.0
                    else:  # time_stop
                        z_level = None

                    exit_lr = level_to_lr(i, z_level) if z_level is not None else lr_t

                    # Correct today's MTM: hold from yesterday's close to
                    # the fill price, not all the way to today's close.
                    pnl_dol      += (exit_lr - lr_t) * units * notional

                    gross_pnl_dol = (exit_lr - entry_lr) * units * notional
                    exit_cost_dol = leg_cost_ratio * notional * abs(units)
                    pnl_dol      -= exit_cost_dol

                    trade_pnl_dol = gross_pnl_dol - entry_cost_dol - exit_cost_dol
                    trade_pnl_pct = trade_pnl_dol / self.TRADE_CAPITAL

                    trades.append({
                        'entry_date':     lr.index[entry_idx],
                        'exit_date':      date,
                        'hold_days':      hold_days,
                        'direction':      direction,
                        'entry_z':        entry_z,
                        'exit_z':         z_t,
                        'entry_ze':       ze_long if direction == 'long' else ze_short,
                        'exit_zx':        zx_long if direction == 'long' else zx_short,
                        'stop_zs':        zs_long if direction == 'long' else zs_short,
                        'notional':       notional,
                        'leverage':       self.LEVERAGE,
                        'beta':           entry_beta,
                        'units':          units,
                        'entry_lr':       entry_lr,
                        'exit_lr':        exit_lr,
                        'entry_cost_dol': entry_cost_dol,
                        'exit_cost_dol':  exit_cost_dol,
                        'gross_pnl_dol':  gross_pnl_dol,
                        'pnl':            trade_pnl_dol / self.TRADE_CAPITAL,
                        'pnl_dol':        trade_pnl_dol,
                        'exit_reason':    reason,
                        **entry_diag,
                    })

                    units          = 0
                    position       = 0
                    entry_lr       = 0.0
                    entry_cost_dol = 0.0
                    entry_diag     = {}
                    exited_this_bar = True

            # ── Entry Logic ───────────────────────────────
            # Skip if we just exited on this very bar — a cross_long/
            # cross_short can be mathematically true on the same bar a
            # stop fires (the old position's threshold bounced near the
            # entry level, then the same down-leg blew through the stop),
            # but that's the SAME move, not a fresh signal. Real-time
            # we'd only ever see this on the next bar's close anyway.
            if units == 0 and not exited_this_bar:
                is_long  = bool(sig['long_entry'].iloc[i])
                is_short = bool(sig['short_entry'].iloc[i])

                if is_long or is_short:
                    sign           = 1 if is_long else -1
                    z_level        = -ze_long if is_long else ze_short
                    fill_lr        = level_to_lr(i, z_level)

                    units          = sign * 1
                    position       = sign
                    entry_idx      = i
                    entry_lr       = fill_lr
                    entry_z        = z_t
                    entry_beta     = float(beta_series.iloc[i])
                    entry_cost_dol = leg_cost_ratio * notional
                    # Position is filled at the threshold, then held to
                    # today's close — mark that intraday move now, since
                    # pnl_dol above was computed with yesterday's (zero) units.
                    pnl_dol       += (lr_t - fill_lr) * units * notional
                    pnl_dol       -= entry_cost_dol
                    entry_diag     = {c: float(feat[c].iloc[i]) for c in diag_cols}

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
                'entry_date':     lr.index[entry_idx],
                'exit_date':      lr.index[-1],
                'hold_days':      len(lr) - 1 - entry_idx,
                'direction':      direction,
                'entry_z':        entry_z,
                'exit_z':         float(z.iloc[-1]),
                'entry_ze':       ze_long if direction == 'long' else ze_short,
                'exit_zx':        zx_long if direction == 'long' else zx_short,
                'stop_zs':        zs_long if direction == 'long' else zs_short,
                'notional':       notional,
                'leverage':       self.LEVERAGE,
                'beta':           entry_beta,
                'units':          units,
                'entry_lr':       entry_lr,
                'exit_lr':        float(lr.iloc[-1]),
                'entry_cost_dol': entry_cost_dol,
                'exit_cost_dol':  exit_cost_dol,
                'gross_pnl_dol':  gross_pnl_dol,
                'pnl':            trade_pnl_dol / self.TRADE_CAPITAL,
                'pnl_dol':        trade_pnl_dol,
                'exit_reason':    'end_of_data',
                **entry_diag,
            })

        pnl_s     = pd.Series(daily_pnl_pct, index=lr.index[1:])
        equity_s  = pd.Series(equity_ts,     index=lr.index[1:])
        trades_df = (pd.DataFrame(trades) if trades
                     else pd.DataFrame(columns=[
                         'entry_date', 'exit_date', 'hold_days', 'direction',
                         'entry_z', 'exit_z', 'entry_ze', 'exit_zx', 'stop_zs',
                         'notional', 'leverage', 'beta', 'units',
                         'entry_lr', 'exit_lr', 'entry_cost_dol', 'exit_cost_dol',
                         'gross_pnl_dol', 'pnl', 'pnl_dol', 'exit_reason'] + diag_cols))

        return pnl_s, equity_s, trades_df