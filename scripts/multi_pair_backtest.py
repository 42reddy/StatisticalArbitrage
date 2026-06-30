"""
Multi-pair backtest with a single shared capital slot.

Capital constraint: at most ONE position (one pair, long or short the
spread) open at any time across the whole basket. When flat, every
pair's entry signal is checked on each bar; if more than one pair
triggers on the same bar, the one with the largest |z_slow| gets the
capital (most extreme dislocation wins the tie).

Reuses params.py's global entry/exit/stop thresholds for every pair
(deliberately not tuned per pair — see window_optimize.py's docstring for
why). slow_window/medium_window ARE tuned per pair, via WINDOW_OVERRIDES
below (produced by window_optimize.py). Uses main.py's current hedge-ratio
/ capital methodology: causal Kalman beta, 200k capital, 4x leverage.

Organized as a MultiPairBacktest class so other scripts (e.g. robustness
tests) can reuse load_pairs()/run_per_pair()/allocate_capital() on
sub-slices or sub-baskets without copy-pasting this pipeline.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from features import features
from backtest import backtest
from metrics import metrics
from params import PARAMS
from data import data_loader

RESULTS_TABLES  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'tables')
RESULTS_FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'figures')

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
PAIRS = [
    ('DABUR.NS',      'HINDUNILVR.NS'),
    ('NHPC.NS',       'NTPC.NS'),
    ('BDL.NS',        'HAL.NS'),
    ('NHPC.NS',       'POWERGRID.NS'),
    ('BAJFINANCE.NS', 'KOTAKBANK.NS'),
    ('HDFCBANK.NS',   'KOTAKBANK.NS'),
    ('OIL.NS',        'ONGC.NS')
]

# Per-pair (slow_window, medium_window) — from window_optimize.py, which
# centres a walk-forward CV search on each pair's OWN OU half-life (measured
# with a static full-sample beta, never the live Kalman beta — see that
# file's docstring for why). Entry/exit/stop stay at PARAMS' global,
# untuned values for every pair, deliberately not optimized per pair.
WINDOW_OVERRIDES = {
    'DABUR/HINDUNILVR':     dict(slow_window=25, medium_window=15),
    'NHPC/NTPC':            dict(slow_window=50, medium_window=29),
    'BDL/HAL':              dict(slow_window=40, medium_window=24),
    'NHPC/POWERGRID':       dict(slow_window=41, medium_window=24),
    'BAJFINANCE/KOTAKBANK': dict(slow_window=23, medium_window=13),
    'HDFCBANK/KOTAKBANK':   dict(slow_window=34, medium_window=20),
    'OIL/ONGC':             dict(slow_window=64, medium_window=37),
}

START    = PARAMS['start']
CAPITAL  = 200_000
LEVERAGE = 4

KALMAN_DELTA       = 1e-6
KALMAN_INIT_WINDOW = 1000

THRESHOLD_SOURCE_PAIR = "DABUR/HINDUNILVR"  # the pair params.py's z-thresholds were tuned on


class MultiPairBacktest:
    """
    Shared-capital multi-pair backtest pipeline.

    Usage:
        mp = MultiPairBacktest()
        mp.load_pairs()
        mp.run()                 # -> mp.trades_df, mp.pnl_s, mp.equity_s, mp.met
        mp.print_summary()
        mp.print_diagnostics()
        mp.plot()

    Each stage is its own method so callers (e.g. robustness/stability
    tests) can reuse pieces — run_per_pair() on a different engine,
    allocate_capital() on a subset of pairs, etc. — without re-deriving
    the data-loading / feature-building boilerplate.
    """

    def __init__(self, pairs=PAIRS, window_overrides=WINDOW_OVERRIDES, params=PARAMS,
                 start=START, capital=CAPITAL, leverage=LEVERAGE,
                 kalman_delta=KALMAN_DELTA, kalman_init_window=KALMAN_INIT_WINDOW,
                 verbose=True):
        self.pairs = pairs
        self.window_overrides = window_overrides
        self.params = params
        self.start = start
        self.capital = capital
        self.leverage = leverage
        self.kalman_delta = kalman_delta
        self.kalman_init_window = kalman_init_window
        self.verbose = verbose

        self.loader = data_loader()
        self.feature_builder = features()
        self.metrics_calc = metrics()
        self.backtest_engine = backtest(trade_capital=capital, leverage=leverage)

        self.pair_data = {}
        self.names = []
        self.per_pair = {}
        self.trades_df = None
        self.pnl_s = None
        self.equity_s = None
        self.met = None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ─────────────────────────────────────────────
    #  DATA LOADING (reuses data.py's data_loader, batch T1+T2 download)
    #
    #  yfinance rate-limits back-to-back requests — main.py never hits this
    #  because it only loads one pair, but looping over many pairs here
    #  trips it intermittently. Retry with backoff rather than touching
    #  data.py, which is fine for its normal single-pair use.
    # ─────────────────────────────────────────────
    def _load_pair(self, t1, t2, attempts=6, backoff=6):
        self.loader.T1, self.loader.T2, self.loader.START = t1, t2, self.start
        for attempt in range(attempts):
            try:
                return self.loader.load_data()
            except ValueError:
                if attempt == attempts - 1:
                    raise
                time.sleep(backoff)

    # ─────────────────────────────────────────────
    #  BUILD FEATURES / SIGNALS PER PAIR
    # ─────────────────────────────────────────────
    def load_pairs(self):
        """Download data and build features/signals for every configured pair."""
        self.pair_data = {}
        for t1, t2 in self.pairs:
            name = f"{t1.split('.')[0]}/{t2.split('.')[0]}"
            df = self._load_pair(t1, t2)

            if len(df) < self.kalman_init_window + 100:
                raise ValueError(
                    f"{name}: only {len(df)} bars available — need > "
                    f"{self.kalman_init_window + 100} for Kalman warm-up + a usable test span.")

            self.feature_builder.T1, self.feature_builder.T2 = t1, t2
            beta = self.feature_builder.kalman_hedge_ratio(
                df, t1, t2, delta=self.kalman_delta, init_window=self.kalman_init_window)

            pair_params = self.params.copy()
            pair_params.update(self.window_overrides.get(name, {}))

            feat, _ = self.feature_builder.build_features(df, pair_params, beta=beta)
            sig = self.feature_builder.generate_signals(feat, pair_params)

            valid = feat.dropna(subset=['z_slow', 'mu_slow', 'vol']).index
            feat = feat.loc[valid]
            sig = sig.loc[valid]

            self.pair_data[name] = dict(feat=feat, sig=sig, T1=t1, T2=t2)
            self._log(f"  {name:24s}: {len(df)} bars total, {len(valid)} usable after "
                      f"Kalman warm-up  ({valid[0].date()} -> {valid[-1].date()})  "
                      f"beta(last)={beta.dropna().iloc[-1]:.3f}")

        self.names = list(self.pair_data.keys())
        return self.pair_data

    # ─────────────────────────────────────────────
    #  PER-PAIR BACKTEST — reuse the existing, verified backtest engine.
    #  Each pair is run completely independently (as if it had its own
    #  unlimited capital) so the entry/exit/fill/cost logic stays exactly
    #  what backtest.py already validates. The shared-capital constraint
    #  is then enforced post-hoc by selecting a non-overlapping subset
    #  of these trades (see allocate_capital).
    # ─────────────────────────────────────────────
    def run_per_pair(self, engine=None, pair_data=None):
        """Run every pair through `engine` independently (unlimited capital)."""
        engine = engine or self.backtest_engine
        pair_data = pair_data if pair_data is not None else self.pair_data
        out = {}
        for name, d in pair_data.items():
            pnl_p, _, trades_p = engine.backtest(d['feat'], d['sig'], self.params, cost=True)
            trades_p = trades_p.copy()
            trades_p['pair'] = name
            out[name] = dict(pnl=pnl_p, trades=trades_p)
        return out

    def allocate_capital(self, per_pair_results):
        """
        Single shared capital slot. Pool every pair's candidate trades, walk
        them in chronological order by entry_date (ties broken by largest
        |entry_z| — most extreme dislocation wins), and greedily accept a
        trade only if it starts strictly after the previously accepted
        trade's exit_date. This is equivalent to a day-by-day "whichever
        pair triggers first gets capital" simulation, since each pair's own
        entries are already non-overlapping (one trade at a time within
        that pair) by construction of backtest.py.
        """
        all_trades = pd.concat([r['trades'] for r in per_pair_results.values()], ignore_index=True)
        all_trades['abs_entry_z'] = all_trades['entry_z'].abs()
        all_trades = all_trades.sort_values(
            ['entry_date', 'abs_entry_z'], ascending=[True, False]).reset_index(drop=True)

        accepted = []
        busy_until = None
        for _, t in all_trades.iterrows():
            if busy_until is None or t['entry_date'] > busy_until:
                accepted.append(t)
                busy_until = t['exit_date']

        trades = (pd.DataFrame(accepted).drop(columns=['abs_entry_z']).reset_index(drop=True)
                  if accepted else all_trades.drop(columns=['abs_entry_z']).iloc[0:0])

        master_idx = sorted(set().union(*(r['pnl'].index for r in per_pair_results.values())))
        master_idx = pd.DatetimeIndex(master_idx)
        pnl = pd.Series(0.0, index=master_idx)
        for _, t in trades.iterrows():
            seg = per_pair_results[t['pair']]['pnl'].loc[t['entry_date']:t['exit_date']]
            pnl.loc[seg.index] += seg.values

        equity = self.capital + (pnl * self.capital).cumsum()
        return trades, pnl, equity

    def run(self):
        """Full pipeline: load pairs (if needed) -> per-pair backtest -> capital allocation -> metrics."""
        if not self.pair_data:
            self.load_pairs()
        self.per_pair = self.run_per_pair()
        self.trades_df, self.pnl_s, self.equity_s = self.allocate_capital(self.per_pair)
        self.met = self.metrics_calc.calc_metrics(self.pnl_s, self.equity_s, self.trades_df)
        return self.trades_df, self.pnl_s, self.equity_s

    def cost_stress_test(self, slippage_mult=2.0):
        """Re-run the whole pipeline with slippage scaled by `slippage_mult`."""
        stress_engine = backtest(trade_capital=self.capital, leverage=self.leverage)
        stress_engine.SLIPPAGE = stress_engine.SLIPPAGE * slippage_mult
        stress_engine.COST_RT = stress_engine.COMMISSION + stress_engine.STT + stress_engine.SLIPPAGE * 2
        stress_per_pair = self.run_per_pair(stress_engine)
        stress_trades, stress_pnl, stress_equity = self.allocate_capital(stress_per_pair)
        stress_met = self.metrics_calc.calc_metrics(stress_pnl, stress_equity, stress_trades)
        return stress_met, stress_trades, stress_pnl, stress_equity

    def save_trade_log(self, path=None):
        """
        Save the pooled, unconstrained-capital trade log (all candidate
        trades from every pair, not just the capital-allocated subset).
        Feeds multi_pair_scratch.py's collective feature-discrimination check.
        """
        path = path or os.path.join(RESULTS_TABLES, 'trades_multipair.csv')
        all_trades_log = pd.concat([r['trades'] for r in self.per_pair.values()], ignore_index=True)
        all_trades_log.to_csv(path, index=False)
        self._log(f"  Saved {len(all_trades_log)} trade logs (guards disabled) -> {path}")
        return all_trades_log

    # ─────────────────────────────────────────────
    #  SUMMARY
    # ─────────────────────────────────────────────
    def print_summary(self):
        met = self.met
        n_days = len(self.pnl_s)
        util = self.trades_df['hold_days'].sum() / n_days * 100 if len(self.trades_df) else 0.0

        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  MULTI-PAIR PORTFOLIO  (single shared capital slot, {self.capital:,} cap, {self.leverage}x lev)
╠══════════════════════════════════════════════════════════════════╣
║  Ann. return        :  {met['ann_ret']*100:>8.2f}%
║  Ann. vol           :  {met['ann_vol']*100:>8.2f}%
║  Max drawdown       :  {met['max_dd']*100:>8.2f}%
║  Calmar ratio       :  {met['calmar']:>8.2f}   <- primary risk-adj metric
║  Sharpe ratio       :  {met['sharpe']:>8.2f}
║  Sortino ratio      :  {met['sortino']:>8.2f}
╠══════════════════════════════════════════════════════════════════╣
║  Total trades       :  {met['n_trades']:>8}
║  Trades / year      :  {met['n_trades'] / (n_days/252):>8.1f}
║  Win rate           :  {met['win_rate']*100:>8.1f}%
║  Avg win            :  {met['avg_win']*self.capital:>8,.0f}
║  Avg loss           :  {met['avg_loss']*self.capital:>8,.0f}
║  Profit factor      :  {met['profit_factor']:>8.2f}
║  Avg hold           :  {met['avg_hold']:>8.1f}d (std {met['std_hold']:.1f}d)
║  Capital utilization:  {util:>8.1f}%
╚══════════════════════════════════════════════════════════════════╝
""")

        if len(self.trades_df):
            print("Per-pair breakdown:")
            print(f"  {'Pair':<24} {'n':>4} {'WR':>7} {'Avg PnL':>10} {'Total PnL':>12} {'Avg Hold':>9} {'Std Hold':>9}")
            for pair, grp in self.trades_df.groupby('pair'):
                wr  = (grp['pnl'] > 0).mean() * 100
                avg = grp['pnl'].mean() * self.capital
                tot = grp['pnl'].sum() * self.capital
                hd  = grp['hold_days'].mean()
                hs  = grp['hold_days'].std()
                print(f"  {pair:<24} {len(grp):>4} {wr:>6.1f}% {avg:>10,.0f} {tot:>12,.0f} {hd:>8.1f}d {hs:>8.1f}d")

        if len(self.trades_df):
            print("\nLast 30 trades:")
            print(f"  {'Pair':<24} {'Entry':<11} {'Exit':<11} {'Hold':>5} "
                  f"{'Dir':<6} {'PnL':>12} {'Exit Reason':<12}")
            for _, t in self.trades_df.sort_values('exit_date').tail(30).iterrows():
                print(f"  {t['pair']:<24} {t['entry_date'].date()!s:<11} {t['exit_date'].date()!s:<11} "
                      f"{t['hold_days']:>5} {t['direction']:<6} {t['pnl_dol']:>12,.0f} {t['exit_reason']:<12}")

    # ─────────────────────────────────────────────
    #  RED-FLAG / GREEN-FLAG DIAGNOSTICS
    #  None of this is optional polish — each check targets a specific way
    #  a capital-constrained, untuned multi-pair result can look good while
    #  being undeployable: small samples, one lucky regime, one fat trade,
    #  correlated legs masking the diversification this whole exercise is
    #  meant to buy, or thresholds that quietly don't generalize.
    # ─────────────────────────────────────────────
    def print_diagnostics(self):
        trades_df, pnl_s, equity_s = self.trades_df, self.pnl_s, self.equity_s

        print("\n" + "═" * 70)
        print("  DIAGNOSTICS — red flags / green flags")
        print("═" * 70)

        # ── 1. Sample size per pair ─────────────────────────────────────
        print("\n[1] Sample size (n<15 trades -> not statistically meaningful)")
        for pair, grp in trades_df.groupby('pair'):
            flag = "  <-- RED: too few trades to trust" if len(grp) < 15 else ""
            tag = "  (threshold-source pair, in-sample)" if pair == THRESHOLD_SOURCE_PAIR else ""
            print(f"  {pair:<24} n={len(grp):>3}{tag}{flag}")

        # ── 2. Year-by-year stability ─────────────────────────────────────
        print("\n[2] Year-by-year stability (is the edge concentrated in one regime?)")
        print(f"  {'Year':<6}{'n':>5}{'WR':>8}{'Ann.Ret':>10}{'Ann.Sharpe':>12}")
        for yr, grp_pnl in pnl_s.groupby(pnl_s.index.year):
            yr_trades = trades_df[trades_df['exit_date'].dt.year == yr] if len(trades_df) else trades_df
            n = len(yr_trades)
            wr = (yr_trades['pnl'] > 0).mean() * 100 if n else float('nan')
            ann_ret = grp_pnl.mean() * 252
            ann_vol = grp_pnl.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
            print(f"  {yr:<6}{n:>5}{wr:>7.1f}%{ann_ret*100:>9.1f}%{sharpe:>12.2f}")

        # ── 3. Trade concentration ─────────────────────────────────────
        print("\n[3] Trade concentration (is total P&L riding on a few fat trades?)")
        if len(trades_df):
            gross_profit = trades_df.loc[trades_df['pnl_dol'] > 0, 'pnl_dol'].sum()
            top5_profit = trades_df.nlargest(5, 'pnl_dol')['pnl_dol'].sum()
            top5_pct = top5_profit / gross_profit * 100 if gross_profit > 0 else 0.0
            skew = trades_df['pnl_dol'].skew()
            flag = "  <-- RED: top 5 trades drive most of the profit" if top5_pct > 40 else ""
            print(f"  Top-5 winning trades  : {top5_pct:.1f}% of gross profit{flag}")
            print(f"  Trade P&L skew        : {skew:.2f}")

        # ── 4. Drawdown duration ───────────────────────────────────────
        print("\n[4] Drawdown duration (how long underwater, not just how deep)")
        roll_max = equity_s.cummax()
        underwater = equity_s < roll_max
        run_id = (~underwater).cumsum()
        longest_dd_days = underwater.groupby(run_id).sum().max() if underwater.any() else 0
        print(f"  Longest stretch underwater: {longest_dd_days} trading days "
              f"(~{longest_dd_days/252:.1f}y)")

        # ── 5. Beta / cointegration stability per pair ──────────────────
        print("\n[5] Beta stability (drifting/sign-flipping beta = cointegration breaking down)")
        for name, d in self.pair_data.items():
            b = d['feat']['beta'].dropna()
            sign_flips = ((b > 0).astype(int).diff().abs() > 0).sum()
            flag = "  <-- RED: sign flip(s), relationship unstable" if sign_flips > 0 else ""
            print(f"  {name:<24} beta range=[{b.min():.2f}, {b.max():.2f}]  "
                  f"std={b.std():.3f}  sign_flips={sign_flips}{flag}")

        # ── 6. Cross-pair correlation (true diversification check) ──────
        # This is the whole premise of the multi-pair idea — if the legs are
        # correlated, the sqrt(N) reduction in Sharpe std doesn't show up.
        print("\n[6] Cross-pair daily P&L correlation, UNCONSTRAINED capital "
              "(tests the diversification premise directly)")
        unconstrained_pnl = pd.DataFrame({name: r['pnl'] for name, r in self.per_pair.items()}).fillna(0.0)
        corr = unconstrained_pnl.corr()
        avg_offdiag_corr = (corr.values.sum() - len(corr)) / (len(corr) ** 2 - len(corr))
        print(corr.round(2).to_string())
        flag = "  <-- RED: legs not diversified, sqrt(N) thesis weakened" if avg_offdiag_corr > 0.2 else "  <-- GREEN: pairs are largely uncorrelated"
        print(f"  Average pairwise correlation: {avg_offdiag_corr:.3f}{flag}")

        # ── 7. Cost-stress test ──────────────────────────────────────────
        print("\n[7] Cost stress test (2x slippage — is the edge cost-fragile?)")
        stress_met, _, _, _ = self.cost_stress_test(slippage_mult=2.0)
        sharpe_degradation = ((self.met['sharpe'] - stress_met['sharpe']) / self.met['sharpe'] * 100
                               if self.met['sharpe'] != 0 else float('nan'))
        flag = "  <-- RED: edge is cost-fragile" if sharpe_degradation > 40 else ""
        print(f"  Base Sharpe     : {self.met['sharpe']:.2f}")
        print(f"  2x-slippage Sharpe: {stress_met['sharpe']:.2f}  "
              f"({sharpe_degradation:+.0f}% change){flag}")

        print("\n" + "═" * 70 + "\n")

    # ─────────────────────────────────────────────
    #  PLOTS
    # ─────────────────────────────────────────────
    def plot(self, save_path=None, show=True):
        save_path = save_path or os.path.join(RESULTS_FIGURES, 'multi_pair_backtest.png')
        trades_df, equity_s = self.trades_df, self.equity_s

        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False,
                                  gridspec_kw={'height_ratios': [2, 1, 1.2]})

        ax = axes[0]
        ax.plot(equity_s.index, equity_s.values, color='steelblue', lw=1.2)
        ax.set_title('Portfolio equity curve (single shared capital slot)')
        ax.set_ylabel('Equity ()')
        ax.grid(alpha=0.3)

        ax = axes[1]
        dd = (equity_s - equity_s.cummax()) / equity_s.cummax() * 100
        ax.fill_between(dd.index, dd.values, 0, color='firebrick', alpha=0.4)
        ax.set_title('Drawdown (%)')
        ax.set_ylabel('%')
        ax.grid(alpha=0.3)

        ax = axes[2]
        if len(trades_df):
            pair_pnl = trades_df.groupby('pair')['pnl_dol'].sum().sort_values()
            colors = ['firebrick' if v < 0 else 'seagreen' for v in pair_pnl.values]
            ax.barh(pair_pnl.index, pair_pnl.values, color=colors)
            ax.set_title('Total P&L by pair ()')
            ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(save_path, dpi=130)
        self._log(f"Saved plot -> {save_path}")
        if show:
            plt.show()
        return fig


if __name__ == "__main__":
    mp = MultiPairBacktest()
    mp.load_pairs()
    mp.run()
    mp.save_trade_log()
    mp.print_summary()
    mp.print_diagnostics()
    mp.plot()
