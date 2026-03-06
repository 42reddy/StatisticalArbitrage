import numpy as np


class metrics():

    def __init__(self):
        pass

    # ══════════════════════════════════════════════════════
    # 6. METRICS
    # ══════════════════════════════════════════════════════
    def calc_metrics(self, pnl, equity, trades):
        ann_ret = pnl.mean() * 252
        ann_vol = pnl.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        down = pnl[pnl < 0]
        sortino = ann_ret / (down.std() * np.sqrt(252)) if len(down) > 1 else 0.0

        roll_max = equity.cummax()
        dd = (equity - roll_max) / roll_max
        max_dd = dd.min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

        n = len(trades)
        if n > 0:
            wr = (trades['pnl'] > 0).mean()
            wins = trades.loc[trades['pnl'] > 0, 'pnl']
            loss = trades.loc[trades['pnl'] < 0, 'pnl']
            aw = wins.mean() if len(wins) > 0 else 0.0
            al = loss.mean() if len(loss) > 0 else 0.0
            ah = trades['hold_days'].mean()
            pf = abs(wins.sum() / loss.sum()) if loss.sum() != 0 else np.inf
            py_ = trades['pyramided'].mean() * 100 if 'pyramided' in trades else 0
        else:
            wr = aw = al = ah = pf = py_ = 0.0

        return dict(sharpe=sharpe, sortino=sortino, ann_ret=ann_ret,
                    ann_vol=ann_vol, max_dd=max_dd, calmar=calmar,
                    n_trades=n, win_rate=wr, avg_win=aw, avg_loss=al,
                    avg_hold=ah, profit_factor=pf, pct_pyramided=py_)



