"""
One-off bulk download: pulls full history (from 2012, or the ticker's
earliest available date if listed later) for every ticker used anywhere
in this project, and saves each as a CSV in price_data/.

Run this once (and again whenever you add a new pair). After that,
data.py's data_loader reads from this cache automatically and only
falls back to yfinance for tickers it doesn't find on disk — avoids the
rate-limiting that hits multi_pair_backtest.py / window_optimize.py when
they loop over many pairs in one process.
"""
import time
import pandas as pd
import yfinance as yf

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from data import CACHE_DIR, cache_path

START = '2012-01-01'

TICKERS = [
    'DABUR.NS', 'HINDUNILVR.NS',
    'HAL.NS', 'BDL.NS',
    'SHREECEM.NS', 'HEIDELBERG.NS',
    'NHPC.NS', 'POWERGRID.NS',
    'BAJFINANCE.NS', 'KOTAKBANK.NS',
    'HDFCBANK.NS',
    'OIL.NS', 'ONGC.NS',
]


def download_with_retry(ticker, start, attempts=5, backoff=5):
    for attempt in range(attempts):
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if raw is not None and len(raw) > 0:
            return raw
        time.sleep(backoff)
    return None


if __name__ == '__main__':
    os.makedirs(CACHE_DIR, exist_ok=True)

    for ticker in TICKERS:
        raw = download_with_retry(ticker, START)
        if raw is None:
            print(f"  {ticker:<16}: FAILED after retries — not saved")
            continue

        close = raw['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = 'Close'
        close.to_frame().to_csv(cache_path(ticker))

        print(f"  {ticker:<16}: {len(close)} bars  "
              f"({close.index[0].date()} -> {close.index[-1].date()})  -> {cache_path(ticker)}")

    print(f"\nDone. Cached files in {CACHE_DIR}")
