import yfinance as yf
import pandas as pd
import numpy as np
from params import PARAMS


class data_loader():

    def __init__(self):
        self.T1    = PARAMS['T1']
        self.T2    = PARAMS['T2']
        self.START = PARAMS.get('start', '2012-01-01')

    def load_data(self, needs_fx=False):
        """
        Load and align price data for T1 and T2.

        Parameters
        ----------
        needs_fx : bool
            Set True when T1 and T2 are denominated in different
            currencies (e.g. GOLDBEES.NS in INR vs GC=F in USD).

            When True:
            - USDINR=X is downloaded alongside T1 and T2
            - The USD leg is multiplied by USDINR to convert to INR
            - No unit or weight conversion is applied anywhere.
              The OLS hedge ratio β estimated in features.py absorbs
              any size difference between instruments automatically.
              e.g. GOLDBEES units vs gold ounces — β handles this.

            Convention: with needs_fx=True, T2 is treated as the
            USD leg and T1 as the INR leg. The reverse is also
            handled automatically via suffix detection.

            When False (default):
            - No FX download, no conversion applied
            - Use for same-currency pairs (both INR or both USD)
        """
        INR_SUFFIXES = ('.NS', '.BO')
        t1_is_inr    = any(self.T1.endswith(s) for s in INR_SUFFIXES)
        t2_is_inr    = any(self.T2.endswith(s) for s in INR_SUFFIXES)

        # ── Validate needs_fx usage ────────────────────────────
        if needs_fx and (t1_is_inr == t2_is_inr):
            print(f"  ⚠ needs_fx=True but both tickers appear to be "
                  f"{'INR' if t1_is_inr else 'USD'}-denominated. "
                  f"No conversion applied.")
            needs_fx = False

        # ── Download ───────────────────────────────────────────
        tickers_to_download = ([self.T1, self.T2, 'USDINR=X']
                                if needs_fx
                                else [self.T1, self.T2])

        raw = yf.download(
            tickers_to_download,
            start=self.START,
            auto_adjust=True,
            progress=False
        )['Close']

        if isinstance(raw, pd.Series):
            raw = raw.to_frame()

        # ── Clean ──────────────────────────────────────────────
        raw = raw.replace([np.inf, -np.inf], np.nan)
        raw = (raw[raw > 0]
               .ffill()
               .bfill()
               .dropna())

        # ── Currency conversion ────────────────────────────────
        # Multiply the USD leg by USDINR to bring it into INR terms.
        # This is the only transformation applied — no unit/weight
        # conversion. β absorbs instrument size differences.
        if needs_fx:
            if 'USDINR=X' not in raw.columns:
                raise ValueError(
                    "USDINR=X failed to download. "
                    "Check your internet connection or yfinance.")

            usdinr = raw['USDINR=X']

            # Identify the USD leg via suffix detection
            if t1_is_inr and not t2_is_inr:
                usd_ticker = self.T2   # T2 is USD (your convention)
            else:
                usd_ticker = self.T1   # T1 is USD

            # Pure currency conversion — no unit normalisation
            raw[usd_ticker] = raw[usd_ticker] * usdinr
            print(f"  FX: {usd_ticker}  USD → INR  (× USDINR)")

            # Drop USDINR — no longer needed
            raw = raw.drop(columns=['USDINR=X'])

        # ── Final alignment and validation ─────────────────────
        raw = raw[~raw.index.duplicated(keep='first')]
        raw = raw.sort_index()
        raw = raw[[self.T1, self.T2]].dropna()

        if len(raw) < 500:
            raise ValueError(
                f"Only {len(raw)} clean bars after processing. "
                f"Check ticker validity and date range.")

        fx_note = "  (FX-adjusted, USD→INR)" if needs_fx else ""
        print(f"  Loaded  : {len(raw)} bars{fx_note}  "
              f"({raw.index[0].date()} → {raw.index[-1].date()})")

        return raw

