import yfinance as yf
import pandas as pd


class data_loader():

    def __init__(self):

        self.T1 = "BZ=F"
        self.T2 = "CL=F"
        self.lookback = "15y"


    def load_data(self):

        T1 = self.T1
        T2 = self.T2
        lookback = self.lookback

        print(f"Downloading {T1} and {T2}...")
        p1 = yf.download(T1, period=lookback, auto_adjust=True, progress=False)['Close'].squeeze()
        p2 = yf.download(T2, period=lookback, auto_adjust=True, progress=False)['Close'].squeeze()
        df = pd.concat([p1, p2], axis=1, join='inner').dropna()
        df.columns = [T1, T2]
        print(f"  {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
        return df





