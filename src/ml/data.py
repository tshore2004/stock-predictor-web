from __future__ import annotations
from datetime import date
import numpy as np, pandas as pd, yfinance as yf, ta
import pandas_datareader.data as web     
from sklearn.preprocessing import StandardScaler
from .poly import daily_bars

SEQ_LEN = 60
FEATS = [
    "Close", "MA10", "MA50", "RSI",
    "MACD", "MACD_Signal", "BB_High", "BB_Low"
]

# def _download(stock: str, start: str, end: str) -> pd.DataFrame:

#     # url = f"https://stooq.com/q/d/l/?s={stock}.us&i=d"
#     # df  = pd.read_csv(url, parse_dates=["Date"])
#     # df  = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
#     # df.sort_values("Date", inplace=True)
#     # df.rename(columns=lambda c: c.title(), inplace=True)   # Open -> Open, etc.
   
#     df = web.DataReader(stock, "stooq", start, end)
#     df.rename(columns=lambda c: c.title(), inplace=True)  
#     df.reset_index(inplace=True)
#     return df

def _download(stock, start, end):
    df = daily_bars(stock, start, end)
    df.rename(columns=lambda c: c.title(), inplace=True)  
    df.reset_index(inplace=True)
    return df

def _windowize(mat: np.ndarray, n: int) -> np.ndarray:
    return np.stack([mat[i:i+n] for i in range(len(mat)-n)]).astype("float32")

def load_data(stock: str, start: str, end: str):
    df = _download(stock, start, end)          # ← uses retry + Stooq
    if df.empty:
        raise RuntimeError(f"No price data for {stock} after all attempts")

    # --- technical indicators ---
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"]  = ta.momentum.RSIIndicator(df["Close"]).rsi()

    macd = ta.trend.MACD(df["Close"])
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"]  = bb.bollinger_lband()

    df.dropna(inplace=True)
    if len(df) < SEQ_LEN + 1:
        raise ValueError("Not enough rows after indicators; extend your date range")

    # --- scale & window ----------------------------------------------
    # 1) Drop the last row so shift(-1) can never create a trailing NaN
    df = df.iloc[:-1].copy() 


    # 2) Target = next‑day close (no NaNs now)
    df["Return"] = df["Close"].pct_change().shift(-1)
    df.dropna(inplace=True)
    
    scaler   = StandardScaler()
    features = scaler.fit_transform(df[FEATS].to_numpy("float32"))
  
    X = _windowize(features, SEQ_LEN)      
    
    y = df["Return"].to_numpy("float32")[SEQ_LEN:].reshape(-1, 1)
    return X, y, scaler

def fetch_features(ticker: str, api_key: str | None = None) -> np.ndarray | None:
    """
    Return the most-recent 60×8 *raw* feature matrix for `ticker`.
    Raw = not standard-scaled; /predict will scale it.

    Returns:
        np.ndarray shape (SEQ_LEN, len(FEATS))  or  None if no data.
    """
    today = date.today().isoformat()
    start = "2020-09-01"
    try:
        df = daily_bars(ticker, start, today, api_key=api_key)
        print("Polygon rows:", len(df)) 
    except Exception:
        df = web.DataReader(ticker, "stooq", start, today)

    if df.empty:
        return None

    df.rename(columns=str.title, inplace=True)
    df.reset_index(inplace=True)

    # technical indicators
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"]  = ta.momentum.RSIIndicator(df["Close"]).rsi()

    macd = ta.trend.MACD(df["Close"])
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"]  = bb.bollinger_lband()
    df.dropna(inplace=True)

    if len(df) < SEQ_LEN:
        return None

    return df[FEATS].tail(SEQ_LEN).to_numpy("float32")
