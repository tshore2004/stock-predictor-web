from __future__ import annotations
from datetime import date
import numpy as np, pandas as pd, yfinance as yf, ta
import pandas_datareader.data as web     
from sklearn.preprocessing import StandardScaler
from .poly import daily_bars

SEQ_LEN = 60
FEATS = [
    "Close", "Open", "High", "Low", "Volume",  # Add OHLCV
    "MA10", "MA50", "MA200",  # Add longer MA
    "RSI",
    "MACD", "MACD_Signal", "MACD_Hist",  # Add MACD histogram
    "BB_High", "BB_Low", "BB_Width",  # Add BB width
    "ATR",  # Average True Range (volatility)
    "ADX",  # Trend strength
    "Stoch",  # Stochastic oscillator
    "Volume_MA",  # Volume moving average
    "Price_Change",  # Daily price change
    "Volatility",  # Rolling volatility
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
    df = _download(stock, start, end)
    if df.empty:
        raise RuntimeError(f"No price data for {stock} after all attempts")

    # --- technical indicators ---
    # Moving Averages
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    
    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()  # MACD histogram

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["Close"]  # Normalized width

    # ATR (Average True Range)
    df["ATR"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"]
    ).average_true_range()

    # ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"])
    df["ADX"] = adx.adx()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"]
    )
    df["Stoch"] = stoch.stoch()  # %K line

    # Volume features
    df["Volume_MA"] = df["Volume"].rolling(20).mean()

    # Price features
    df["Price_Change"] = df["Close"].pct_change()
    df["Volatility"] = df["Close"].rolling(20).std()

    df.dropna(inplace=True)
    if len(df) < SEQ_LEN + 1:
        raise ValueError("Not enough rows after indicators; extend your date range")

    # --- scale & window ----------------------------------------------
    # 1) Drop the last row so shift(-1) can never create a trailing NaN
    df = df.iloc[:-1].copy() 

    # 2) Target = next‑day close (no NaNs now)
    df["Return"] = df["Close"].pct_change().shift(-1)
    df.dropna(inplace=True)
    
    scaler = StandardScaler()
    features = scaler.fit_transform(df[FEATS].to_numpy("float32"))
  
    X = _windowize(features, SEQ_LEN)      
    
    y = df["Return"].to_numpy("float32")[SEQ_LEN:].reshape(-1, 1)
    return X, y, scaler

def fetch_features(ticker: str, api_key: str | None = None) -> np.ndarray | None:
    """
    Return the most-recent 60×len(FEATS) *raw* feature matrix for `ticker`.
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

    # technical indicators - MUST MATCH load_data() exactly
    # Moving Averages
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    
    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["Close"]

    # ATR
    df["ATR"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"]
    ).average_true_range()

    # ADX
    adx = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"])
    df["ADX"] = adx.adx()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"]
    )
    df["Stoch"] = stoch.stoch()

    # Volume features
    df["Volume_MA"] = df["Volume"].rolling(20).mean()

    # Price features
    df["Price_Change"] = df["Close"].pct_change()
    df["Volatility"] = df["Close"].rolling(20).std()

    df.dropna(inplace=True)

    if len(df) < SEQ_LEN:
        return None

    return df[FEATS].tail(SEQ_LEN).to_numpy("float32")
