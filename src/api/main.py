from pathlib import Path
import numpy as np, tensorflow as tf, joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import date, timedelta
from pandas_datareader import data as web
import ta, pandas as pd
import yfinance as yf 
import joblib, os

from src.ml.poly import daily_bars, latest_close

from src.ml.train import (
    build_model,
    load_cached_model,
    train_for_ticker,
    model_path,
    scaler_path,
)
from src.ml.data import (
    load_data, fetch_features, SEQ_LEN, FEATS
)

# Pretrained models are bundled under src/ml/models and loaded before
# triggering a new training run. This avoids long cold-starts on Vercel.

METRICS_PATH = Path("models/metrics.json")

SEQ_LEN = 60
# MODEL_PATH = Path("models/stock_predictor.keras")
# SCALER_PATH = Path("models/scaler.joblib")

# model  = tf.keras.models.load_model(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)     

app = FastAPI(title="Stock Predictor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stock-predictor-web.vercel.app",
                   "http://localhost:3000"],  # adjust for prod
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping(): return {"msg": "pong"}

@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return json.loads(METRICS_PATH.read_text())

@app.get("/latest-price")
def latest_price(ticker: str = "AAPL"):
    today = date.today()
    start = today - timedelta(days=7)   # enough days to guarantee at least 1 row
    df = web.DataReader(ticker, "stooq", start, today)
    if df.empty:
        raise HTTPException(503, "No recent price")
    last_close = float(df["Close"].iloc[-1])
    return {"ticker": ticker, "price": latest_close(ticker)}

@app.get("/latest-window")
def latest_window(ticker: str = "AAPL"):
    """
    Return the most recent 60×8 RAW feature matrix for AAPL.
    Raw = not standard‑scaled; the /predict endpoint will scale.
    """
    today  = date.today().isoformat()
    start  = "2020-09-01"
    # df = web.DataReader(ticker, "stooq", start, today)
    df = daily_bars(ticker, start, today)
    df.rename(columns=str.title, inplace=True)
    df.reset_index(inplace=True)
    
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"]  = ta.momentum.RSIIndicator(df["Close"]).rsi()
    macd       = ta.trend.MACD(df["Close"])
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"]  = bb.bollinger_lband()
    df.dropna(inplace=True)

    window = df[FEATS].tail(SEQ_LEN).to_numpy("float32").tolist()
    return window

# @app.get("/train-and-predict")
# def train_and_predict(ticker: str = "AAPL"):
#     try:
#         start = "2020-01-01"
#         end = date.today().strftime("%Y-%m-%d")

#         # 1. Load data
#         X, y, scaler = load_data(ticker, start, end)
#         window_raw = X[-1]  # latest 60-day window (scaled)

#         # 2. Build & train model
#         model = build_model()
#         model.compile(optimizer="adam", loss="mse", metrics=["mae"])
#         model.fit(X, y, epochs=10, batch_size=32, verbose=0)

#         # 3. Predict
#         yhat = model.predict(window_raw.reshape(1, *window_raw.shape))[0, 0]
#         latest_close = scaler.inverse_transform([window_raw[-1]])[0][0]
#         pred_close = latest_close * (1 + yhat)

#         return {"prediction": float(pred_close)}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/predict")
# def predict(series: list[list[float]]):
#     if len(series) != SEQ_LEN:
#         raise HTTPException(status_code=400,
#             detail=f"Need {SEQ_LEN} timesteps, got {len(series)}")
#     X = np.asarray(series, dtype="float32").reshape(SEQ_LEN, -1)
#     X = scaler.transform(X).reshape(1, SEQ_LEN, -1)
#     yhat = model.predict(X, verbose=0)[0, 0]
#     latest_close = series[-1][0]                # raw Close in last row
#     pred_close   = latest_close * (1 + yhat)
#     return {"prediction": float(pred_close)}

@app.get("/predict")
def predict(ticker: str = Query(..., max_length=8)):
    """
    Train on first use, then predict next-day close for `ticker`.
    """
    ticker = ticker.upper()
    today  = date.today().isoformat()
    five_years_ago = (date.today() - timedelta(days=5*365)).isoformat()

    # ---------- 1. model cache ----------
    model = load_cached_model(ticker)
    if model is None:
        train_for_ticker(ticker, five_years_ago, today)
        model = load_cached_model(ticker)

    # ---------- 2. scaler ----------
    import joblib, tensorflow as tf                           # lazy import
    try:
        scaler = joblib.load(scaler_path(ticker))
    except FileNotFoundError:
        raise HTTPException(500, detail="Scaler missing after training")

    # ---------- 3. fetch latest window ----------
    window_raw = fetch_features(ticker, os.getenv("POLYGON_API_KEY"))
    if window_raw is None or len(window_raw) != SEQ_LEN:
        raise HTTPException(503, detail="Could not fetch latest market data")

    X = scaler.transform(window_raw).reshape(1, SEQ_LEN, -1)
    yhat = float(model.predict(X, verbose=0)[0, 0])

    latest_close = window_raw[-1][0]
    pred_close   = latest_close * (1 + yhat)

    return {"ticker": ticker, "prediction": pred_close}
