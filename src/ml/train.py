# from pathlib import Path
# import tensorflow as tf, joblib
# from datetime import date, timedelta
# from sklearn.model_selection import train_test_split
# import json, numpy as np
# from pandas_datareader import data as web
# from src.ml.data import load_data, _download, SEQ_LEN, FEATS
# from src.ml.poly import daily_bars

# MODEL_DIR = Path(tempfile.gettempdir()) / "stock_models"
# MODEL_DIR.mkdir(parents=True, exist_ok=True)


# def build_model() -> tf.keras.Model:
#     return tf.keras.Sequential([
#         tf.keras.layers.Input((SEQ_LEN, len(FEATS))),
#         tf.keras.layers.LSTM(128, return_sequences=True),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.LSTM(64),
#         tf.keras.layers.Dense(1),
#     ])



# if __name__ == "__main__":
#     TICKER      = "AAPL"
#     RANGE_FROM  = "2020-09-01"
#     RANGE_TO = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

#     X, y, scaler = load_data(TICKER, RANGE_FROM, RANGE_TO)
#     # df_all = web.DataReader(TICKER, "stooq", RANGE_FROM, RANGE_TO)
#     df_all = daily_bars(TICKER, RANGE_FROM, RANGE_TO)
#     baseline = float(df_all["Close"].iloc[-1])
                     
#     # 2. train/validation split (time‑order safe)
#     X_tr, X_val, y_tr, y_val = train_test_split(
#         X, y, test_size=0.2, shuffle=False
#     )

#     # 3. build & train
#     model = build_model()
#     model.compile(optimizer="adam", loss="mse", metrics=["mae"])
#     cb = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)

#     history = model.fit(X_tr, y_tr,
#               validation_data=(X_val, y_val),
#               epochs=50,
#               batch_size=32,
#               callbacks=[cb],
#               verbose=2)

#     best_mae = float(np.nanmin(history.history["val_mae"]))

# with open("models/metrics.json", "w") as f:
#     json.dump({"val_mae": best_mae, "baseline": baseline}, f)

#     # 4. save artefacts for the FastAPI server
# model.save("models/stock_predictor.keras", include_optimizer=False)
# joblib.dump(scaler, "models/scaler.joblib")
# print("✓ model (.keras) and scaler saved")

from __future__ import annotations
from pathlib import Path
import tempfile, json, joblib, numpy as np, tensorflow as tf
from datetime import date, timedelta
from sklearn.model_selection import train_test_split

from src.ml.data import load_data, SEQ_LEN, FEATS
from src.ml.poly import daily_bars

# ---------- 1. Cross-platform, Vercel-safe model directory ----------
MODEL_DIR = Path(tempfile.gettempdir()) / "stock_models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


# ---------- 2. Model architecture ----------
def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input((SEQ_LEN, len(FEATS))),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(1),
        ]
    )


# ---------- 3. Train-on-demand ----------
def train_for_ticker(
    ticker: str,
    date_from: str,
    date_to: str,
) -> Path:
    """
    Train a model for `ticker`, save artefacts under MODEL_DIR,
    and return the .keras path.
    """
    # 3-a  Load & split data
    X, y, scaler = load_data(ticker, date_from, date_to)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 3-b  Train
    model = build_model()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    cb = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)

    hist = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[cb],
        verbose=0,
    )
    best_mae = float(np.nanmin(hist.history["val_mae"]))

    # 3-c  Metrics for baseline comparison
    baseline = float(daily_bars(ticker, date_from, date_to)["Close"].iloc[-1])

    metrics_path = MODEL_DIR / f"{ticker}.json"
    metrics_path.write_text(json.dumps({"val_mae": best_mae, "baseline": baseline}))

    # 3-d  Save artefacts
    model_path = MODEL_DIR / f"{ticker}.keras"
    model.save(model_path, include_optimizer=False)
    joblib.dump(scaler, MODEL_DIR / f"{ticker}.joblib")

    print(f"✓ trained {ticker}: {model_path.relative_to(MODEL_DIR)}")
    return model_path

def model_path(ticker: str) -> Path:
    """Return full Path to the cached keras file for this ticker."""
    return MODEL_DIR / f"{ticker.upper()}.keras"

def scaler_path(ticker: str) -> Path:
    """Return Path to the cached scaler.joblib for this ticker."""
    return MODEL_DIR / f"{ticker.upper()}.joblib"

def load_cached_model(ticker: str):
    """Load keras model if it exists, else return None (for FastAPI endpoint)."""
    from tensorflow import keras
    p = model_path(ticker)
    return keras.models.load_model(p) if p.exists() else None

# ---------- 4. Helper for FastAPI endpoint ----------
def load_model(ticker: str):
    from tensorflow import keras

    path = MODEL_DIR / f"{ticker}.keras"
    return keras.models.load_model(path) if path.exists() else None


# ---------- 5. CLI demo (runs only when you execute this file directly) ----------
if __name__ == "__main__":
    TICKER = "AAPL"
    RANGE_FROM = "2020-09-01"
    RANGE_TO = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    train_for_ticker(TICKER, RANGE_FROM, RANGE_TO)
