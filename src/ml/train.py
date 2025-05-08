from pathlib import Path
import tensorflow as tf, joblib
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
import json, numpy as np
from pandas_datareader import data as web
from src.ml.data import load_data, _download, SEQ_LEN, FEATS
from src.ml.poly import daily_bars

MODEL_DIR = Path("models/stock_predictor")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input((SEQ_LEN, len(FEATS))),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1),
    ])

if __name__ == "__main__":
    TICKER      = "AAPL"
    RANGE_FROM  = "2020-09-01"
    RANGE_TO = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    X, y, scaler = load_data(TICKER, RANGE_FROM, RANGE_TO)
    # df_all = web.DataReader(TICKER, "stooq", RANGE_FROM, RANGE_TO)
    df_all = daily_bars(TICKER, RANGE_FROM, RANGE_TO)
    baseline = float(df_all["Close"].iloc[-1])
                     
    # 2. train/validation split (time‑order safe)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 3. build & train
    model = build_model()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    cb = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)

    history = model.fit(X_tr, y_tr,
              validation_data=(X_val, y_val),
              epochs=50,
              batch_size=32,
              callbacks=[cb],
              verbose=2)

    best_mae = float(np.nanmin(history.history["val_mae"]))

with open("models/metrics.json", "w") as f:
    json.dump({"val_mae": best_mae, "baseline": baseline}, f)

    # 4. save artefacts for the FastAPI server
model.save("models/stock_predictor.keras", include_optimizer=False)
joblib.dump(scaler, "models/scaler.joblib")
print("✓ model (.keras) and scaler saved")