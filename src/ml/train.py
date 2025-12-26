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
# Pretrained models shipped with the repo
REPO_MODEL_DIR = Path(__file__).resolve().parent / "models"


# ---------- 2. Model architecture ----------
def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input((SEQ_LEN, len(FEATS))),
            # Add bidirectional LSTM for better pattern recognition
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999
        ),
        loss="mse",
        metrics=["mae"]
    )

    # Add learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )

    cb = tf.keras.callbacks.EarlyStopping(
        patience=15,  # Increased patience
        restore_best_weights=True,
        monitor='val_loss'
    )

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,  # More epochs
        batch_size=64,  # Try different batch sizes
        callbacks=[cb, lr_scheduler],
        verbose=0,
    )
    
    best_mae = float(np.nanmin(hist.history["val_mae"]))
    yhat = model.predict(X_val, verbose=0).reshape(-1)
    yv = y_val.reshape(-1)

    val_rmse = float(np.sqrt(np.mean((yhat - yv) ** 2)))
    hit_rate = float(np.mean((yhat > 0) == (yv > 0)))  # directional accuracy (0..1)

    # Add additional resume-friendly metrics
    # R² (Coefficient of Determination)
    ss_res = np.sum((yv - yhat) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

    # Correlation coefficient
    correlation = float(np.corrcoef(yhat, yv)[0, 1]) if len(yhat) > 1 else 0.0

    # Accuracy within threshold (within 2% of actual return)
    threshold = 0.02
    within_threshold = float(np.mean(np.abs(yhat - yv) < threshold))

    # Mean Absolute Percentage Error
    mape = float(np.mean(np.abs((yv - yhat) / (np.abs(yv) + 1e-8)))) * 100

    baseline = float(daily_bars(ticker, date_from, date_to)["Close"].iloc[-1])

    metrics = {
        "val_mae": best_mae,                 # percent error on returns
        "val_rmse": val_rmse,                # percent RMSE on returns
        "hit_rate": hit_rate,                # e.g., 0.62 = 62% direction accuracy
        "r_squared": r_squared,              # R² score (0-1, higher is better)
        "correlation": correlation,           # Correlation with actual returns
        "accuracy_within_2pct": within_threshold,  # % predictions within 2%
        "mape": mape,                        # Mean Absolute Percentage Error
        "baseline": baseline,                # last close
        "mae_dollar": best_mae * baseline,   # $ MAE
        "rmse_dollar": val_rmse * baseline,  # $ RMSE
    }

    # write per‑ticker (temp cache) and legacy repo file
    metrics_path = MODEL_DIR / f"{ticker}.json"
    metrics_path.write_text(json.dumps(metrics))

    repo_metrics = Path("models") / "metrics.json"
    repo_metrics.parent.mkdir(exist_ok=True)
    repo_metrics.write_text(json.dumps(metrics))

    # 3-d  Save artefacts
    model_path = MODEL_DIR / f"{ticker}.keras"
    model.save(model_path, include_optimizer=False)
    joblib.dump(scaler, MODEL_DIR / f"{ticker}.joblib")

    print(f"✓ trained {ticker}: {model_path.relative_to(MODEL_DIR)}")
    return model_path

def model_path(ticker: str) -> Path:
    """Return full Path to the cached keras file for this ticker."""
    return MODEL_DIR / f"{ticker.upper()}.keras"

def repo_model_path(ticker: str) -> Path:
    return REPO_MODEL_DIR / f"{ticker.upper()}.keras"

def scaler_path(ticker: str) -> Path:
    """Return Path to the cached scaler.joblib for this ticker."""
    p = MODEL_DIR / f"{ticker.upper()}.joblib"
    return p if p.exists() else repo_scaler_path(ticker)

def repo_scaler_path(ticker: str) -> Path:
    return REPO_MODEL_DIR / f"{ticker.upper()}.joblib"

def load_cached_model(ticker: str):
    """Load keras model from tmp or repo, else return None."""
    from tensorflow import keras
    for path in [model_path(ticker), repo_model_path(ticker)]:
        if path.exists():
            return keras.models.load_model(path)
    return None

# Add custom loss that penalizes wrong direction more
def directional_loss(y_true, y_pred):
    """Penalize wrong direction predictions more heavily"""
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    direction_penalty = tf.keras.backend.mean(
        tf.keras.backend.cast(
            tf.keras.backend.not_equal(
                tf.keras.backend.sign(y_true), 
                tf.keras.backend.sign(y_pred)
            ), 
            tf.float32
        )
    )
    return mse + 0.5 * direction_penalty

# Add this helper function
def clear_cache(ticker: str = None):
    """Clear cached models and scalers for a ticker, or all if ticker is None."""
    if ticker:
        ticker = ticker.upper()
        for ext in [".keras", ".joblib", ".json"]:
            path = MODEL_DIR / f"{ticker}{ext}"
            if path.exists():
                path.unlink()
    else:
        # Clear all
        for path in MODEL_DIR.glob("*"):
            if path.is_file():
                path.unlink()
