import os, requests, pandas as pd
from datetime import date

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

BASE = "https://api.polygon.io"

def _params():
    key = os.getenv("POLYGON_API_KEY", "")
    return {"apiKey": key} if key else {}

def daily_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Return split‑*unadjusted* daily OHLCV from Polygon."""
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    
    response = requests.get(url, params=_params())
    print("URL:", response.url)  # ← show full API URL with key
    # print("Response JSON:", response.json())  # ← check for status & error
    
    r   = requests.get(url, params=_params()).json()
    if r.get("status") not in {"OK", "DELAYED"}:
        raise RuntimeError(r.get("error", "Polygon error"))
    df = pd.DataFrame(r["results"])

    df.rename(columns={
        "t":"Date","o":"Open","h":"High","l":"Low",
        "c":"Close","v":"Volume"
    }, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], unit="ms")
    df.sort_values("Date", inplace=True)
    return df

def latest_close(ticker: str) -> float:
    url = f"{BASE}/v2/aggs/ticker/{ticker}/prev"
    r   = requests.get(url, params=_params()).json()
    return float(r["results"][0]["c"])

