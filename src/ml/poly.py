import os, requests, pandas as pd
from datetime import date

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

BASE = "https://api.polygon.io"

def _params(api_key: str | None = None):
    key = api_key if api_key is not None else os.getenv("POLYGON_API_KEY", "")
    return {"apiKey": key} if key else {}

def daily_bars(ticker: str, start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    """Return split‑*unadjusted* daily OHLCV from Polygon."""
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"

    r = requests.get(url, params=_params(api_key))
    # print("URL:", r.url)  # debug
    r = r.json()
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

