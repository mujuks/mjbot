import time as _time

import pandas as pd
import requests

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def fetch_forex_data(symbol: str, interval: str, lookback_days: int = 30) -> pd.DataFrame:
    max_retries = 5
    for attempt in range(max_retries):
        try:
            r = requests.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol,
                params={"range": f"{lookback_days}d", "interval": interval, "includePrePost": "false"},
                headers=_HEADERS,
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            result = data["chart"]["result"][0]
            ts = result["timestamp"]
            q = result["indicators"]["quote"][0]

            df = pd.DataFrame({
                "Open": q["open"],
                "High": q["high"],
                "Low": q["low"],
                "Close": q["close"],
                "Volume": q.get("volume", [0] * len(ts)),
            }, index=pd.to_datetime(ts, unit="s", utc=True))

            df = df.dropna()
            if df.empty:
                raise ValueError("Empty dataframe after filtering")
            return df

        except Exception as e:
            if attempt < max_retries - 1:
                _time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"Failed to fetch {symbol} {interval} after {max_retries} attempts: {e}")
