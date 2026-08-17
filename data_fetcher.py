import time as _time

import pandas as pd
import yfinance as yf


def fetch_forex_data(symbol: str, interval: str, lookback_days: int = 30) -> pd.DataFrame:
    max_retries = 5
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            interval_map = {
                "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
                "60m": "1h", "1h": "1h", "90m": "90m", "1d": "1d", "5d": "5d",
            }
            yf_interval = interval_map.get(interval, interval)

            if yf_interval in ("1m", "2m", "5m", "15m", "30m"):
                period_days = min(lookback_days, 60)
                df = ticker.history(period=f"{period_days}d", interval=yf_interval)
            else:
                df = ticker.history(period=f"{lookback_days}d", interval=yf_interval)

            if df.empty:
                raise ValueError("Empty dataframe returned from yfinance")

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
            df = df.dropna()

            if df.empty:
                raise ValueError("Empty dataframe after filtering")
            return df

        except Exception as e:
            if attempt < max_retries - 1:
                _time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"Failed to fetch {symbol} {interval} after {max_retries} attempts: {e}")
