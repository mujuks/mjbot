import time as _time

import pandas as pd
import yfinance as yf


def fetch_forex_data(symbol: str, interval: str, lookback_days: int = 30) -> pd.DataFrame:
    max_retries = 5
    for attempt in range(max_retries):
        try:
            df = yf.download(
                symbol,
                period=f"{lookback_days}d",
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df = df.dropna()
            if df.empty:
                raise ValueError("Empty dataframe after filtering")
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                _time.sleep(wait)
                continue
            raise RuntimeError(f"Failed to fetch {symbol} {interval} after {max_retries} attempts: {e}")
