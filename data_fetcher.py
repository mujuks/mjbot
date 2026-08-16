import yfinance as yf
import pandas as pd


def fetch_forex_data(symbol: str, interval: str, lookback_days: int = 30) -> pd.DataFrame:
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
    return df
