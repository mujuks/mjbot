import asyncio
import logging

import pandas as pd

from data_store import load_candles
from tv_history import TvHistoryError, fetch_tv_history

log = logging.getLogger("data_fetcher")

_NETWORK_ERRORS = (
    TvHistoryError, OSError, asyncio.TimeoutError,
    ConnectionError, ConnectionRefusedError, ConnectionResetError,
    TimeoutError, OSError, OSError,
)
try:
    import websockets.exceptions
    _NETWORK_ERRORS = _NETWORK_ERRORS + (
        websockets.exceptions.WebSocketException,
    )
except ImportError:
    pass


def _merge_with_store(df: pd.DataFrame, symbol: str, interval: str,
                      lookback_days: int) -> pd.DataFrame:
    """Union TV candles with locally stored history so depth stays consistent."""
    if not df.index.tz:
        return df
    store = load_candles(symbol, interval)
    if store is None or not len(store):
        return df
    try:
        merged = pd.concat([store[~store.index.isin(df.index)], df])
        merged.index = pd.to_datetime(pd.Index(merged.index), utc=True)
        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    except Exception as e:
        log.warning("store merge failed: %s", e)
        return df
    cutoff = merged.index[-1] - pd.Timedelta(days=lookback_days)
    return merged[merged.index >= cutoff]


def fetch_forex_data(symbol: str, interval: str, lookback_days: int = 30) -> pd.DataFrame:
    """Candle history from TradingView (spot feed), falling back to local store."""
    last_err = None
    for attempt in range(3):
        try:
            df = fetch_tv_history(symbol, interval, lookback_days)
            return _merge_with_store(df, symbol, interval, lookback_days)
        except _NETWORK_ERRORS as e:
            last_err = e
            log.warning("TradingView attempt %d failed for %s %s: %s",
                        attempt + 1, symbol, interval, e)
            if attempt < 2:
                import time
                time.sleep(2 ** attempt)
        except Exception as e:
            last_err = e
            log.warning("TradingView attempt %d unexpected error for %s %s: %s",
                        attempt + 1, symbol, interval, e)
            if attempt < 2:
                import time
                time.sleep(2 ** attempt)

    store = load_candles(symbol, interval)
    if store is None or not len(store):
        raise RuntimeError(
            f"No price data for {symbol} {interval}: TradingView unreachable ({last_err}) "
            f"and no local store"
        )
    log.warning("Using local candle store fallback for %s %s (%d bars)",
                symbol, interval, len(store))
    return store
