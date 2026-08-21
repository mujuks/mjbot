import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _path(pair: str, tf: str) -> str:
    safe = f"{pair.replace('=', '_').replace('-', '_')}_{tf}.csv"
    return os.path.join(DATA_DIR, safe)


def append_candles(pair: str, tf: str, df: pd.DataFrame) -> int:
    """Merge new candles into the local store; returns total rows stored."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _path(pair, tf)
    if os.path.exists(path):
        try:
            old = pd.read_csv(path, index_col=0, parse_dates=True)
            combined = pd.concat([old[~old.index.isin(df.index)], df]).sort_index()
        except Exception:
            combined = df
    else:
        combined = df
    combined.to_csv(path)
    return len(combined)


def load_candles(pair: str, tf: str) -> pd.DataFrame | None:
    path = _path(pair, tf)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if df is not None and len(df) else None
    except Exception:
        return None
