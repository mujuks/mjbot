"""TradingView historical OHLCV via the public chart-data websocket.

Replaces yfinance as the primary candle source. Uses an anonymous session
(no login required) against wss://data.tradingview.com and paginates with
request_more_data until the requested window is filled.

Symbols are mapped so futures-style config pairs keep working while pulling
the more liquid spot feed (e.g. GC=F -> OANDA:XAUUSD). The local candle
store stays keyed by the original pair name, preserving accumulated history.
"""
import asyncio
import json
import logging
import random
import re
import string
from datetime import datetime, timezone

import pandas as pd
import websockets

log = logging.getLogger("tv_history")

TV_WS_URL = "wss://data.tradingview.com/socket.io/websocket?origin=www.tradingview.com"

INTERVALS = {
    "1m": "1", "2m": "2", "3m": "3", "5m": "5", "15m": "15",
    "30m": "30", "45m": "45", "1h": "60", "60m": "60",
    "2h": "120", "4h": "240", "1d": "D",
}

_INTERVAL_MINUTES = {
    "1": 1, "2": 2, "3": 3, "5": 5, "15": 15, "30": 30,
    "45": 45, "60": 60, "120": 120, "240": 240,
}

SYMBOL_MAP = {
    "GC=F": "OANDA:XAUUSD",
    "MGC=F": "OANDA:XAUUSD",
    "XAUUSD=X": "OANDA:XAUUSD",
    "XAUUSD": "OANDA:XAUUSD",
}

_FRAME_RE = re.compile(r"~m~(\d+)~m~")
_MAX_PAGES = 12


class TvHistoryError(RuntimeError):
    pass


def resolve_symbol(pair: str) -> str:
    return SYMBOL_MAP.get(pair, pair)


def _frame(payload: str) -> str:
    return f"~m~{len(payload)}~m~{payload}"


def _session(prefix: str) -> str:
    return prefix + "".join(random.choices(string.ascii_lowercase, k=12))


def _split_frames(raw: str) -> list[str]:
    out = []
    pos = 0
    while pos < len(raw):
        m = _FRAME_RE.match(raw, pos)
        if not m:
            break
        n = int(m.group(1))
        start = m.end()
        out.append(raw[start:start + n])
        pos = start + n
    return out


def _parse_bars(block) -> dict[int, tuple]:
    """Accept {'s': [{'i': n, 'v': [t,o,h,l,c,v]}]} plus list/dict variants."""
    rows: dict[int, tuple] = {}
    if isinstance(block, dict) and "s" in block:
        items = block["s"]
        for r in items if isinstance(items, list) else []:
            v = r.get("v") if isinstance(r, dict) else r
            if not isinstance(v, (list, tuple)) or len(v) < 5:
                continue
            try:
                t = int(v[0])
                vol = float(v[5]) if len(v) > 5 else 0.0
                rows[t] = (float(v[1]), float(v[2]), float(v[3]), float(v[4]), vol)
            except (ValueError, TypeError, IndexError):
                continue
    elif isinstance(block, dict) and "t" in block:
        for i, t in enumerate(block["t"]):
            try:
                rows[int(t)] = (
                    float(block["o"][i]), float(block["h"][i]),
                    float(block["l"][i]), float(block["c"][i]),
                    float(block["v"][i]) if block.get("v") else 0.0,
                )
            except (ValueError, TypeError, KeyError, IndexError):
                continue
    return rows


async def _fetch_async(tv_symbol: str, interval_code: str, n_bars: int,
                       timeout: float = 20.0) -> pd.DataFrame:
    chart = _session("cs_")
    rows: dict[int, tuple] = {}
    completed = False
    pages = 0

    async with websockets.connect(
        TV_WS_URL, open_timeout=timeout, close_timeout=5,
        additional_headers={"Origin": "https://www.tradingview.com"},
    ) as ws:
        await ws.send(_frame(json.dumps({"m": "set_auth_token", "p": ["unauthorized_user_token"]})))
        await ws.send(_frame(json.dumps({"m": "chart_create_session", "p": [chart, ""]})))
        await ws.send(_frame(json.dumps({"m": "resolve_symbol", "p": [
            chart, "sds_sym_1",
            '={"adjustment":"splits","session":"extended","symbol":"%s"}' % tv_symbol,
        ]})))
        await ws.send(_frame(json.dumps({"m": "create_series", "p": [
            chart, "sds_1", "s1", "sds_sym_1", interval_code, min(n_bars, 5000), "",
        ]})))

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            for payload in _split_frames(raw):
                if payload.startswith("~h~"):
                    await ws.send(_frame(payload))
                    continue
                try:
                    msg = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict) or "m" not in msg:
                    continue
                method, params = msg["m"], msg.get("p", [])

                if method == "timescale_update":
                    for entry in params[1:] if len(params) > 1 else []:
                        if isinstance(entry, dict) and "sds_1" in entry:
                            rows.update(_parse_bars(entry["sds_1"]))
                elif method == "series_completed":
                    if len(rows) >= n_bars or pages >= _MAX_PAGES:
                        completed = True
                        break
                    pages += 1
                    await ws.send(_frame(json.dumps({
                        "m": "request_more_data",
                        "p": [chart, "sds_1", min(n_bars - len(rows), 5000)],
                    })))
                elif method in ("critical_error", "protocol_error"):
                    raise TvHistoryError(f"TradingView error: {params}")
            else:
                continue
            break

    if not rows:
        raise TvHistoryError(f"no bars returned for {tv_symbol} {interval_code}")

    records = sorted(rows.items())
    idx = pd.DatetimeIndex(
        [datetime.fromtimestamp(t, tz=timezone.utc) for t, _ in records],
        name="Datetime",
    )
    df = pd.DataFrame(
        [r for _, r in records],
        index=idx,
        columns=["Open", "High", "Low", "Close", "Volume"],
        dtype=float,
    ).dropna()
    return df


def fetch_tv_history(symbol: str, interval: str = "5m", lookback_days: int = 30) -> pd.DataFrame:
    """Fetch OHLCV candles from TradingView; returns a UTC tz-aware frame."""
    iv = INTERVALS.get(interval)
    if iv is None:
        raise TvHistoryError(f"unsupported interval: {interval}")
    iv_min = _INTERVAL_MINUTES.get(iv, 1440)

    tv_symbol = resolve_symbol(symbol)
    bars_per_trading_day = max(1, int(1440 / iv_min * 5 / 7)) if iv != "D" else 1
    n_bars = int(lookback_days * bars_per_trading_day) + 60

    df = asyncio.run(_fetch_async(tv_symbol, iv, n_bars))
    cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
    df = df[df.index >= cutoff]

    log.info(
        "TradingView %s %s: %d bars (%s -> %s)",
        tv_symbol, interval, len(df),
        df.index[0].strftime("%m-%d %H:%M"), df.index[-1].strftime("%m-%d %H:%M"),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = fetch_tv_history("GC=F", "5m", 30)
    print(d.tail(3))
