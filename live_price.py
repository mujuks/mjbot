import json
import logging
import threading
import time
from datetime import datetime, timezone

import requests

log = logging.getLogger("live_price")

TV_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Content-Type": "application/json",
}

_price_cache = {
    "data": None,
    "timestamp": None,
    "lock": threading.Lock(),
}
_last_warning_ts = 0.0
_WARNING_COOLDOWN = 300


def _cache_price(data: dict):
    with _price_cache["lock"]:
        _price_cache["data"] = data
        _price_cache["timestamp"] = datetime.now(timezone.utc)


def _get_cached_price(max_age_sec: float = 60.0):
    with _price_cache["lock"]:
        data = _price_cache["data"]
        ts = _price_cache["timestamp"]
        if data and ts:
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age < max_age_sec:
                return data
    return None


def _retry_fetch(func, max_attempts: int = 3, base_delay: float = 1.0):
    last_err = None
    for attempt in range(max_attempts):
        try:
            result = func()
            if result is not None:
                return result
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    return None


def fetch_xauusd_from_tradingview():
    symbols = [
        "OANDA:XAUUSD",
        "TVC:GOLD",
        "FX_IDC:XAUUSD",
        "FOREXCOM:XAUUSD",
        "PEPPERSTONE:XAUUSD",
    ]
    payload = {
        "columns": ["close", "bid", "ask", "high", "low", "open", "description", "name",
                     "exchange", "volume", "prev_close_price_change"],
        "symbols": {"tickers": symbols},
        "options": {"lang": "en"},
    }
    try:
        r = requests.post(
            "https://scanner.tradingview.com/cfd/scan",
            json=payload, timeout=10, headers=TV_HEADERS,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.debug("TradingView scan attempt failed: %s", e)
        return None

    if not data.get("data"):
        return None

    feeds = []
    for item in data["data"]:
        try:
            d = item["d"]
            bid = d[1]
            ask = d[2]
            close = d[0]
            high = d[4]
            low = d[5]
            opn = d[6]
            exchange = d[8] if isinstance(d[8], str) else str(d[7]) if isinstance(d[7], str) else "UNKNOWN"
            if exchange in ("Gold", "", None):
                exchange = str(d[7]) if isinstance(d[7], str) and d[7] not in ("Gold", "") else "TradingView"
            volume = d[9] if len(d) > 9 else 0
            change_pct = d[10] if len(d) > 10 else 0

            if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)):
                continue
            if bid <= 1000 or ask <= 1000:
                continue

            mid = (bid + ask) / 2.0
            spread = ask - bid
            feeds.append({
                "bid": float(bid),
                "ask": float(ask),
                "mid": float(mid),
                "close": float(close) if isinstance(close, (int, float)) else float(mid),
                "high": float(high) if isinstance(high, (int, float)) else float(mid),
                "low": float(low) if isinstance(low, (int, float)) else float(mid),
                "open": float(opn) if isinstance(opn, (int, float)) else float(mid),
                "spread": float(spread),
                "exchange": exchange,
                "volume": float(volume) if isinstance(volume, (int, float)) else 0,
                "change_pct": float(change_pct) if isinstance(change_pct, (int, float)) else 0,
            })
        except (ValueError, TypeError, IndexError):
            continue

    if not feeds:
        return None

    feeds.sort(key=lambda f: 0 if f["exchange"] == "OANDA" else 1)
    best = feeds[0]

    best_spread = best["spread"]
    for f in feeds:
        if f["spread"] < best_spread:
            best = f
            best_spread = f["spread"]

    all_mids = [f["mid"] for f in feeds]
    avg_mid = sum(all_mids) / len(all_mids)

    ts = datetime.now(timezone.utc)
    result = {
        "bid": best["bid"],
        "ask": best["ask"],
        "mid": best["mid"],
        "avg_mid": float(avg_mid),
        "close": best["close"],
        "high": best["high"],
        "low": best["low"],
        "open": best["open"],
        "spread": best["spread"],
        "exchange": best["exchange"],
        "volume": best["volume"],
        "change_pct": best["change_pct"],
        "feeds_count": len(feeds),
        "timestamp": ts,
        "source": f"TradingView {best['exchange']}:XAUUSD ({len(feeds)} feeds)",
    }

    log.info(
        "XAUUSD real-time: bid=%.2f ask=%.2f mid=%.2f spread=%.2f change=%.2f%% (%s, %d feeds)",
        best["bid"], best["ask"], best["mid"], best["spread"],
        best["change_pct"], best["exchange"], len(feeds),
    )
    return result


def fetch_gold_ohlcv_from_tradingview():
    payload = {
        "columns": ["close", "bid", "ask", "high", "low", "open", "volume", "name", "exchange"],
        "symbols": {"tickers": ["OANDA:XAUUSD"]},
        "options": {"lang": "en"},
    }
    try:
        r = requests.post(
            "https://scanner.tradingview.com/cfd/scan",
            json=payload, timeout=10, headers=TV_HEADERS,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("data"):
            d = data["data"][0]["d"]
            return {
                "open": float(d[6]) if d[6] else 0,
                "high": float(d[4]) if d[4] else 0,
                "low": float(d[5]) if d[5] else 0,
                "close": float(d[0]) if d[0] else 0,
                "bid": float(d[1]) if d[1] else 0,
                "ask": float(d[2]) if d[2] else 0,
                "volume": float(d[7]) if len(d) > 7 and d[7] else 0,
            }
    except Exception as e:
        log.debug("TradingView OHLCV fetch failed: %s", e)
    return None


def get_live_price():
    data = fetch_xauusd_from_tradingview()
    if data:
        _cache_price(data)
        return data["mid"], data["timestamp"], data["source"]
    cached = _get_cached_price()
    if cached:
        return cached["mid"], cached["timestamp"], cached["source"] + " (cached)"
    return None, None, None


def fetch_gold_spot_price():
    global _last_warning_ts
    data = fetch_xauusd_from_tradingview()
    if data and data["mid"] > 1000:
        _cache_price(data)
        return data["mid"], data["timestamp"], data["source"]

    cached = _get_cached_price(max_age_sec=120)
    if cached and cached["mid"] > 1000:
        return cached["mid"], cached["timestamp"], cached["source"] + " (cached)"

    now = time.time()
    if now - _last_warning_ts > _WARNING_COOLDOWN:
        log.warning("All live price sources failed - using cache if available")
        _last_warning_ts = now

    with _price_cache["lock"]:
        stale = _price_cache["data"]
        if stale:
            return stale["mid"], stale["timestamp"], stale["source"] + " (stale)"
    return None, None, None


def validate_data_freshness(df, max_age_minutes=20):
    if df is None or df.empty:
        return False, "No data"
    last_ts = df.index[-1]
    if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is None:
        from pandas import Timestamp
        last_ts = Timestamp(last_ts).tz_localize("UTC")
    now = datetime.now(timezone.utc)
    age = (now - last_ts).total_seconds() / 60
    if age > max_age_minutes:
        return False, f"Data is {age:.0f}min old (max {max_age_minutes}min)"
    return True, f"Data age: {age:.0f}min"


def calibrate_to_spot(gc_price, live_spot_price):
    if live_spot_price and gc_price and gc_price > 0:
        offset = live_spot_price - gc_price
        log.info("Spot calibration: offset=%.2f (GC=%.2f, spot=%.2f)", offset, gc_price, live_spot_price)
        return offset
    return 0.0
