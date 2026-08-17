import json
import logging
from datetime import datetime, timezone

import requests

log = logging.getLogger("live_price")

TV_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Content-Type": "application/json",
}


def fetch_xauusd_from_tradingview():
    payload = {
        "columns": ["close", "bid", "ask", "high", "low", "open", "description", "name", "exchange"],
        "symbols": {"tickers": ["OANDA:XAUUSD", "TVC:GOLD", "FX_IDC:XAUUSD"]},
        "options": {"lang": "en"},
    }
    r = requests.post(
        "https://scanner.tradingview.com/cfd/scan",
        json=payload, timeout=15, headers=TV_HEADERS,
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("data"):
        return None, None, None

    best = None
    for item in data["data"]:
        d = item["d"]
        bid = d[1]
        ask = d[2]
        close = d[0]
        exchange = d[6]
        name = d[7]
        if bid and ask and bid > 1000:
            mid = (bid + ask) / 2.0
            spread = ask - bid
            if best is None or exchange == "OANDA":
                best = {
                    "bid": float(bid),
                    "ask": float(ask),
                    "mid": float(mid),
                    "close": float(close),
                    "spread": float(spread),
                    "exchange": exchange,
                    "name": name,
                }
    if best:
        ts = datetime.now(timezone.utc)
        log.info(
            "TradingView XAUUSD: bid=%.2f ask=%.2f mid=%.2f spread=%.2f (%s)",
            best["bid"], best["ask"], best["mid"], best["spread"], best["exchange"],
        )
        return best["mid"], ts, f"TradingView {best['exchange']}:XAUUSD"
    return None, None, None


def fetch_gold_spot_price():
    try:
        price, ts, source = fetch_xauusd_from_tradingview()
        if price and price > 1000:
            return price, ts, source
    except Exception as e:
        log.warning("TradingView fetch failed: %s", e)
    log.warning("All live price sources failed")
    return None, None, None


def validate_data_freshness(df, max_age_minutes=30):
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
