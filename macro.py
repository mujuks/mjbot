"""Market-context intelligence: dollar bias, volatility regime, session quality.

Implements the research-backed gold behaviors:
  1. Gold trades inversely to the US dollar (DXY proxy via TVC:DXY).
  2. Volatility regimes matter: compressed ATR precedes expansions.
  3. Session quality: hourly activity profile measured from this bot's own
     stored candles (see data/GC_F_5m.csv hourly range study).
  4. COMEX OpEx pinning window (3rd Wednesday monthly).
"""
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy import _atr

# Measured avg move-per-bar vs overall mean (100 = average), by UTC hour,
# from 7,000+ stored 5m candles of OANDA:XAUUSD.
_SESSION_INDEX = {
    0: 122, 1: 143, 2: 108, 3: 76, 4: 67, 5: 103, 6: 97, 7: 90,
    8: 91, 9: 81, 10: 73, 11: 93, 12: 144, 13: 172, 14: 152, 15: 125,
    16: 90, 17: 83, 18: 89, 19: 85, 20: 57, 21: 117, 22: 65, 23: 75,
}

_dxy_cache: dict = {"ts": 0.0, "line": None}
_DXY_TTL = 1800


def session_quality(hour_utc: int) -> tuple[str, int]:
    """-> (label, index) where label in PEAK / GOOD / SLOW."""
    idx = _SESSION_INDEX.get(hour_utc % 24, 100)
    label = "PEAK" if idx >= 120 else ("GOOD" if idx >= 85 else "SLOW")
    return label, idx


def dxy_snapshot() -> str | None:
    """Dollar-index direction -> tailwind/headwind note for gold. Cached 30min."""
    now = time.time()
    if _dxy_cache["line"] and now - _dxy_cache["ts"] < _DXY_TTL:
        return _dxy_cache["line"]
    try:
        from tv_history import fetch_tv_history
        df = fetch_tv_history("TVC:DXY", "1h", 6)
        close = float(df["Close"].iloc[-1])
        chg = close / float(df["Close"].iloc[-25]) - 1 if len(df) > 25 else None
        ema = float(df["Close"].tail(20).mean())
        arrow = "\u25b2" if close >= ema else "\u25bc"
        if chg is not None:
            pct = f"{chg * 100:+.2f}%"
            wind = "headwind" if chg > 0 else "tailwind"
            line = f"Dollar {arrow} {pct}/24h ({wind} for gold)"
        else:
            line = f"Dollar {arrow} ({'headwind' if close >= ema else 'tailwind'} for gold)"
        _dxy_cache.update(ts=now, line=line)
        return line
    except Exception as e:
        print(f"[macro] dxy failed: {e}", file=sys.stderr)
        return None


def volatility_regime(df) -> str | None:
    """ATR percentile over the trailing day(s) -> compressed/normal/expansion."""
    try:
        atr = _atr(df, 14)
        series = atr.dropna().iloc[-288:]  # ~24h of 5m bars
        if len(series) < 50:
            return None
        cur = float(series.iloc[-1])
        pct = (series < cur).mean() * 100
        label = "EXPANSION" if pct >= 75 else ("COMPRESSED" if pct <= 25 else "NORMAL")
        return f"Volatility {label} ({pct:.0f}th pct)"
    except Exception:
        return None


def opex_flag(now_utc: datetime) -> str | None:
    """COMEX options expire on the 3rd Wednesday; flag the pinning window."""
    y, m = now_utc.year, now_utc.month
    # first Wednesday
    d = datetime(y, m, 1, tzinfo=timezone.utc)
    offset = (2 - d.weekday()) % 7
    third_wed = d.day + offset + 14
    try:
        opex = datetime(y, m, third_wed, tzinfo=timezone.utc)
    except ValueError:
        return None
    days = (opex.date() - now_utc.date()).days
    if 0 <= days <= 3:
        return f"COMEX OpEx {'today' if days == 0 else f'in {days}d'} - expect pinning"
    return None


def build_context(df, now_utc: datetime) -> list[str]:
    """All context lines for alerts/console; skips whatever fails."""
    lines = []
    s_label, _ = session_quality(now_utc.hour)
    eat = (now_utc.hour + 3) % 24
    lines.append(f"Session: {s_label} ({now_utc:%H:%M} UTC / {eat:02d}:{now_utc:%M} EAT)")
    vol = volatility_regime(df)
    if vol:
        lines.append(vol)
    dx = dxy_snapshot()
    if dx:
        lines.append(dx)
    ox = opex_flag(now_utc)
    if ox:
        lines.append(ox)
    return lines
