"""MJBot Scalp Engine - Institutional Order Flow Scalping.

Focuses on:
- Market Structure (BOS/CHoCH at key levels)
- Strong S/D Zones (score 60+)
- Market Movers (volume + displacement)
- Killzones (London/NY opens)

All 4 must align for a scalp entry.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ============================================================================
# KILLZONE DETECTION
# ============================================================================

def in_killzone(now_utc: datetime) -> dict:
    """ICT Killzone detection for scalping.
    Returns which killzone we're in and quality score."""
    hour = now_utc.hour
    minute = now_utc.minute

    # London Open Killzone: 07:00-10:00 UTC
    if 7 <= hour < 10:
        progress = (hour - 7) * 60 + minute
        quality = "PEAK" if 30 <= progress <= 120 else "GOOD"
        return {
            "in_killzone": True,
            "zone": "london_open",
            "quality": quality,
            "score": 2 if quality == "PEAK" else 1,
        }

    # NY Open Killzone: 13:00-16:00 UTC
    if 13 <= hour < 16:
        progress = (hour - 13) * 60 + minute
        quality = "PEAK" if 30 <= progress <= 120 else "GOOD"
        return {
            "in_killzone": True,
            "zone": "ny_open",
            "quality": quality,
            "score": 2 if quality == "PEAK" else 1,
        }

    # London-NY Overlap: 14:00-16:00 UTC (best liquidity)
    if 14 <= hour < 16:
        return {
            "in_killzone": True,
            "zone": "lny_overlap",
            "quality": "PREMIUM",
            "score": 3,
        }

    return {
        "in_killzone": False,
        "zone": None,
        "quality": "OFF",
        "score": 0,
    }


# ============================================================================
# MARKET STRUCTURE CHECK
# ============================================================================

def _detect_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> dict:
    """Detect swing highs and lows."""
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    swing_highs = []
    swing_lows = []

    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        window_l = lows[i - left:i + right + 1]
        if highs[i] == np.max(window_h):
            swing_highs.append({"index": i, "price": float(highs[i])})
        if lows[i] == np.min(window_l):
            swing_lows.append({"index": i, "price": float(lows[i])})

    return {"highs": swing_highs[-5:], "lows": swing_lows[-5:]}


def _check_structure(df: pd.DataFrame, direction: str) -> dict:
    """Check market structure for BOS/CHoCH in direction."""
    if df is None or df.empty or len(df) < 20:
        return {"aligned": False, "event": None}

    swings = _detect_swings(df)
    close = df["Close"].values
    last_close = close[-1]
    prev_close = close[-2]

    sh_list = swings["highs"]
    sl_list = swings["lows"]

    # Check for BOS/CHoCH
    if direction == "bearish":
        # Look for break below recent swing low
        for sl in reversed(sl_list):
            if prev_close >= sl["price"] and last_close < sl["price"]:
                return {
                    "aligned": True,
                    "event": "BOS_BEAR",
                    "level": sl["price"],
                    "reason": f"BOS bearish @ {sl['price']:.2f}",
                }
        # Look for CHoCH (break of structure in opposite direction)
        if len(sh_list) >= 2:
            last_sh = sh_list[-1]["price"]
            if last_close < last_sh and prev_close >= last_sh:
                return {
                    "aligned": True,
                    "event": "CHoCH_BEAR",
                    "level": last_sh,
                    "reason": f"CHoCH bearish @ {last_sh:.2f}",
                }
        # Check if price is below EMA (trend alignment)
        ema20 = float(_ema(df["Close"], 20).iloc[-1])
        if last_close < ema20:
            return {"aligned": True, "event": "TREND_BEAR", "reason": "below EMA20"}

    elif direction == "bullish":
        # Look for break above recent swing high
        for sh in reversed(sh_list):
            if prev_close <= sh["price"] and last_close > sh["price"]:
                return {
                    "aligned": True,
                    "event": "BOS_BULL",
                    "level": sh["price"],
                    "reason": f"BOS bullish @ {sh['price']:.2f}",
                }
        # Look for CHoCH
        if len(sl_list) >= 2:
            last_sl = sl_list[-1]["price"]
            if last_close > last_sl and prev_close <= last_sl:
                return {
                    "aligned": True,
                    "event": "CHoCH_BULL",
                    "level": last_sl,
                    "reason": f"CHoCH bullish @ {last_sl:.2f}",
                }
        # Check if price is above EMA
        ema20 = float(_ema(df["Close"], 20).iloc[-1])
        if last_close > ema20:
            return {"aligned": True, "event": "TREND_BULL", "reason": "above EMA20"}

    return {"aligned": False, "event": None}


# ============================================================================
# STRONG ZONE CHECK
# ============================================================================

def _check_near_zone(df: pd.DataFrame, price: float, direction: str, atr_val: float, cfg: dict) -> dict:
    """Check if price is near a strong S/D zone (score 70+)."""
    from zones import detect_zones

    zones = detect_zones(df, cfg)

    min_score = cfg.get("scalp_engine", {}).get("min_zone_score", 70)

    if direction == "bearish":
        for zone in zones.get("supply", []):
            score = zone.get("strength_score", 0)
            if score < min_score:
                continue
            # Require strong displacement
            if zone.get("displacement_atr", 0) < cfg.get("scalp_engine", {}).get("min_displacement_atr", 1.0):
                continue
            # Require strong body ratio
            if zone.get("body_ratio", 0) < cfg.get("scalp_engine", {}).get("min_body_ratio", 0.60):
                continue
            z_lo = zone["bottom"]
            z_hi = zone["top"]
            # Price should be inside or near the zone
            if z_lo - 0.3 * atr_val <= price <= z_hi + 0.3 * atr_val:
                return {
                    "aligned": True,
                    "zone_score": score,
                    "zone_top": z_hi,
                    "zone_bottom": z_lo,
                    "displacement_atr": zone.get("displacement_atr", 0),
                    "reason": f"supply zone (score {score}, {zone.get('displacement_atr', 0):.1f} ATR displacement)",
                }

    elif direction == "bullish":
        for zone in zones.get("demand", []):
            score = zone.get("strength_score", 0)
            if score < min_score:
                continue
            if zone.get("displacement_atr", 0) < cfg.get("scalp_engine", {}).get("min_displacement_atr", 1.0):
                continue
            if zone.get("body_ratio", 0) < cfg.get("scalp_engine", {}).get("min_body_ratio", 0.60):
                continue
            z_lo = zone["bottom"]
            z_hi = zone["top"]
            if z_lo - 0.3 * atr_val <= price <= z_hi + 0.3 * atr_val:
                return {
                    "aligned": True,
                    "zone_score": score,
                    "zone_top": z_hi,
                    "zone_bottom": z_lo,
                    "displacement_atr": zone.get("displacement_atr", 0),
                    "reason": f"demand zone (score {score}, {zone.get('displacement_atr', 0):.1f} ATR displacement)",
                }

    return {"aligned": False, "zone_score": 0}


# ============================================================================
# MARKET MOVER CHECK
# ============================================================================

def _check_market_mover(df: pd.DataFrame, direction: str) -> dict:
    """Check for market mover: volume spike + displacement candle."""
    if df is None or df.empty or len(df) < 20:
        return {"aligned": False}

    c = df["Close"].values
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values

    atr_val = float(_atr(df, 14).iloc[-1])
    if atr_val <= 0:
        return {"aligned": False}

    # Volume check
    if "Volume" in df.columns:
        vol = df["Volume"].values
        avg_vol = np.mean(vol[-20:])
        curr_vol = vol[-1]
        vol_spike = curr_vol > avg_vol * 1.3 if avg_vol > 0 else False
    else:
        vol_spike = True  # No volume data, pass check

    # Displacement check
    curr_body = abs(c[-1] - o[-1])
    curr_range = h[-1] - l[-1] if h[-1] > l[-1] else 0.001
    body_ratio = curr_body / curr_range

    if direction == "bearish":
        # Strong bearish candle
        is_displacement = (c[-1] < o[-1] and
                          body_ratio > 0.60 and
                          curr_body > 0.5 * atr_val and
                          c[-1] < c[-2])
    else:
        # Strong bullish candle
        is_displacement = (c[-1] > o[-1] and
                          body_ratio > 0.60 and
                          curr_body > 0.5 * atr_val and
                          c[-1] > c[-2])

    if vol_spike and is_displacement:
        return {
            "aligned": True,
            "body_ratio": round(body_ratio, 3),
            "displacement_atr": round(curr_body / atr_val, 2),
            "reason": f"market mover ({body_ratio:.0%} body, {curr_body/atr_val:.1f} ATR)",
        }

    return {"aligned": False}


# ============================================================================
# MAIN SCALP SIGNAL
# ============================================================================

def scalp_entry_signal(dfs: dict, cfg: dict, now_utc: datetime = None) -> dict:
    """Generate institutional scalp signal.
    
    Requires ALL 4 factors:
    1. Killzone active
    2. Market structure aligned (BOS/CHoCH)
    3. Price at strong S/D zone (score 70+, displacement 1.0+ ATR, body 60%+)
    4. Market mover confirmation (volume + displacement)
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    scalp_cfg = cfg.get("scalp_engine", {})
    if not scalp_cfg.get("enabled", True):
        return {"signal": "NONE", "reason": "scalp engine disabled"}

    df_5m = dfs.get("5m")
    if df_5m is None or df_5m.empty or len(df_5m) < 30:
        return {"signal": "NONE", "reason": "insufficient 5m data"}

    price = float(df_5m["Close"].iloc[-1])
    atr_val = float(_atr(df_5m, 14).iloc[-1])

    # ============================================================
    # FACTOR 1: KILLZONE CHECK
    # ============================================================
    kz = in_killzone(now_utc)
    if not kz["in_killzone"]:
        return {"signal": "NONE", "reason": f"not in killzone (hour {now_utc.hour} UTC)"}

    # ============================================================
    # FACTOR 2: MARKET STRUCTURE CHECK
    # ============================================================
    # Check structure on 15m for better swing detection
    df_15m = dfs.get("15m", df_5m)
    struct_bear = _check_structure(df_15m, "bearish")
    struct_bull = _check_structure(df_15m, "bullish")

    # ============================================================
    # FACTOR 3: STRONG ZONE CHECK
    # ============================================================
    zone_bear = _check_near_zone(df_5m, price, "bearish", atr_val, cfg)
    zone_bull = _check_near_zone(df_5m, price, "bullish", atr_val, cfg)

    # ============================================================
    # FACTOR 4: MARKET MOVER CHECK
    # ============================================================
    mover_bear = _check_market_mover(df_5m, "bearish")
    mover_bull = _check_market_mover(df_5m, "bullish")

    # ============================================================
    # DETERMINE DIRECTION (need all 4 aligned)
    # ============================================================
    sell_score = sum([
        struct_bear["aligned"],
        zone_bear["aligned"],
        mover_bear["aligned"],
        kz["score"] > 0,
    ])

    buy_score = sum([
        struct_bull["aligned"],
        zone_bull["aligned"],
        mover_bull["aligned"],
        kz["score"] > 0,
    ])

    # Need ALL 4 aligned
    if sell_score >= 4:
        direction = "bearish"
        structure = struct_bear
        zone = zone_bear
        mover = mover_bear
    elif buy_score >= 4:
        direction = "bullish"
        structure = struct_bull
        zone = zone_bull
        mover = mover_bull
    else:
        reasons = []
        if not kz["in_killzone"]:
            reasons.append("no killzone")
        if not struct_bear["aligned"] and not struct_bull["aligned"]:
            reasons.append("no structure alignment")
        if not zone_bear["aligned"] and not zone_bull["aligned"]:
            reasons.append("no strong zone")
        if not mover_bear["aligned"] and not mover_bull["aligned"]:
            reasons.append("no market mover")
        return {"signal": "NONE", "reason": f"{sell_score}/4 sell or {buy_score}/4 buy - need all 4: {'; '.join(reasons)}"}

    # ============================================================
    # CALCULATE ENTRY, SL, TP
    # ============================================================
    sl_buffer = scalp_cfg.get("sl_atr", 0.5) * atr_val
    min_rr = scalp_cfg.get("min_rr", 1.5)

    if direction == "bearish":
        entry = price
        # SL above zone or structure level
        sl = max(zone.get("zone_top", price) + 0.2 * atr_val,
                 structure.get("level", price) + 0.2 * atr_val,
                 price + sl_buffer)
        tp1 = price - 1.5 * (sl - price)  # 1.5R target
        tp2 = price - 2.0 * (sl - price)  # 2R target

        risk = sl - entry
        reward = entry - tp1
    else:
        entry = price
        sl = min(zone.get("zone_bottom", price) - 0.2 * atr_val,
                 structure.get("level", price) - 0.2 * atr_val,
                 price - sl_buffer)
        tp1 = price + 1.5 * (price - sl)
        tp2 = price + 2.0 * (price - sl)

        risk = entry - sl
        reward = tp1 - entry

    rr = reward / risk if risk > 0 else 0

    if rr < min_rr:
        return {"signal": "NONE", "reason": f"R:R too low ({rr:.1f} < {min_rr})"}

    # ============================================================
    # BUILD SIGNAL
    # ============================================================
    signal_type = "SCALP_SELL" if direction == "bearish" else "SCALP_BUY"

    return {
        "signal": signal_type,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "rr": round(rr, 2),
        "confirmations": 4,
        "reason": (f"4-FACTOR: {kz['zone']}({kz['quality']}) + "
                  f"{structure.get('reason', '?')} + "
                  f"{zone.get('reason', '?')} + "
                  f"{mover.get('reason', '?')}"),
        "killzone": kz,
        "structure": structure,
        "zone": zone,
        "mover": mover,
    }


def format_scalp_alert(signal: dict, pair: str) -> str:
    """Format scalp signal for Telegram."""
    if signal["signal"] == "NONE":
        return ""

    side = "SELL" if "SELL" in signal["signal"] else "BUY"
    kz = signal.get("killzone", {})
    struct = signal.get("structure", {})
    zone = signal.get("zone", {})
    mover = signal.get("mover", {})

    lines = [
        f"{'🔴' if side == 'SELL' else '🟢'} {pair} {signal['signal']}",
        f"4-FACTOR ALIGNMENT ✓",
        f"",
        f"Entry: {signal['entry']:.2f}",
        f"SL: {signal['sl']:.2f}",
        f"TP1: {signal['tp1']:.2f} | TP2: {signal['tp2']:.2f}",
        f"R:R 1:{signal['rr']:.1f}",
        f"",
        f"⏰ Killzone: {kz.get('zone', '?')} ({kz.get('quality', '?')})",
        f"📊 Structure: {struct.get('reason', '?')}",
        f"🎯 Zone: {zone.get('reason', '?')} (score {zone.get('zone_score', 0)})",
        f"💥 Mover: {mover.get('reason', '?')}",
    ]
    return "\n".join(lines)
