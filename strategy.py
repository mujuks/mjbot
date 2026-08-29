import numpy as np
import pandas as pd

from liquidity import (detect_liquidity_sweep, detect_liquidity_pools,
                       compute_dynamic_tp, compute_multi_tp, detect_fvg, detect_liquidity_voids,
                       detect_volume_profile)
from market_structure import analyze_structure, compute_premium_discount
from order_blocks import detect_order_blocks, find_nearest_ob
from zones import detect_zones, _atr
from mitigation_blocks import detect_mitigation_blocks, find_nearest_mitigation
from patterns import detect_sweep_retest
from breaker_blocks import detect_breaker_blocks, find_nearest_breaker
from rejection_blocks import detect_rejection_blocks, find_nearest_rejection
from session_timing import session_timing_score, validate_displacement


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int, slow: int, signal: int):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def compute_bias(df: pd.DataFrame, cfg: dict) -> int:
    if df is None or df.empty or len(df) < cfg.get("ma_slow", 30):
        return 0
    close = df["Close"]
    fast = ema(close, cfg["ma_fast"])
    slow = ema(close, cfg["ma_slow"])
    if fast.iloc[-1] > slow.iloc[-1]:
        return 1
    if fast.iloc[-1] < slow.iloc[-1]:
        return -1
    return 0


def bias_series(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    if df is None or df.empty:
        return np.zeros(0, int)
    close = df["Close"]
    f = ema(close, cfg["ma_fast"]).to_numpy(float)
    s = ema(close, cfg["ma_slow"]).to_numpy(float)
    b = np.zeros(len(close), int)
    b[f > s] = 1
    b[f < s] = -1
    return b


def _utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is not None:
        return index.tz_convert("UTC")
    return index.tz_localize("UTC")


def _parse_hhmm(value: str):
    h, m = value.split(":")
    return int(h) * 60 + int(m)


def _minutes_of(ts) -> int:
    return ts.hour * 60 + ts.minute


def trading_allowed(now_utc, cfg: dict) -> bool:
    s = cfg.get("sessions", {})
    if not s.get("enabled", True):
        return True
    if now_utc.weekday() not in s.get("days", [0, 1, 2, 3, 4]):
        return False
    start = _parse_hhmm(s.get("utc_start", "07:00"))
    end = _parse_hhmm(s.get("utc_end", "21:00"))
    t = _minutes_of(now_utc)
    in_window = (t >= start or t < end) if end < start else start <= t < end
    if not in_window:
        return False
    for window in cfg.get("blackout_windows", []):
        start_dt = pd.Timestamp(window["start"]).tz_localize("UTC")
        end_dt = pd.Timestamp(window["end"]).tz_localize("UTC")
        if start_dt <= now_utc < end_dt:
            return False
    return True


def trading_mask(index: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    n = len(index)
    s = cfg.get("sessions", {})
    if not s.get("enabled", True):
        return np.ones(n, bool)
    days = s.get("days", [0, 1, 2, 3, 4])
    start = _parse_hhmm(s.get("utc_start", "07:00"))
    end = _parse_hhmm(s.get("utc_end", "21:00"))
    blackouts = [
        (pd.Timestamp(w["start"]).tz_localize("UTC"), pd.Timestamp(w["end"]).tz_localize("UTC"))
        for w in cfg.get("blackout_windows", [])
    ]
    idx_utc = _utc_index(index)
    out = np.zeros(n, bool)
    for i, ts in enumerate(idx_utc):
        if ts.weekday() not in days:
            continue
        t = _minutes_of(ts)
        if not ((t >= start or t < end) if end < start else start <= t < end):
            continue
        if any(bs <= ts < be for bs, be in blackouts):
            continue
        out[i] = True
    return out


def _momentum(close: pd.Series, atr_series: pd.Series, cfg: dict) -> int:
    fast = ema(close, cfg["ma_fast"])
    slow = ema(close, cfg["ma_slow"])
    trend_up = fast.iloc[-1] > slow.iloc[-1]
    trend_down = fast.iloc[-1] < slow.iloc[-1]

    last = close.iloc[-1]
    body = abs(last - close.iloc[-2])
    strong_up = close.iloc[-1] > close.iloc[-2] and body > cfg.get("market_mover_atr", 0.5) * atr_series.iloc[-1]
    strong_dn = close.iloc[-1] < close.iloc[-2] and body > cfg.get("market_mover_atr", 0.5) * atr_series.iloc[-1]

    score = 0
    score += 1 if trend_up else -1
    score += 1 if strong_up else (-1 if strong_dn else 0)
    return score


def _volume_weighted_momentum(close: pd.Series, open_: pd.Series,
                              volume: pd.Series, cfg: dict) -> float:
    """Volume-weighted momentum score: direction × volume per candle.
    Returns value between -1.0 and +1.0."""
    lookback = cfg.get("vw_mom_lookback", 10)
    n = min(lookback, len(close))
    if n < 2 or volume.iloc[-n:].sum() == 0:
        return 0.0
    c = close.iloc[-n:].to_numpy(float)
    o = open_.iloc[-n:].to_numpy(float)
    v = volume.iloc[-n:].to_numpy(float)
    rets = np.where(o > 0, (c - o) / o, 0.0)
    weighted = np.sum(v * rets)
    total_vol = np.sum(v)
    return float(np.clip(weighted / total_vol * 100 if total_vol else 0, -1.0, 1.0))


def _vwap_bias(close: pd.Series, high: pd.Series, low: pd.Series,
               volume: pd.Series, cfg: dict) -> int:
    """Session VWAP directional bias. +1 = price above VWAP, -1 = below."""
    if close.empty or volume.empty:
        return 0
    lookback = cfg.get("vwap_lookback", 78)
    n = min(lookback, len(close))
    c = close.iloc[-n:].to_numpy(float)
    h = high.iloc[-n:].to_numpy(float)
    l = low.iloc[-n:].to_numpy(float)
    v = volume.iloc[-n:].to_numpy(float)
    typical = (h + l + c) / 3.0
    total_vol = np.sum(v)
    if total_vol <= 0:
        return 0
    vwap = np.sum(typical * v) / total_vol
    if c[-1] > vwap:
        return 1
    elif c[-1] < vwap:
        return -1
    return 0


def _volume_climax(volume: pd.Series, cfg: dict) -> bool:
    """Detect volume climax: current volume > climax_mult × average."""
    if volume.empty or len(volume) < 5:
        return False
    period = cfg.get("volume_period", 14)
    climax_mult = cfg.get("climax_mult", 3.0)
    avg = volume.iloc[-min(period, len(volume)):].mean()
    if avg <= 0:
        return False
    return bool(volume.iloc[-1] > climax_mult * avg)


def _find_nearest_zone(zones: list, price: float, atr_val: float) -> dict | None:
    best = None
    for zone in zones:
        if zone.get("mitigated", False):
            continue
        distance = zone["bottom"] - price if zone["type"] == "demand" else price - zone["top"]
        if distance <= 2.0 * atr_val and distance >= -0.5 * atr_val:
            if best is None or abs(distance) < abs(best["_distance"]):
                zone["_distance"] = distance
                best = zone
    if best is not None:
        best.pop("_distance", None)
    return best


def _fib_ote(df: pd.DataFrame, atr_val: float, lookback: int = 90) -> dict | None:
    """Fibonacci retracement of the latest displacement leg (ICT OTE model).

    Measures the most recent confirmed swing pair (pivot low -> pivot high for
    an up leg, or pivot high -> pivot low for a down leg) and computes how far
    price has pulled back into it. The professional entry band is 62-79%
    (the OTE / golden pocket region).

    Returns None when no qualifying leg exists (range < 1.5 ATR), else:
        direction ('bull'|'bear'), level_low, level_high, retrace (0-1),
        in_ote (bool)
    """
    n = len(df)
    if n < 40 or not atr_val or atr_val != atr_val:
        return None
    hi, lo, close_col = df["High"], df["Low"], df["Close"]
    close = float(close_col.iloc[-1])
    L = R = 2

    raw_pivots: list[tuple[int, str, float]] = []
    start = max(L, n - lookback)
    for i in range(start, n - R):
        win_hi = hi.iloc[i - L:i + R + 1]
        win_lo = lo.iloc[i - L:i + R + 1]
        if hi.iloc[i] == win_hi.max():
            raw_pivots.append((i, "H", float(hi.iloc[i])))
        elif lo.iloc[i] == win_lo.min():
            raw_pivots.append((i, "L", float(lo.iloc[i])))

    # collapse consecutive same-type pivots, keeping the extreme price
    pivots: list[tuple[int, str, float]] = []
    for idx, kind, px in raw_pivots:
        if pivots and pivots[-1][1] == kind:
            prev = pivots[-1]
            kept = px if ((kind == "H") == (px >= prev[2])) else prev[2]
            pivots[-1] = (prev[0], kind, kept)
        else:
            pivots.append((idx, kind, px))
    if len(pivots) < 2:
        return None

    (_, t1, v1), (_, t2, v2) = pivots[-2], pivots[-1]
    if t1 == "L" and t2 == "H" and v2 > v1:
        direction, leg_lo, leg_hi = "bull", v1, v2
    elif t1 == "H" and t2 == "L" and v2 < v1:
        direction, leg_lo, leg_hi = "bear", v2, v1
    else:
        return None

    if leg_hi - leg_lo < 1.5 * atr_val:
        return None

    rng = leg_hi - leg_lo
    retrace = (leg_hi - close) / rng if direction == "bull" else (close - leg_lo) / rng
    retrace = max(0.0, min(1.5, retrace))
    return {
        "direction": direction,
        "level_low": leg_lo,
        "level_high": leg_hi,
        "retrace": round(retrace, 3),
        "in_ote": 0.62 <= retrace <= 0.79,
    }


def _crt_analysis(df: pd.DataFrame, atr_val: float) -> dict:
    """Candle Range Theory analysis.
    Detects: inside bars, outside bars, breakout direction, wick rejection at level."""
    if df is None or df.empty or len(df) < 5:
        return {"inside_bar": False, "breakout": None, "wick_rejection": None, "crt_score": 0}

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)

    curr_h, curr_l, curr_c, curr_o = h[-1], l[-1], c[-1], o[-1]
    prev_h, prev_l = h[-2], l[-2]
    prev2_h, prev2_l = h[-3], l[-3]

    curr_range = curr_h - curr_l
    prev_range = prev_h - prev_l

    result = {"inside_bar": False, "breakout": None, "wick_rejection": None, "crt_score": 0}

    # Inside bar detection: current range < previous range
    is_inside = curr_range < prev_range and curr_h <= prev_h and curr_l >= prev_l
    result["inside_bar"] = is_inside

    # Outside bar: current range engulfs previous
    is_outside = curr_h > prev_h and curr_l < prev_l
    result["outside_bar"] = is_outside

    prev2_range = prev2_h - prev2_l

    # Breakout from inside bar: if prev bar was inside, current breaks it
    prev_inside = prev_range < prev2_range if prev2_range > 0 else False
    if prev_inside:
        if curr_c > prev_h:
            result["breakout"] = "bullish"
            result["crt_score"] += 1
        elif curr_c < prev_l:
            result["breakout"] = "bearish"
            result["crt_score"] += 1

    # Expansion bar: strong directional move with large range
    if curr_range > 1.5 * atr_val:
        body = abs(curr_c - curr_o)
        if body > 0.6 * curr_range:
            if curr_c > curr_o:
                result["expansion"] = "bullish"
            else:
                result["expansion"] = "bearish"

    # Wick rejection: long wick showing institutional defense
    body = abs(curr_c - curr_o)
    upper_wick = curr_h - max(curr_o, curr_c)
    lower_wick = min(curr_o, curr_c) - curr_l

    if body > 0 and atr_val > 0:
        # Bullish rejection: long lower wick at support
        if lower_wick > 2.0 * body and lower_wick > 0.5 * atr_val:
            result["wick_rejection"] = "bullish"
            result["crt_score"] += 1
        # Bearish rejection: long upper wick at resistance
        elif upper_wick > 2.0 * body and upper_wick > 0.5 * atr_val:
            result["wick_rejection"] = "bearish"
            result["crt_score"] += 1

    return result


def _elliott_oabc(df: pd.DataFrame, structure: dict, atr_val: float) -> dict:
    """Elliott Wave OABC zone detection.
    O = Origin (swing start), A = Auxiliary (impulse end), B = Breakout (pullback), C = Confirmation (breakout retest).
    Returns zones where price may react institutionally."""
    if df is None or df.empty or len(df) < 20:
        return {"zones": [], "nearest_zone": None, "direction": None}

    swings_high = structure.get("swing_highs", [])
    swings_low = structure.get("swing_lows", [])
    price = float(df["Close"].iloc[-1])

    if len(swings_high) < 2 or len(swings_low) < 2:
        return {"zones": [], "nearest_zone": None, "direction": None}

    zones = []

    # Find recent swing pairs to build OABC
    all_swings = []
    for s in swings_high:
        all_swings.append({"type": "H", "price": s["price"], "index": s["index"]})
    for s in swings_low:
        all_swings.append({"type": "L", "price": s["price"], "index": s["index"]})
    all_swings.sort(key=lambda x: x["index"])

    if len(all_swings) < 4:
        return {"zones": [], "nearest_zone": None, "direction": None}

    # Look for impulse legs (L->H for bullish, H->L for bearish)
    for i in range(len(all_swings) - 3):
        s1, s2, s3, s4 = all_swings[i], all_swings[i+1], all_swings[i+2], all_swings[i+3]

        # Bullish OABC: L(O) -> H(A) -> L(B) -> breakout above A -> retest(C)
        if s1["type"] == "L" and s2["type"] == "H" and s3["type"] == "L":
            if s2["price"] > s1["price"] and s3["price"] > s1["price"]:
                o_price = s1["price"]
                a_price = s2["price"]
                b_price = s3["price"]
                leg_size = a_price - o_price

                if leg_size >= 1.5 * atr_val:
                    # O zone: origin area
                    zones.append({
                        "type": "O", "direction": "bullish",
                        "level": o_price, "range": (o_price - 0.3 * atr_val, o_price + 0.3 * atr_val),
                        "strength": 2
                    })
                    # A zone: impulse end (becomes support after breakout)
                    zones.append({
                        "type": "A", "direction": "bullish",
                        "level": a_price, "range": (a_price - 0.2 * atr_val, a_price + 0.2 * atr_val),
                        "strength": 1
                    })
                    # B zone: pullback level
                    b_retrace = (a_price - b_price) / leg_size if leg_size > 0 else 0
                    if 0.38 <= b_retrace <= 0.79:
                        zones.append({
                            "type": "B", "direction": "bullish",
                            "level": b_price, "range": (b_price - 0.2 * atr_val, b_price + 0.2 * atr_val),
                            "strength": 3  # B retest is the best entry
                        })
                    # C zone: confirmation (breakout retest of A)
                    zones.append({
                        "type": "C", "direction": "bullish",
                        "level": a_price, "range": (a_price - 0.15 * atr_val, a_price + 0.15 * atr_val),
                        "strength": 2
                    })

        # Bearish OABC: H(O) -> L(A) -> H(B) -> breakout below A -> retest(C)
        elif s1["type"] == "H" and s2["type"] == "L" and s3["type"] == "H":
            if s2["price"] < s1["price"] and s3["price"] < s1["price"]:
                o_price = s1["price"]
                a_price = s2["price"]
                b_price = s3["price"]
                leg_size = o_price - a_price

                if leg_size >= 1.5 * atr_val:
                    zones.append({
                        "type": "O", "direction": "bearish",
                        "level": o_price, "range": (o_price - 0.3 * atr_val, o_price + 0.3 * atr_val),
                        "strength": 2
                    })
                    zones.append({
                        "type": "A", "direction": "bearish",
                        "level": a_price, "range": (a_price - 0.2 * atr_val, a_price + 0.2 * atr_val),
                        "strength": 1
                    })
                    b_retrace = (b_price - a_price) / leg_size if leg_size > 0 else 0
                    if 0.38 <= b_retrace <= 0.79:
                        zones.append({
                            "type": "B", "direction": "bearish",
                            "level": b_price, "range": (b_price - 0.2 * atr_val, b_price + 0.2 * atr_val),
                            "strength": 3
                        })
                    zones.append({
                        "type": "C", "direction": "bearish",
                        "level": a_price, "range": (a_price - 0.15 * atr_val, a_price + 0.15 * atr_val),
                        "strength": 2
                    })

    # Find nearest zone to price
    nearest = None
    min_dist = float("inf")
    for z in zones:
        lo, hi = z["range"]
        if lo <= price <= hi:
            dist = 0
        else:
            dist = min(abs(price - lo), abs(price - hi))
        if dist < min_dist:
            min_dist = dist
            nearest = z

    # Determine dominant direction from recent swings
    if len(all_swings) >= 2:
        last_two = all_swings[-2:]
        if last_two[0]["type"] == "L" and last_two[1]["type"] == "H":
            direction = "bullish"
        elif last_two[0]["type"] == "H" and last_two[1]["type"] == "L":
            direction = "bearish"
        else:
            direction = None
    else:
        direction = None

    return {"zones": zones, "nearest_zone": nearest, "direction": direction}


def _detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """Detect high-probability scalping candlestick patterns."""
    if df is None or df.empty or len(df) < 4:
        return {"pattern": None, "direction": 0}

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)

    curr_o, curr_h, curr_l, curr_c = o[-1], h[-1], l[-1], c[-1]
    prev_o, prev_h, prev_l, prev_c = o[-2], h[-2], l[-2], c[-2]
    prev2_o, prev2_h, prev2_l, prev2_c = o[-3], h[-3], l[-3], c[-3]

    curr_body = abs(curr_c - curr_o)
    curr_range = curr_h - curr_l if curr_h > curr_l else 0.001
    prev_body = abs(prev_c - prev_o)
    prev_range = prev_h - prev_l if prev_h > prev_l else 0.001

    # Bullish engulfing
    if (curr_c > curr_o and prev_c < prev_o
            and curr_c > prev_o and curr_o < prev_c
            and curr_body > prev_body * 0.8):
        return {"pattern": "bullish_engulfing", "direction": 1, "strength": 0.8}

    # Bearish engulfing
    if (curr_c < curr_o and prev_c > prev_o
            and curr_c < prev_o and curr_o > prev_c
            and curr_body > prev_body * 0.8):
        return {"pattern": "bearish_engulfing", "direction": -1, "strength": 0.8}

    # Bullish pin bar (hammer) - long lower wick, small body at top
    lower_wick = min(curr_o, curr_c) - curr_l
    upper_wick = curr_h - max(curr_o, curr_c)
    if lower_wick > 2.0 * curr_body and upper_wick < curr_body * 0.5 and curr_range > 0:
        return {"pattern": "bullish_pin_bar", "direction": 1, "strength": 0.7}

    # Bearish pin bar (shooting star) - long upper wick, small body at bottom
    if upper_wick > 2.0 * curr_body and lower_wick < curr_body * 0.5 and curr_range > 0:
        return {"pattern": "bearish_pin_bar", "direction": -1, "strength": 0.7}

    # Bullish inside bar breakout (current candle breaks above inside bar high)
    if prev_h < prev2_h and prev_l > prev2_l:
        if curr_c > prev_h and curr_c > curr_o:
            return {"pattern": "inside_bar_breakout_bull", "direction": 1, "strength": 0.6}
        if curr_c < prev_l and curr_c < curr_o:
            return {"pattern": "inside_bar_breakout_bear", "direction": -1, "strength": 0.6}

    # Momentum candle (strong close near high/low with volume)
    if curr_range > 0:
        close_location = (curr_c - curr_l) / curr_range if curr_c > curr_o else (curr_h - curr_c) / curr_range
        if close_location > 0.85 and curr_body > 0.6 * curr_range:
            d = 1 if curr_c > curr_o else -1
            return {"pattern": "momentum_candle", "direction": d, "strength": 0.65}

    return {"pattern": None, "direction": 0, "strength": 0}


def _detect_ema_squeeze(close: pd.Series, cfg: dict) -> dict:
    """Detect EMA squeeze: Bollinger Bands inside Keltner Channels = volatility contraction.

    Expansion after squeeze = high-probability directional breakout for scalping.
    """
    period = cfg.get("squeeze_period", 20)
    if len(close) < period + 10:
        return {"squeeze": False, "releasing": False, "direction": 0}

    c = close.to_numpy(float)

    # Bollinger Bands
    sma = pd.Series(c).rolling(period).mean().to_numpy()
    std = pd.Series(c).rolling(period).std().to_numpy()
    bb_upper = sma + 2.0 * std
    bb_lower = sma - 2.0 * std

    # Keltner Channels (EMA +/- 1.5 * ATR)
    ema_val = pd.Series(c).ewm(span=period, adjust=False).mean().to_numpy()
    atr_approx = pd.Series(np.abs(np.diff(c, prepend=c[0]))).ewm(span=period, adjust=False).mean().to_numpy()
    kc_upper = ema_val + 1.5 * atr_approx
    kc_lower = ema_val - 1.5 * atr_approx

    # Squeeze = BB inside KC
    in_squeeze = bb_upper[-1] < kc_upper[-1] and bb_lower[-1] > kc_lower[-1]
    prev_squeeze = bb_upper[-2] < kc_upper[-2] and bb_lower[-2] > kc_lower[-2]

    # Releasing = was in squeeze, now breaking out
    releasing = prev_squeeze and not in_squeeze

    direction = 0
    if releasing or in_squeeze:
        # Momentum direction from linear regression of last few closes
        lookback = min(6, len(c) - 1)
        y = c[-lookback:]
        x = np.arange(lookback, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        if abs(slope) > 0.0001:
            direction = 1 if slope > 0 else -1

    return {
        "squeeze": bool(in_squeeze),
        "releasing": bool(releasing),
        "direction": direction,
        "bb_width": float((bb_upper[-1] - bb_lower[-1]) / sma[-1] * 100) if sma[-1] > 0 else 0,
    }


def _session_scalp_boost(now_utc, cfg: dict) -> int:
    """Session-aware scalping scoring boost.

    Gold scalping windows (UTC):
    - London Open:    07:00-10:00  (+1 scalp points)
    - NY Open:       12:00-15:00  (+1 scalp points)
    - London-NY Overlap: 12:00-16:00 (+2 scalp points - BEST)
    - Asian session: 00:00-07:00  (avoid - low volatility)
    """
    scalp_cfg = cfg.get("scalper", {})
    if not scalp_cfg.get("enabled", True):
        return 0

    h = now_utc.hour
    m = now_utc.minute
    t = h * 60 + m

    # London-NY overlap = prime scalping
    if 12 * 60 <= t < 16 * 60:
        return 2
    # London open
    if 7 * 60 <= t < 10 * 60:
        return 1
    # NY open
    if 13 * 30 <= t < 15 * 60:
        return 1
    # Late Asian = avoid
    if 0 <= t < 5 * 60:
        return -1
    return 0


def _validate_retracement(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Validate retracement using 3-candle closing rule.
    A valid bullish retracement: 3 consecutive candles closing higher (pullback to demand).
    A valid bearish retracement: 3 consecutive candles closing lower (pullback to supply).
    Returns whether retracement is valid and its direction.
    """
    if df is None or df.empty or len(df) < 4:
        return {"valid": False, "direction": "none"}

    c = df["Close"].values[-4:]
    o = df["Open"].values[-4:]

    bull_closes = sum(1 for i in range(1, len(c)) if c[i] > c[i - 1])
    bear_closes = sum(1 for i in range(1, len(c)) if c[i] < c[i - 1])

    min_candles = cfg.get("retracement_min_candles", 3)
    bull_retrace = bull_closes >= min_candles - 1
    bear_retrace = bear_closes >= min_candles - 1

    return {
        "valid": bull_retrace or bear_retrace,
        "bullish": bull_retrace,
        "bearish": bear_retrace,
        "bull_closes": bull_closes,
        "bear_closes": bear_closes,
    }


def _detect_confluence_setups(fvg: dict, obs: dict, mitigation: dict,
                               breaker: dict, price: float, atr_val: float,
                               cfg: dict) -> dict:
    """
    Detect high-probability confluence setups:
    1. Unicorn: Breaker/Mitigation block + FVG overlap at same level
    2. BPR (Balanced Price Range): Two overlapping FVGs creating equilibrium
    3. OB+FVG: Order Block spatially aligned with FVG

    These setups are among the strongest in SMC because they show
    institutional confluence at a specific price level.
    """
    setups = []
    overlap_tolerance = cfg.get("confluence_overlap_atr", 0.5) * atr_val

    for direction in ["bullish", "bearish"]:
        fvgs = fvg.get(f"{direction}_fvgs", [])
        uns_mitigated = [f for f in fvgs if not f.get("mitigated", False)]

        for fvg_item in uns_mitigated:
            fvg_top = fvg_item["top"]
            fvg_bottom = fvg_item["bottom"]
            fvg_mid = (fvg_top + fvg_bottom) / 2

            if direction == "bullish":
                for mb in mitigation.get("bullish", []):
                    if _zones_overlap(fvg_bottom, fvg_top, mb["bottom"], mb["top"], overlap_tolerance):
                        setups.append({
                            "type": "unicorn",
                            "direction": "bullish",
                            "price_level": fvg_mid,
                            "fvg": fvg_item,
                            "block": mb,
                            "strength": 1.0,
                        })

                for bb in breaker.get("bullish", []):
                    if _zones_overlap(fvg_bottom, fvg_top, bb["bottom"], bb["top"], overlap_tolerance):
                        setups.append({
                            "type": "unicorn",
                            "direction": "bullish",
                            "price_level": fvg_mid,
                            "fvg": fvg_item,
                            "block": bb,
                            "strength": 0.9,
                        })

            else:
                for mb in mitigation.get("bearish", []):
                    if _zones_overlap(fvg_bottom, fvg_top, mb["bottom"], mb["top"], overlap_tolerance):
                        setups.append({
                            "type": "unicorn",
                            "direction": "bearish",
                            "price_level": fvg_mid,
                            "fvg": fvg_item,
                            "block": mb,
                            "strength": 1.0,
                        })

                for bb in breaker.get("bearish", []):
                    if _zones_overlap(fvg_bottom, fvg_top, bb["bottom"], bb["top"], overlap_tolerance):
                        setups.append({
                            "type": "unicorn",
                            "direction": "bearish",
                            "price_level": fvg_mid,
                            "fvg": fvg_item,
                            "block": bb,
                            "strength": 0.9,
                        })

    bull_unmitigated = [f for f in fvg.get("bullish_fvgs", []) if not f.get("mitigated", False)]
    bear_unmitigated = [f for f in fvg.get("bearish_fvgs", []) if not f.get("mitigated", False)]

    for bf in bull_unmitigated:
        for br in bear_unmitigated:
            if _zones_overlap(bf["bottom"], bf["top"], br["bottom"], br["top"], overlap_tolerance):
                setups.append({
                    "type": "bpr",
                    "direction": "neutral",
                    "price_level": (bf["mid"] + br["mid"]) / 2,
                    "fvg_bull": bf,
                    "fvg_bear": br,
                    "strength": 0.8,
                })

    for direction in ["bullish", "bearish"]:
        fvgs = fvg.get(f"{direction}_fvgs", [])
        uns = [f for f in fvgs if not f.get("mitigated", False)]

        for fvg_item in uns:
            fvg_top = fvg_item["top"]
            fvg_bottom = fvg_item["bottom"]

            if direction == "bullish":
                for ob in obs.get("bullish", []):
                    if not ob.get("mitigated", False):
                        if _zones_overlap(fvg_bottom, fvg_top, ob["bottom"], ob["top"], overlap_tolerance):
                            setups.append({
                                "type": "ob_fvg",
                                "direction": "bullish",
                                "price_level": fvg_item["mid"],
                                "fvg": fvg_item,
                                "ob": ob,
                                "strength": 0.7,
                            })
            else:
                for ob in obs.get("bearish", []):
                    if not ob.get("mitigated", False):
                        if _zones_overlap(fvg_bottom, fvg_top, ob["bottom"], ob["top"], overlap_tolerance):
                            setups.append({
                                "type": "ob_fvg",
                                "direction": "bearish",
                                "price_level": fvg_item["mid"],
                                "fvg": fvg_item,
                                "ob": ob,
                                "strength": 0.7,
                            })

    return {"setups": setups, "unicorn_count": sum(1 for s in setups if s["type"] == "unicorn"),
            "bpr_count": sum(1 for s in setups if s["type"] == "bpr"),
            "ob_fvg_count": sum(1 for s in setups if s["type"] == "ob_fvg")}


def _zones_overlap(bottom1: float, top1: float, bottom2: float, top2: float, tolerance: float = 0) -> bool:
    return bottom1 <= top2 + tolerance and bottom2 <= top1 + tolerance


def _choch_in_key_level(structure: dict, zones: dict, obs: dict, price: float, atr_val: float) -> dict:
    """
    Check if a CHoCH event happened inside an HTF key level (supply/demand zone or OB).
    This is one of the strongest confluences in SMC — a lower timeframe structure
    shift happening inside a higher timeframe point of interest.

    Returns whether CHoCH is inside a key level and which level.
    """
    event = structure.get("event")
    choch_level = structure.get("choch_level")

    if event is None or choch_level is None:
        return {"inside_key_level": False, "level_type": None, "level_price": None}

    for zone in zones.get("demand", []):
        if zone.get("mitigated", False):
            continue
        if zone["bottom"] <= choch_level <= zone["top"]:
            return {"inside_key_level": True, "level_type": "demand",
                    "level_price": (zone["bottom"] + zone["top"]) / 2}

    for zone in zones.get("supply", []):
        if zone.get("mitigated", False):
            continue
        if zone["bottom"] <= choch_level <= zone["top"]:
            return {"inside_key_level": True, "level_type": "supply",
                    "level_price": (zone["bottom"] + zone["top"]) / 2}

    for ob in obs.get("bullish", []):
        if not ob.get("mitigated", False):
            if ob["bottom"] <= choch_level <= ob["top"]:
                return {"inside_key_level": True, "level_type": "bullish_ob",
                        "level_price": (ob["bottom"] + ob["top"]) / 2}

    for ob in obs.get("bearish", []):
        if not ob.get("mitigated", False):
            if ob["bottom"] <= choch_level <= ob["top"]:
                return {"inside_key_level": True, "level_type": "bearish_ob",
                        "level_price": (ob["bottom"] + ob["top"]) / 2}

    return {"inside_key_level": False, "level_type": None, "level_price": None}


def _pd_array_priority_scoring(price: float, atr_val: float, cfg: dict,
                                mitigation: dict, breaker: dict, fvg: dict,
                                obs: dict, rejection: dict, zones_data: dict,
                                sweeps: dict, structure: dict) -> dict:
    """
    PD-Array priority scoring. Instead of summing all scores flatly,
    check reference points in priority order:
    1. Mitigation Block (highest priority)
    2. Breaker Block
    3. Liquidity Void
    4. FVG
    5. Order Block
    6. Old High/Low (swing points)
    7. Rejection Block (lowest priority)

    The highest-priority zone that price is near determines the primary
    entry logic. Additional zones at the same level add confluence.
    """
    from liquidity import detect_liquidity_voids
    pd_score = 0
    pd_direction = None
    pd_level_type = None
    pd_level_price = None
    confluence_zones = []

    max_dist = cfg.get("pd_zone_max_dist_atr", 1.0) * atr_val

    for mb in mitigation.get("bullish", []):
        if not mb.get("mitigated", False) and mb["touches"] <= cfg.get("mitigation_max_touches", 5):
            dist = abs(price - mb["mid"])
            if dist <= max_dist:
                confluence_zones.append({"type": "mitigation", "direction": "bullish",
                                        "price": mb["mid"], "strength": 1.0, "dist": dist})

    for mb in mitigation.get("bearish", []):
        if not mb.get("mitigated", False) and mb["touches"] <= cfg.get("mitigation_max_touches", 5):
            dist = abs(price - mb["mid"])
            if dist <= max_dist:
                confluence_zones.append({"type": "mitigation", "direction": "bearish",
                                        "price": mb["mid"], "strength": 1.0, "dist": dist})

    for bb in breaker.get("bullish", []):
        if not bb.get("mitigated", False) and bb["touches"] <= cfg.get("breaker_max_touches", 5):
            dist = abs(price - bb["mid"])
            if dist <= max_dist:
                confluence_zones.append({"type": "breaker", "direction": "bullish",
                                        "price": bb["mid"], "strength": 0.9, "dist": dist})

    for bb in breaker.get("bearish", []):
        if not bb.get("mitigated", False) and bb["touches"] <= cfg.get("breaker_max_touches", 5):
            dist = abs(price - bb["mid"])
            if dist <= max_dist:
                confluence_zones.append({"type": "breaker", "direction": "bearish",
                                        "price": bb["mid"], "strength": 0.9, "dist": dist})

    for fvg_item in fvg.get("bullish_fvgs", []):
        if not fvg_item.get("mitigated", False):
            dist = abs(price - fvg_item["mid"])
            if dist <= max_dist:
                str_mult = 1.0 if fvg_item.get("size_class") == "big" else 0.6
                confluence_zones.append({"type": "fvg", "direction": "bullish",
                                        "price": fvg_item["mid"], "strength": str_mult, "dist": dist})

    for fvg_item in fvg.get("bearish_fvgs", []):
        if not fvg_item.get("mitigated", False):
            dist = abs(price - fvg_item["mid"])
            if dist <= max_dist:
                str_mult = 1.0 if fvg_item.get("size_class") == "big" else 0.6
                confluence_zones.append({"type": "fvg", "direction": "bearish",
                                        "price": fvg_item["mid"], "strength": str_mult, "dist": dist})

    for ob in obs.get("bullish", []):
        if not ob.get("mitigated", False):
            ob_mid = (ob["top"] + ob["bottom"]) / 2
            dist = abs(price - ob_mid)
            if dist <= max_dist:
                confluence_zones.append({"type": "order_block", "direction": "bullish",
                                        "price": ob_mid, "strength": 0.7, "dist": dist})

    for ob in obs.get("bearish", []):
        if not ob.get("mitigated", False):
            ob_mid = (ob["top"] + ob["bottom"]) / 2
            dist = abs(price - ob_mid)
            if dist <= max_dist:
                confluence_zones.append({"type": "order_block", "direction": "bearish",
                                        "price": ob_mid, "strength": 0.7, "dist": dist})

    for rb in rejection.get("bullish", []):
        if not rb.get("mitigated", False) and rb["touches"] <= cfg.get("rejection_max_touches", 5):
            dist = abs(price - rb["price"])
            if dist <= max_dist:
                confluence_zones.append({"type": "rejection", "direction": "bullish",
                                        "price": rb["price"], "strength": 0.5, "dist": dist})

    for rb in rejection.get("bearish", []):
        if not rb.get("mitigated", False) and rb["touches"] <= cfg.get("rejection_max_touches", 5):
            dist = abs(price - rb["price"])
            if dist <= max_dist:
                confluence_zones.append({"type": "rejection", "direction": "bearish",
                                        "price": rb["price"], "strength": 0.5, "dist": dist})

    if not confluence_zones:
        return {"pd_score": 0, "pd_direction": None, "pd_level_type": None,
                "pd_level_price": None, "confluence_zones": [], "confluence_count": 0}

    priority = {"mitigation": 1, "breaker": 2, "fvg": 3, "order_block": 4, "rejection": 5}
    confluence_zones.sort(key=lambda z: (priority.get(z["type"], 99), z["dist"]))

    bull_zones = [z for z in confluence_zones if z["direction"] == "bullish"]
    bear_zones = [z for z in confluence_zones if z["direction"] == "bearish"]

    bull_score = sum(z["strength"] for z in bull_zones)
    bear_score = sum(z["strength"] for z in bear_zones)

    primary = confluence_zones[0]
    pd_direction = primary["direction"]
    pd_level_type = primary["type"]
    pd_level_price = primary["price"]

    if bull_score > bear_score:
        pd_score = int(round(bull_score * 2))
        pd_direction = "bullish"
    elif bear_score > bull_score:
        pd_score = int(round(bear_score * 2))
        pd_direction = "bearish"
    else:
        pd_score = 0
        pd_direction = None

    return {
        "pd_score": pd_score,
        "pd_direction": pd_direction,
        "pd_level_type": pd_level_type,
        "pd_level_price": pd_level_price,
        "confluence_zones": confluence_zones,
        "confluence_count": len(confluence_zones),
        "bull_score": bull_score,
        "bear_score": bear_score,
    }


def analyze(df: pd.DataFrame, cfg: dict, bias: int = 0, live_price: float = None) -> dict:
    empty = {
        "signal": "NONE", "score": 0, "price": 0.0, "entry": 0.0,
        "stop_loss": 0.0, "take_profit": 0.0, "details": {},
    }
    if df is None or df.empty or len(df) < 60:
        return empty

    close = df["Close"]
    gc_price = float(close.iloc[-1])
    atr_val = float(_atr(df, cfg.get("atr_period", 14)).iloc[-1])

    price_offset = 0.0
    if live_price and gc_price > 0:
        price_offset = live_price - gc_price
        price = live_price
    else:
        price = gc_price

    zones = detect_zones(df, cfg)
    sweep, all_sweeps = detect_liquidity_sweep(df, cfg)
    obs = detect_order_blocks(df, cfg)
    structure = analyze_structure(df, cfg)
    pools = detect_liquidity_pools(df, cfg)
    fvg = detect_fvg(df, cfg)
    pd_info = compute_premium_discount(df, cfg)
    mitigation = detect_mitigation_blocks(df, obs, cfg)
    breaker = detect_breaker_blocks(df, obs, structure, cfg)

    crt = _crt_analysis(df, atr_val)
    oabc = _elliott_oabc(df, structure, atr_val)

    if price_offset != 0:
        for z in zones["demand"]:
            z["bottom"] += price_offset
            z["top"] += price_offset
        for z in zones["supply"]:
            z["bottom"] += price_offset
            z["top"] += price_offset

    demand_zone = _find_nearest_zone(zones["demand"], price, atr_val)
    supply_zone = _find_nearest_zone(zones["supply"], price, atr_val)

    sweep_level = sweep["level"]
    if sweep_level is not None and price_offset != 0:
        sweep_level += price_offset

    buy_score = 0
    sell_score = 0
    buy_factors = 0
    sell_factors = 0
    details = {}

    # ========================================================================
    # INSTITUTIONAL ORDER FLOW SCORING (Clean - No Noise)
    # Only: SMC, CRT, FMSO, OABC, Strong S/D, Liquidity
    # ========================================================================

    # --- 1. MARKET STRUCTURE (SMC) - Core Direction ---
    if structure["trend"] == "bullish":
        buy_score += 2
        buy_factors += 1
        details["structure"] = f"bullish ({structure['counts'].get('HH', 0)}HH {structure['counts'].get('HL', 0)}HL)"
    elif structure["trend"] == "bearish":
        sell_score += 2
        sell_factors += 1
        details["structure"] = f"bearish ({structure['counts'].get('LH', 0)}LH {structure['counts'].get('LL', 0)}LL)"

    event = structure.get("event")
    if event == "CHoCH_BULL":
        buy_score += 3
        buy_factors += 1
        details["structure_event"] = f"CHoCH bullish @ {structure.get('choch_level', 0):.2f}"
        details["choch_level"] = structure.get("choch_level", 0)
    elif event == "CHoCH_BEAR":
        sell_score += 3
        sell_factors += 1
        details["structure_event"] = f"CHoCH bearish @ {structure.get('choch_level', 0):.2f}"
        details["choch_level"] = structure.get("choch_level", 0)
    elif event == "BOS_BULL":
        buy_score += 1
        buy_factors += 1
        details["structure_event"] = f"BOS bullish @ {structure.get('bos_level', 0):.2f}"
        details["bos_level"] = structure.get("bos_level", 0)
    elif event == "BOS_BEAR":
        sell_score += 1
        sell_factors += 1
        details["structure_event"] = f"BOS bearish @ {structure.get('bos_level', 0):.2f}"
        details["bos_level"] = structure.get("bos_level", 0)

    # --- 2. LIQUIDITY EVENTS (SMC) - Sweep + Pools + FVG ---
    if sweep["bullish"] and sweep_level is not None:
        bars = sweep["bars_ago"]
        sweep_pts = 1 if bars > 3 else (2 if bars > 1 else 3)
        if sweep.get("volume_confirmed", False):
            sweep_pts += 1
        buy_score += sweep_pts
        buy_factors += 1
        vol_tag = "+vol" if sweep.get("volume_confirmed", False) else ""
        details["sweep"] = f"bullish sweep {sweep_level:.2f} ({bars}ba {vol_tag})"

    if sweep["bearish"] and sweep_level is not None:
        bars = sweep["bars_ago"]
        sweep_pts = 1 if bars > 3 else (2 if bars > 1 else 3)
        if sweep.get("volume_confirmed", False):
            sweep_pts += 1
        sell_score += sweep_pts
        sell_factors += 1
        vol_tag = "+vol" if sweep.get("volume_confirmed", False) else ""
        details["sweep"] = f"bearish sweep {sweep_level:.2f} ({bars}ba {vol_tag})"

    if fvg.get("nearest"):
        fvg_dir = fvg["nearest"]["direction"]
        if fvg_dir == "bullish":
            buy_score += 1
            buy_factors += 1
            details["fvg"] = f"bullish FVG {fvg['nearest']['bottom']:.2f}-{fvg['nearest']['top']:.2f}"
        elif fvg_dir == "bearish":
            sell_score += 1
            sell_factors += 1
            details["fvg"] = f"bearish FVG {fvg['nearest']['bottom']:.2f}-{fvg['nearest']['top']:.2f}"

    if pools.get("nearest_bsl"):
        details["nearest_bsl"] = f"{pools['nearest_bsl']['price']:.2f}"
    if pools.get("nearest_ssl"):
        details["nearest_ssl"] = f"{pools['nearest_ssl']['price']:.2f}"

    # --- 3. STRONG SUPPLY/DEMAND ZONES (Enhanced with 0-100 scoring) ---
    max_touches = cfg.get("max_zone_touches", 3)
    min_zone_score = cfg.get("min_zone_score", 40)

    if demand_zone:
        zone_score = demand_zone.get("strength_score", 0)
        zone_status = demand_zone.get("status", "FAR")
        zone_fresh = demand_zone.get("fresh", False)
        zone_reactions = demand_zone.get("reaction_count", 0)
        zone_liq_sweep = demand_zone.get("liquidity_sweep", False)
        zone_struct = demand_zone.get("structure_confirmed", False)

        if zone_score >= min_zone_score and zone_status in ("INSIDE_ZONE", "AT_ZONE", "APPROACHING", "REJECTING"):
            zone_pts = int(zone_score / 25)
            if zone_fresh:
                zone_pts += 1
            if zone_liq_sweep:
                zone_pts += 1
            if zone_struct:
                zone_pts += 1
            zone_pts = min(zone_pts, 5)
            buy_score += zone_pts
            buy_factors += 1
            fresh_tag = "FRESH" if zone_fresh else f"mitigated ({demand_zone['touches']}x)"
            details["zone"] = (f"demand {demand_zone['bottom']:.2f}-{demand_zone['top']:.2f} "
                              f"[{demand_zone['strength_label']}] ({fresh_tag}, {zone_status})")
            details["zone_score"] = zone_score
            details["zone_reactions"] = zone_reactions
            details["zone_reason"] = demand_zone.get("reason", "")
            if zone_status in ("INSIDE_ZONE", "AT_ZONE", "REJECTING"):
                sell_score = max(0, sell_score - 3)
                details["demand_proximity"] = f"price {zone_status.lower()} demand zone"
        else:
            details["zone_weak"] = f"demand zone too weak (score {zone_score}) or too far ({zone_status})"

    if supply_zone:
        zone_score = supply_zone.get("strength_score", 0)
        zone_status = supply_zone.get("status", "FAR")
        zone_fresh = supply_zone.get("fresh", False)
        zone_reactions = supply_zone.get("reaction_count", 0)
        zone_liq_sweep = supply_zone.get("liquidity_sweep", False)
        zone_struct = supply_zone.get("structure_confirmed", False)

        if zone_score >= min_zone_score and zone_status in ("INSIDE_ZONE", "AT_ZONE", "APPROACHING", "REJECTING"):
            zone_pts = int(zone_score / 25)
            if zone_fresh:
                zone_pts += 1
            if zone_liq_sweep:
                zone_pts += 1
            if zone_struct:
                zone_pts += 1
            zone_pts = min(zone_pts, 5)
            sell_score += zone_pts
            sell_factors += 1
            fresh_tag = "FRESH" if zone_fresh else f"mitigated ({supply_zone['touches']}x)"
            details["zone"] = (f"supply {supply_zone['bottom']:.2f}-{supply_zone['top']:.2f} "
                              f"[{supply_zone['strength_label']}] ({fresh_tag}, {zone_status})")
            details["zone_score"] = zone_score
            details["zone_reactions"] = zone_reactions
            details["zone_reason"] = supply_zone.get("reason", "")
            if zone_status in ("INSIDE_ZONE", "AT_ZONE", "REJECTING"):
                buy_score = max(0, buy_score - 3)
                details["supply_proximity"] = f"price {zone_status.lower()} supply zone"
        else:
            details["zone_weak"] = f"supply zone too weak (score {zone_score}) or too far ({zone_status})"

    # --- 4. ORDER BLOCKS (SMC) ---
    bullish_ob = find_nearest_ob(obs["bullish"], price, atr_val, cfg)
    bearish_ob = find_nearest_ob(obs["bearish"], price, atr_val, cfg)

    if bullish_ob:
        ob_touches = bullish_ob["touches"]
        max_ob_touches = cfg.get("ob_max_touches", 8)
        if ob_touches <= max_ob_touches:
            ob_pts = 1
            if ob_touches == 0:
                ob_pts = 2
            if bullish_ob.get("bos_confirmed", False):
                ob_pts += 1
            buy_score += ob_pts
            buy_factors += 1
            tags = []
            freshness = "fresh" if ob_touches == 0 else f"touched {ob_touches}x"
            tags.append(freshness)
            if bullish_ob.get("bos_confirmed", False):
                tags.append("BOS")
            details["order_block"] = f"bullish OB {bullish_ob['bottom']:.2f}-{bullish_ob['top']:.2f} ({', '.join(tags)})"

    if bearish_ob:
        ob_touches = bearish_ob["touches"]
        max_ob_touches = cfg.get("ob_max_touches", 8)
        if ob_touches <= max_ob_touches:
            ob_pts = 1
            if ob_touches == 0:
                ob_pts = 2
            if bearish_ob.get("bos_confirmed", False):
                ob_pts += 1
            sell_score += ob_pts
            sell_factors += 1
            tags = []
            freshness = "fresh" if ob_touches == 0 else f"touched {ob_touches}x"
            tags.append(freshness)
            if bearish_ob.get("bos_confirmed", False):
                tags.append("BOS")
            details["order_block"] = f"bearish OB {bearish_ob['bottom']:.2f}-{bearish_ob['top']:.2f} ({', '.join(tags)})"

    # --- 5. FMSO - FAILED MITIGATION / BREAKER BLOCKS ---
    bull_mit = find_nearest_mitigation(mitigation.get("bullish", []), price, atr_val, cfg)
    bear_mit = find_nearest_mitigation(mitigation.get("bearish", []), price, atr_val, cfg)

    if bull_mit:
        mit_pts = 2
        if bull_mit["touches"] == 0:
            mit_pts = 3
        buy_score += mit_pts
        buy_factors += 1
        details["mitigation_block"] = f"bullish {bull_mit['bottom']:.2f}-{bull_mit['top']:.2f}"

    if bear_mit:
        mit_pts = 2
        if bear_mit["touches"] == 0:
            mit_pts = 3
        sell_score += mit_pts
        sell_factors += 1
        details["mitigation_block"] = f"bearish {bear_mit['bottom']:.2f}-{bear_mit['top']:.2f}"

    bull_brk = find_nearest_breaker(breaker.get("bullish", []), price, atr_val, cfg)
    bear_brk = find_nearest_breaker(breaker.get("bearish", []), price, atr_val, cfg)

    if bull_brk:
        brk_pts = 2
        if bull_brk.get("bos_confirmed", False):
            brk_pts = 3
        buy_score += brk_pts
        buy_factors += 1
        details["breaker_block"] = f"bullish {bull_brk['bottom']:.2f}-{bull_brk['top']:.2f}"

    if bear_brk:
        brk_pts = 2
        if bear_brk.get("bos_confirmed", False):
            brk_pts = 3
        sell_score += brk_pts
        sell_factors += 1
        details["breaker_block"] = f"bearish {bear_brk['bottom']:.2f}-{bear_brk['top']:.2f}"

    # --- 6. ELLIOTT WAVE OABC ZONES ---
    if oabc.get("nearest_zone"):
        nz = oabc["nearest_zone"]
        if nz["direction"] == "bullish":
            buy_score += nz["strength"]
            buy_factors += 1
            details["elliott_oabc"] = f"bullish {nz['type']} zone @ {nz['level']:.2f}"
        elif nz["direction"] == "bearish":
            sell_score += nz["strength"]
            sell_factors += 1
            details["elliott_oabc"] = f"bearish {nz['type']} zone @ {nz['level']:.2f}"

    # --- 7. CRT - CANDLE RANGE THEORY ---
    if crt.get("crt_score", 0) > 0:
        if crt.get("breakout") == "bullish":
            buy_score += crt["crt_score"]
            buy_factors += 1
            details["crt"] = f"bullish breakout"
        elif crt.get("breakout") == "bearish":
            sell_score += crt["crt_score"]
            sell_factors += 1
            details["crt"] = f"bearish breakout"
        if crt.get("wick_rejection") == "bullish":
            buy_score += 1
            buy_factors += 1
            details["crt_wick"] = "bullish wick rejection"
        elif crt.get("wick_rejection") == "bearish":
            sell_score += 1
            sell_factors += 1
            details["crt_wick"] = "bearish wick rejection"
        if crt.get("expansion") == "bullish":
            buy_score += 1
            buy_factors += 1
            details["crt_expansion"] = "bullish expansion"
        elif crt.get("expansion") == "bearish":
            sell_score += 1
            sell_factors += 1
            details["crt_expansion"] = "bearish expansion"

    # --- 8. PREMIUM/DISCOUNT ALIGNMENT ---
    pd_zone = pd_info.get("zone", "equilibrium")
    if pd_zone in ("deep_discount", "discount") and buy_score > 0:
        if not supply_zone or price > supply_zone["top"] + 0.5 * atr_val:
            buy_score += 1
            buy_factors += 1
        details["pd_zone"] = pd_zone
    elif pd_zone in ("deep_premium", "premium") and sell_score > 0:
        if not demand_zone or price < demand_zone["bottom"] - 0.5 * atr_val:
            sell_score += 1
            sell_factors += 1
        details["pd_zone"] = pd_zone

    # --- BIAS FILTER (Higher TF) ---
    bias_filter = cfg.get("bias_enabled", True)
    if bias_filter:
        if bias > 0:
            buy_score += 1
            buy_factors += 1
        elif bias < 0:
            sell_score += 1
            sell_factors += 1

    bias_label = {1: "bullish", -1: "bearish", 0: "flat"}.get(bias, "flat")
    details["bias"] = bias_label

    # ========================================================================
    # SIGNAL DECISION - Clean thresholds, quality over quantity
    # ========================================================================
    threshold = cfg.get("signal_threshold", 5)
    strong = cfg.get("strong_threshold", 7)
    min_factors = cfg.get("min_confluence_factors", 3)
    sl_buffer = cfg.get("sl_atr_buffer", 1.0)

    signal = "NONE"
    entry = sl = tp = price
    tp1 = tp2 = price

    pre_threshold = cfg.get("get_ready_threshold", 2)

    if buy_score >= threshold and buy_score > sell_score and buy_factors >= min_factors:
        signal = "STRONG_BUY" if buy_score >= strong else "BUY"
        entry = price
        if sweep["bullish"] and sweep_level is not None:
            sl = min(sweep_level, demand_zone["bottom"] if demand_zone else price) - sl_buffer * atr_val
        elif demand_zone:
            sl = demand_zone["bottom"] - sl_buffer * atr_val
        else:
            sl = price - 2 * sl_buffer * atr_val
        tp = compute_dynamic_tp(entry, sl, "BUY", pools, atr_val)
        tp1, tp2 = compute_multi_tp(entry, sl, "BUY", pools, atr_val)

    elif sell_score >= threshold and sell_score > buy_score and sell_factors >= min_factors:
        signal = "STRONG_SELL" if sell_score >= strong else "SELL"
        entry = price
        if sweep["bearish"] and sweep_level is not None:
            sl = max(sweep_level, supply_zone["top"] if supply_zone else price) + sl_buffer * atr_val
        elif supply_zone:
            sl = supply_zone["top"] + sl_buffer * atr_val
        else:
            sl = price + 2 * sl_buffer * atr_val
        tp = compute_dynamic_tp(entry, sl, "SELL", pools, atr_val)
        tp1, tp2 = compute_multi_tp(entry, sl, "SELL", pools, atr_val)

    elif buy_score >= pre_threshold and buy_score > sell_score and buy_factors >= 2:
        signal = "WATCH_BUY"
        entry = price
        if demand_zone:
            sl = demand_zone["bottom"] - sl_buffer * atr_val
        else:
            sl = price - 2 * sl_buffer * atr_val
        tp = compute_dynamic_tp(entry, sl, "BUY", pools, atr_val)
        tp1, tp2 = compute_multi_tp(entry, sl, "BUY", pools, atr_val)

    elif sell_score >= pre_threshold and sell_score > buy_score and sell_factors >= 2:
        signal = "WATCH_SELL"
        entry = price
        if supply_zone:
            sl = supply_zone["top"] + sl_buffer * atr_val
        else:
            sl = price + 2 * sl_buffer * atr_val
        tp = compute_dynamic_tp(entry, sl, "SELL", pools, atr_val)
        tp1, tp2 = compute_multi_tp(entry, sl, "SELL", pools, atr_val)

    if price_offset != 0:
        details["data_source"] = "GC=F calibrated to live XAU/USD spot"
        details["offset"] = f"{price_offset:.2f}"

    details["buy_score"] = buy_score
    details["sell_score"] = sell_score
    details["confluence"] = buy_factors if buy_score > sell_score else sell_factors

    return {
        "signal": signal,
        "score": max(buy_score, sell_score),
        "price": price,
        "entry": round(entry, 2),
        "stop_loss": round(sl, 2),
        "take_profit": round(tp, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "details": details,
    }


def _build_zones_arrays(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray, cfg: dict):
    n = len(close)
    left = cfg.get("zone_left", 3)
    right = cfg.get("zone_right", 3)
    move_atr = cfg.get("zone_move_atr", 1.0)

    piv_low_idx = []
    piv_high_idx = []
    for i in range(left, n - left):
        if low[i] == min(low[i - left:i + left + 1]):
            piv_low_idx.append(i)
        if high[i] == max(high[i - left:i + left + 1]):
            piv_high_idx.append(i)

    def build(kind, piv_idx):
        zones = []
        for idx in piv_idx:
            if idx + right + 1 >= n:
                continue
            start = max(0, idx - left)
            base_high = float(max(high[start:idx])) if idx > start else float(high[idx])
            base_low = float(min(low[start:idx])) if idx > start else float(low[idx])
            if kind == "demand":
                touch_idx = np.where(low[idx + 1:] <= base_high)[0] + idx + 1
            else:
                touch_idx = np.where(high[idx + 1:] <= base_low)[0] + idx + 1
            zones.append(
                {"kind": kind, "top": base_high, "bottom": base_low, "index": idx, "touch": touch_idx}
            )
        return zones

    demand = build("demand", piv_low_idx)
    supply = build("supply", piv_high_idx)
    return demand, supply


def build_signal_series(df: pd.DataFrame, cfg: dict, bias: np.ndarray | None = None) -> pd.DataFrame:
    n = len(df)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    vol = df["Volume"].to_numpy(float)
    atr_s = _atr(df, cfg.get("atr_period", 14)).to_numpy(float)

    fast = ema(df["Close"], cfg["ma_fast"]).to_numpy(float)
    slow = ema(df["Close"], cfg["ma_slow"]).to_numpy(float)
    diff = np.diff(close, prepend=close[0])
    body = np.abs(diff)
    atr_ma = cfg.get("market_mover_atr", 0.5)
    strong_up = (diff > 0) & (body > atr_ma * atr_s)
    strong_dn = (diff < 0) & (body > atr_ma * atr_s)
    momentum = np.where(fast > slow, 1, -1) + np.where(strong_up, 1, np.where(strong_dn, -1, 0))

    vw_mom_lookback = cfg.get("vw_mom_lookback", 10)
    opens = df["Open"].to_numpy(float)
    vw_mom = np.zeros(n, float)
    for i in range(1, n):
        lb = min(vw_mom_lookback, i + 1)
        c_slice = close[i - lb + 1:i + 1]
        o_slice = opens[i - lb + 1:i + 1]
        v_slice = vol[i - lb + 1:i + 1]
        total_v = np.sum(v_slice)
        if total_v > 0 and np.all(o_slice > 0):
            rets = (c_slice - o_slice) / o_slice
            vw_mom[i] = np.clip(np.sum(v_slice * rets) / total_v * 100, -1.0, 1.0)

    vwap_lookback = cfg.get("vwap_lookback", 78)
    typical = (high + low + close) / 3.0
    tv_series = pd.Series(typical * vol).rolling(vwap_lookback, min_periods=1).sum()
    v_series = pd.Series(vol).rolling(vwap_lookback, min_periods=1).sum()
    vwap_arr = np.full(n, np.nan)
    valid = v_series.to_numpy() > 0
    vwap_arr[valid] = (tv_series.to_numpy()[valid] / v_series.to_numpy()[valid])
    vwap_bias = np.where(close > vwap_arr, 1, np.where(close < vwap_arr, -1, 0)).astype(int)

    vol_period = cfg.get("volume_period", 14)
    vol_avg = pd.Series(vol).rolling(vol_period, min_periods=1).mean().to_numpy(float)
    climax_mult = cfg.get("climax_mult", 3.0)
    climax = (vol > climax_mult * vol_avg) & (vol_avg > 0)

    rsi_val = rsi(df["Close"], cfg["rsi_period"]).to_numpy(float)
    macd_line, macd_sig, _ = macd(df["Close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    ml = macd_line.to_numpy(float)
    ms = macd_sig.to_numpy(float)
    bullish_f = (ml > ms) & (rsi_val < cfg["rsi_overbought"])
    bearish_f = (ml < ms) & (rsi_val > cfg["rsi_oversold"])

    window = cfg.get("sweep_window", 4)
    sweep_event_bull = np.zeros(n, bool)
    sweep_event_bear = np.zeros(n, bool)
    sweep_event_bull_level = np.full(n, np.nan)
    sweep_event_bear_level = np.full(n, np.nan)
    piv_low_level = np.full(n, np.nan)
    piv_high_level = np.full(n, np.nan)
    pl_lo, pl_hi = [], []
    left_p = cfg.get("sweep_pivot", 3)
    for i in range(left_p, n - left_p):
        if low[i] == min(low[i - left_p:i + left_p + 1]):
            piv_low_level[i] = low[i]
            pl_lo.append(i)
        if high[i] == max(high[i - left_p:i + left_p + 1]):
            piv_high_level[i] = high[i]
            pl_hi.append(i)

    for i in range(1, n):
        for j in pl_lo:
            if j >= i:
                break
            level = piv_low_level[j]
            if low[i] < level and close[i] > level:
                sweep_event_bull[i] = True
                sweep_event_bull_level[i] = level
                break
        for j in pl_hi:
            if j >= i:
                break
            level = piv_high_level[j]
            if high[i] > level and close[i] < level:
                sweep_event_bear[i] = True
                sweep_event_bear_level[i] = level
                break

    sweep_bull = np.zeros(n, bool)
    sweep_bear = np.zeros(n, bool)
    sweep_bull_level = np.full(n, np.nan)
    sweep_bear_level = np.full(n, np.nan)
    for i in range(n):
        for k in range(max(0, i - window + 1), i + 1):
            if sweep_event_bull[k]:
                sweep_bull[i] = True
                sweep_bull_level[i] = sweep_event_bull_level[k]
                break
        for k in range(max(0, i - window + 1), i + 1):
            if sweep_event_bear[k]:
                sweep_bear[i] = True
                sweep_bear_level[i] = sweep_event_bear_level[k]
                break

    demand_cands, supply_cands = _build_zones_arrays(high, low, close, atr_s, cfg)
    right = cfg.get("zone_right", 3)
    move_atr = cfg.get("zone_move_atr", 1.0)

    near_demand = np.full(n, np.nan)
    near_supply = np.full(n, np.nan)
    zone_demand_bottom = np.full(n, np.nan)
    zone_supply_top = np.full(n, np.nan)
    demand_fresh = np.zeros(n, bool)
    supply_fresh = np.zeros(n, bool)

    def nearest_zone(cands, i, kind):
        active = []
        for z in cands:
            idx = z["index"]
            if idx >= i:
                break
            if i <= idx + right:
                continue
            ahead_i = min(idx + 1 + right, i)
            move = (close[ahead_i] - low[idx]) if kind == "demand" else (high[idx] - close[ahead_i])
            if move <= move_atr * atr_s[idx]:
                continue
            touch_count = int(np.searchsorted(z["touch"], i, side="right"))
            active.append((z["bottom"], z["top"], touch_count))
        if not active:
            return None
        active.sort(key=lambda t: t[0])
        merged = []
        for bottom, top, tc in active:
            if merged and bottom < merged[-1][1]:
                merged[-1] = (min(merged[-1][0], bottom), max(merged[-1][1], top), min(merged[-1][2], tc))
            else:
                merged.append((bottom, top, tc))
        price = close[i]
        best = None
        for bottom, top, tc in merged:
            dist = (bottom - price) if kind == "demand" else (price - top)
            if abs(dist) > 2.0 * atr_s[i]:
                continue
            if best is None or abs(dist) < abs(best[0]):
                best = (dist, bottom, top, tc)
        return best

    for i in range(1, n):
        best_d = nearest_zone(demand_cands, i, "demand")
        if best_d is not None:
            near_demand[i] = best_d[0]
            zone_demand_bottom[i] = best_d[1]
            demand_fresh[i] = best_d[3] == 0
        best_s = nearest_zone(supply_cands, i, "supply")
        if best_s is not None:
            near_supply[i] = best_s[0]
            zone_supply_top[i] = best_s[2]
            supply_fresh[i] = best_s[3] == 0

    buy_score = np.zeros(n, int)
    sell_score = np.zeros(n, int)

    has_demand = ~np.isnan(near_demand)
    has_supply = ~np.isnan(near_supply)
    buy_score[has_demand] += 4
    sell_score[has_supply] += 4
    buy_score[has_demand & demand_fresh] += 2
    sell_score[has_supply & supply_fresh] += 2

    buy_score[sweep_bull] += 2
    sell_score[sweep_bear] += 2
    buy_score += np.maximum(momentum, 0)
    sell_score += np.maximum(-momentum, 0)

    vw_buy = (vw_mom > 0).astype(int) * np.maximum(1, np.round(np.abs(vw_mom) * 2).astype(int))
    vw_sell = (vw_mom < 0).astype(int) * np.maximum(1, np.round(np.abs(vw_mom) * 2).astype(int))
    buy_score += vw_buy
    sell_score += vw_sell

    climax_sell = climax & (buy_score >= sell_score)
    climax_buy = climax & (sell_score > buy_score)
    buy_score[climax_buy] += 1
    sell_score[climax_sell] += 1

    buy_score[vwap_bias > 0] += 1
    sell_score[vwap_bias < 0] += 1

    buy_score[bullish_f] += 1
    sell_score[bearish_f] += 1

    threshold = cfg.get("signal_threshold", 2)
    strong = cfg.get("strong_threshold", 4)
    sl_buffer = cfg.get("sl_atr_buffer", 1.0)
    rr = cfg.get("risk_reward", 2.0)

    signal_buy = (buy_score >= threshold) & (buy_score > sell_score)
    signal_sell = (sell_score >= threshold) & (sell_score > buy_score)

    if cfg.get("bias_enabled", True) and bias is not None and len(bias) == n:
        bias_arr = np.asarray(bias, dtype=int)
        signal_buy = signal_buy & (bias_arr >= 0)
        signal_sell = signal_sell & (bias_arr <= 0)

    signals = np.full(n, "NONE", dtype=object)
    signals[signal_buy & (buy_score >= strong)] = "STRONG_BUY"
    signals[signal_buy & (buy_score < strong)] = "BUY"
    signals[signal_sell & (sell_score >= strong)] = "STRONG_SELL"
    signals[signal_sell & (sell_score < strong)] = "SELL"

    sl = np.full(n, np.nan)
    tp = np.full(n, np.nan)

    for i in np.where(signal_buy)[0]:
        p = close[i]
        slv = sweep_bull_level[i]
        if not np.isnan(slv):
            zl = zone_demand_bottom[i]
            ref = min(slv, zl if not np.isnan(zl) else p)
            sl[i] = ref - sl_buffer * atr_s[i]
        elif not np.isnan(zone_demand_bottom[i]):
            sl[i] = zone_demand_bottom[i] - sl_buffer * atr_s[i]
        else:
            sl[i] = p - 2 * sl_buffer * atr_s[i]
        tp[i] = p + rr * (p - sl[i])

    for i in np.where(signal_sell)[0]:
        p = close[i]
        slv = sweep_bear_level[i]
        if not np.isnan(slv):
            zt = zone_supply_top[i]
            ref = max(slv, zt if not np.isnan(zt) else p)
            sl[i] = ref + sl_buffer * atr_s[i]
        elif not np.isnan(zone_supply_top[i]):
            sl[i] = zone_supply_top[i] + sl_buffer * atr_s[i]
        else:
            sl[i] = p + 2 * sl_buffer * atr_s[i]
        tp[i] = p - rr * (sl[i] - p)

    return pd.DataFrame({
        "signal": signals,
        "entry": close,
        "stop_loss": sl,
        "take_profit": tp,
        "buy_score": buy_score,
        "sell_score": sell_score,
    }, index=df.index)
