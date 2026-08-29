"""MJBot Zone Detection Engine -- XAUUSD Reaction Zone Training Implementation.

Detects supply/demand zones using institutional order flow principles:
- Impulsive departure detection
- Base formation analysis
- Multiple historical reaction counting
- Liquidity interaction scoring
- Market structure confirmation
- 0-100 strength scoring system
- Zone status tracking (FAR/APPROACHING/AT_ZONE/INSIDE/REJECTING/BREAKING/BROKEN)
- Fresh vs mitigated tracking
- Multi-timeframe zone clustering
"""

import numpy as np
import pandas as pd


# ============================================================================
# ATR HELPER
# ============================================================================

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ============================================================================
# PIVOT DETECTION
# ============================================================================

def find_pivots(high: pd.Series, low: pd.Series, left: int = 3, right: int = 3) -> list:
    pivots = []
    highs = high.values
    lows = low.values
    n = len(high)
    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        window_l = lows[i - left:i + right + 1]
        if highs[i] == np.max(window_h) and np.sum(window_h == highs[i]) == 1:
            pivots.append((i, "high", highs[i]))
        if lows[i] == np.min(window_l) and np.sum(window_l == lows[i]) == 1:
            pivots.append((i, "low", lows[i]))
    return pivots


# ============================================================================
# DISPLACEMENT DETECTION
# ============================================================================

def _detect_displacement(df: pd.DataFrame, start_idx: int, direction: str,
                         atr: pd.Series, cfg: dict) -> dict:
    """Detect impulsive departure from a zone.
    Returns displacement strength, length, and quality metrics."""
    n = len(df)
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values if "Volume" in df.columns else np.zeros(n)

    lookback = cfg.get("zone_displacement_lookback", 10)
    end_idx = min(start_idx + lookback, n)
    if end_idx - start_idx < 2:
        return {"strength": 0, "length": 0, "candles": 0, "body_ratio": 0}

    seg_close = close[start_idx:end_idx]
    seg_high = high[start_idx:end_idx]
    seg_low = low[start_idx:end_idx]
    seg_vol = volume[start_idx:end_idx]
    seg_atr = atr.iloc[start_idx:end_idx].values

    if direction == "demand":
        move = seg_close[-1] - seg_close[0]
        bullish_candles = sum(1 for i in range(1, len(seg_close))
                             if seg_close[i] > seg_close[i-1])
        max_body = max(seg_close[i] - df["Open"].values[start_idx + i]
                       for i in range(len(seg_close)))
    else:
        move = seg_close[0] - seg_close[-1]
        bearish_candles = sum(1 for i in range(1, len(seg_close))
                              if seg_close[i] < seg_close[i-1])
        max_body = max(df["Open"].values[start_idx + i] - seg_close[i]
                       for i in range(len(seg_close)))

    avg_atr = np.mean(seg_atr) if len(seg_atr) > 0 else 1.0
    displacement_atr = move / avg_atr if avg_atr > 0 else 0

    body_range_ratios = []
    for i in range(len(seg_close)):
        o = df["Open"].values[start_idx + i]
        c = seg_close[i]
        h = seg_high[i]
        l = seg_low[i]
        rng = h - l if h > l else 0.001
        body = abs(c - o)
        body_range_ratios.append(body / rng)
    avg_body_ratio = np.mean(body_range_ratios) if body_range_ratios else 0

    vol_avg = np.mean(seg_vol) if len(seg_vol) > 0 else 0
    vol_surge = seg_vol[-1] > vol_avg * 1.5 if vol_avg > 0 else False

    strength = min(1.0, displacement_atr / 3.0)

    return {
        "strength": round(strength, 3),
        "displacement_atr": round(displacement_atr, 2),
        "length": len(seg_close),
        "candles": bullish_candles if direction == "demand" else bearish_candles,
        "body_ratio": round(avg_body_ratio, 3),
        "vol_surge": vol_surge,
    }


# ============================================================================
# REACTION COUNTING
# ============================================================================

def _count_meaningful_reactions(df: pd.DataFrame, zone_low: float, zone_high: float,
                                start_idx: int, direction: str, atr: pd.Series,
                                cfg: dict) -> int:
    """Count meaningful reactions from a zone.
    A reaction requires measurable rejection, displacement, or structural movement."""
    n = len(df)
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    reactions = 0
    min_reaction_atr = cfg.get("zone_min_reaction_atr", 0.5)

    i = start_idx + 1
    while i < n - 1:
        if direction == "demand":
            if low[i] <= zone_high and low[i] >= zone_low - 0.2 * atr.iloc[i]:
                reaction_move = close[i] - low[i]
                if reaction_move > min_reaction_atr * atr.iloc[i]:
                    reactions += 1
                    i += 3
                    continue
        elif direction == "supply":
            if high[i] >= zone_low and high[i] <= zone_high + 0.2 * atr.iloc[i]:
                reaction_move = high[i] - close[i]
                if reaction_move > min_reaction_atr * atr.iloc[i]:
                    reactions += 1
                    i += 3
                    continue
        i += 1

    return reactions


# ============================================================================
# LIQUIDITY SWEEP DETECTION
# ============================================================================

def _detect_liquidity_sweep_at_zone(df: pd.DataFrame, zone_idx: int,
                                     zone_low: float, zone_high: float,
                                     direction: str, cfg: dict) -> bool:
    """Detect if a liquidity sweep occurred before the zone formed."""
    n = len(df)
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    left = cfg.get("swing_left", 3)

    lookback = min(zone_idx, 20)
    if lookback < left + 1:
        return False

    if direction == "demand":
        for i in range(zone_idx - lookback, zone_idx):
            if i < left:
                continue
            window = low[i - left:i + left + 1]
            if low[i] == np.min(window):
                if low[i] < zone_low:
                    return True
    else:
        for i in range(zone_idx - lookback, zone_idx):
            if i < left:
                continue
            window = high[i - left:i + left + 1]
            if high[i] == np.max(window):
                if high[i] > zone_high:
                    return True

    return False


# ============================================================================
# STRUCTURE CONFIRMATION
# ============================================================================

def _check_structure_confirmation(df: pd.DataFrame, zone_idx: int,
                                   direction: str, cfg: dict) -> bool:
    """Check if market structure confirms the zone."""
    n = len(df)
    close = df["Close"].values
    left = cfg.get("swing_left", 3)
    right = cfg.get("swing_right", 3)

    lookforward = min(n - zone_idx, 30)
    if lookforward < left + right + 1:
        return False

    seg_close = close[zone_idx:zone_idx + lookforward]

    if direction == "demand":
        swing_lows = []
        for i in range(left, len(seg_close) - right):
            window = seg_close[i - left:i + right + 1]
            if seg_close[i] == np.min(window):
                swing_lows.append((i, seg_close[i]))
        if len(swing_lows) >= 2:
            if swing_lows[-1][1] > swing_lows[-2][1]:
                return True
    else:
        swing_highs = []
        for i in range(left, len(seg_close) - right):
            window = seg_close[i - left:i + right + 1]
            if seg_close[i] == np.max(window):
                swing_highs.append((i, seg_close[i]))
        if len(swing_highs) >= 2:
            if swing_highs[-1][1] < swing_highs[-2][1]:
                return True

    return False


# ============================================================================
# ZONE STRENGTH SCORING (0-100)
# ============================================================================

def _compute_zone_strength(impulse: dict, reaction_count: int,
                           liquidity_sweep: bool, structure_confirmed: bool,
                           vol_surge: bool, freshness: float,
                           htf_confirmed: bool = False) -> int:
    """Compute zone strength score 0-100 based on training specification.

    Components:
    - Impulsive displacement: 25 points
    - Number of meaningful reactions: 20 points
    - Liquidity sweep: 15 points
    - Market-structure confirmation: 15 points
    - Volume/momentum confirmation: 10 points
    - Freshness of zone: 10 points
    - Higher-timeframe confirmation: 5 points
    """
    score = 0

    # Impulsive displacement (25 points)
    score += int(impulse["strength"] * 25)

    # Meaningful reactions (20 points)
    if reaction_count >= 4:
        score += 20
    elif reaction_count == 3:
        score += 15
    elif reaction_count == 2:
        score += 10
    elif reaction_count == 1:
        score += 5

    # Liquidity sweep (15 points)
    if liquidity_sweep:
        score += 15

    # Market-structure confirmation (15 points)
    if structure_confirmed:
        score += 15

    # Volume/momentum confirmation (10 points)
    if vol_surge:
        score += 10
    elif impulse.get("body_ratio", 0) > 0.6:
        score += 5

    # Freshness (10 points)
    score += int(freshness * 10)

    # Higher-timeframe confirmation (5 points)
    if htf_confirmed:
        score += 5

    return max(0, min(100, score))


def _zone_strength_label(score: int) -> str:
    if score >= 90:
        return "MAJOR"
    elif score >= 75:
        return "VERY_STRONG"
    elif score >= 60:
        return "STRONG"
    elif score >= 40:
        return "MODERATE"
    else:
        return "WEAK"


# ============================================================================
# ZONE STATUS CLASSIFICATION
# ============================================================================

def _classify_zone_status(price: float, zone_low: float, zone_high: float,
                          atr_val: float, direction: str,
                          prev_close: float = None) -> str:
    """Classify zone status based on price-to-zone distance."""
    mid = (zone_low + zone_high) / 2
    zone_width = zone_high - zone_low

    if zone_low <= price <= zone_high:
        return "INSIDE_ZONE"

    dist = price - mid
    dist_atr = abs(dist) / atr_val if atr_val > 0 else 999

    if dist_atr < 0.3:
        return "AT_ZONE"

    if direction == "demand":
        if 0 < dist_atr < 2.0:
            return "APPROACHING"
        elif dist_atr >= 2.0:
            return "FAR"
        elif dist_atr < 0 and dist_atr > -0.5:
            if prev_close and prev_close > price:
                return "REJECTING"
            return "AT_ZONE"
        elif dist_atr <= -0.5:
            if prev_close and prev_close > zone_high:
                return "BREAKING"
            return "BROKEN"
    else:
        if 0 < dist_atr < 2.0:
            return "APPROACHING"
        elif dist_atr >= 2.0:
            return "FAR"
        elif dist_atr < 0 and dist_atr > -0.5:
            if prev_close and prev_close < price:
                return "REJECTING"
            return "AT_ZONE"
        elif dist_atr <= -0.5:
            if prev_close and prev_close < zone_low:
                return "BREAKING"
            return "BROKEN"

    return "FAR"


# ============================================================================
# ZONE INVALIDATION
# ============================================================================

def _check_zone_invalidated(df: pd.DataFrame, zone_idx: int,
                             zone_low: float, zone_high: float,
                             direction: str, cfg: dict) -> bool:
    """Check if zone has been decisively broken (invalidated).
    A temporary wick through does NOT invalidate."""
    n = len(df)
    close = df["Close"].values
    accept_above = cfg.get("zone_invalidation_candles", 2)

    remaining = n - zone_idx - 1
    if remaining < accept_above:
        return False

    seg_close = close[zone_idx + 1:zone_idx + 1 + min(remaining, 20)]

    if direction == "demand":
        broken_count = sum(1 for c in seg_close if c < zone_low)
        return broken_count >= accept_above
    else:
        broken_count = sum(1 for c in seg_close if c > zone_high)
        return broken_count >= accept_above


# ============================================================================
# ZONE TOUCH COUNTING (ENHANCED)
# ============================================================================

def _count_zone_revisits(df: pd.DataFrame, zone_idx: int,
                          zone_low: float, zone_high: float,
                          direction: str, atr: pd.Series) -> list:
    """Track each zone revisit with details."""
    n = len(df)
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    revisits = []

    i = zone_idx + 1
    while i < n:
        if direction == "demand":
            if low[i] <= zone_high + 0.1 * atr.iloc[i]:
                close_above = close[i] > zone_high
                revisits.append({
                    "index": i,
                    "wick_touch": low[i] <= zone_high,
                    "close_above": close_above,
                    "mitigated": not close_above,
                })
                i += 2
                continue
        else:
            if high[i] >= zone_low - 0.1 * atr.iloc[i]:
                close_below = close[i] < zone_low
                revisits.append({
                    "index": i,
                    "wick_touch": high[i] >= zone_low,
                    "close_above": close_below,
                    "mitigated": not close_below,
                })
                i += 2
                continue
        i += 1

    return revisits


# ============================================================================
# MAIN ZONE DETECTION
# ============================================================================

def detect_zones(df: pd.DataFrame, cfg: dict) -> dict:
    """Enhanced zone detection with 0-100 strength scoring.
    Implements the XAUUSD Reaction Zone Training specification."""
    left = cfg.get("zone_left", 3)
    right = cfg.get("zone_right", 3)
    move_atr = cfg.get("zone_move_atr", 1.0)

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)
    atr = _atr(df, cfg.get("atr_period", 14))
    pivots = find_pivots(high, low, left, right)
    n = len(df)

    pd_lookback = cfg.get("pd_lookback", 50)
    range_high = float(high.iloc[max(0, n - pd_lookback):].max())
    range_low = float(low.iloc[max(0, n - pd_lookback):].min())
    range_mid = (range_high + range_low) / 2 if range_high != range_low else range_high

    supply = []
    demand = []

    price = float(close.iloc[-1])
    atr_val = float(atr.iloc[-1])
    prev_close = float(close.iloc[-2]) if n > 1 else price

    for idx, kind, _ in pivots:
        if idx + right + 1 >= n:
            continue
        start = max(0, idx - left)
        base_high = float(high.iloc[start:idx].max()) if idx > start else float(high.iloc[idx])
        base_low = float(low.iloc[start:idx].min()) if idx > start else float(low.iloc[idx])

        if base_high == base_low:
            continue

        if kind == "low":
            ahead = close.iloc[min(idx + 1 + right, n - 1)]
            move = ahead - low.iloc[idx]
            if move > move_atr * atr.iloc[idx]:
                zone = _build_zone(df, idx, base_low, base_high, "demand",
                                   atr, volume, range_mid, cfg, price, atr_val, prev_close)
                if zone:
                    demand.append(zone)

        elif kind == "high":
            ahead = close.iloc[min(idx + 1 + right, n - 1)]
            move = high.iloc[idx] - ahead
            if move > move_atr * atr.iloc[idx]:
                zone = _build_zone(df, idx, base_low, base_high, "supply",
                                   atr, volume, range_mid, cfg, price, atr_val, prev_close)
                if zone:
                    supply.append(zone)

    demand = _dedupe(demand)
    supply = _dedupe(supply)

    demand = [z for z in demand if not z.get("invalidated", False)]
    supply = [z for z in supply if not z.get("invalidated", False)]

    return {
        "demand": sorted(demand, key=lambda z: z["strength_score"], reverse=True),
        "supply": sorted(supply, key=lambda z: z["strength_score"], reverse=True),
        "range_mid": range_mid,
    }


def _build_zone(df: pd.DataFrame, idx: int, base_low: float, base_high: float,
                direction: str, atr: pd.Series, volume: pd.Series,
                range_mid: float, cfg: dict, price: float, atr_val: float,
                prev_close: float) -> dict | None:
    """Build a complete zone object with all training specification fields."""
    n = len(df)
    close = df["Close"].values

    impulse = _detect_displacement(df, idx, direction, atr, cfg)
    if impulse["strength"] < 0.1:
        return None

    reaction_count = _count_meaningful_reactions(df, base_low, base_high, idx, direction, atr, cfg)

    liq_sweep = _detect_liquidity_sweep_at_zone(df, idx, base_low, base_high, direction, cfg)

    structure_confirmed = _check_structure_confirmation(df, idx, direction, cfg)

    vol_at_zone = volume.iloc[idx] if idx < len(volume) else 0
    avg_vol = volume.iloc[max(0, idx - 20):idx].mean() if idx > 0 else 0
    vol_surge = vol_at_zone > avg_vol * 1.5 if avg_vol > 0 else False

    freshness = 1.0 - min(1.0, (n - idx) / 100.0)

    zone_mid = (base_high + base_low) / 2
    in_discount = zone_mid < range_mid
    in_premium = zone_mid > range_mid

    strength_score = _compute_zone_strength(
        impulse=impulse,
        reaction_count=reaction_count,
        liquidity_sweep=liq_sweep,
        structure_confirmed=structure_confirmed,
        vol_surge=vol_surge,
        freshness=freshness,
    )

    strength_label = _zone_strength_label(strength_score)

    status = _classify_zone_status(price, base_low, base_high, atr_val, direction, prev_close)

    invalidated = _check_zone_invalidated(df, idx, base_low, base_high, direction, cfg)

    revisits = _count_zone_revisits(df, idx, base_low, base_high, direction, atr)
    mitigated = any(r["mitigated"] for r in revisits)

    distance = price - zone_mid
    distance_atr = distance / atr_val if atr_val > 0 else 999

    return {
        "type": direction,
        "top": base_high,
        "bottom": base_low,
        "lo": base_low,
        "hi": base_high,
        "midpoint": round(zone_mid, 2),
        "index": idx,
        "strength_score": strength_score,
        "strength_label": strength_label,
        "strength": strength_score / 100.0,
        "touches": len(revisits),
        "revisits": revisits,
        "mitigated": mitigated,
        "invalidated": invalidated,
        "impulse_strength": impulse["strength"],
        "displacement_atr": impulse.get("displacement_atr", 0),
        "displacement_candles": impulse.get("candles", 0),
        "reaction_count": reaction_count,
        "liquidity_sweep": liq_sweep,
        "structure_confirmed": structure_confirmed,
        "high_volume": vol_surge,
        "body_ratio": impulse.get("body_ratio", 0),
        "in_discount": in_discount,
        "in_premium": in_premium,
        "freshness": round(freshness, 3),
        "fresh": not mitigated and freshness > 0.5,
        "status": status,
        "distance_atr": round(distance_atr, 2),
        "reason": _build_zone_reason(direction, impulse, reaction_count, liq_sweep,
                                     structure_confirmed, strength_label),
    }


def _build_zone_reason(direction: str, impulse: dict, reactions: int,
                       liq_sweep: bool, structure: bool, label: str) -> str:
    parts = []
    if impulse["strength"] > 0.5:
        parts.append(f"strong {impulse.get('displacement_atr', 0):.1f}ATR displacement")
    elif impulse["strength"] > 0.2:
        parts.append(f"moderate displacement")

    if reactions >= 3:
        parts.append(f"{reactions} reactions (major)")
    elif reactions >= 2:
        parts.append(f"{reactions} reactions")
    elif reactions == 1:
        parts.append("1 reaction")

    if liq_sweep:
        parts.append("liquidity sweep")

    if structure:
        parts.append("structure confirmed")

    return f"{label}: {'; '.join(parts)}" if parts else f"{label} zone"


# ============================================================================
# ZONE DEDUPLICATION
# ============================================================================

def _dedupe(zones: list) -> list:
    if not zones:
        return []
    merged = []
    for zone in sorted(zones, key=lambda z: z["bottom"]):
        if merged and zone["bottom"] < merged[-1]["top"]:
            if zone["strength_score"] > merged[-1]["strength_score"]:
                merged[-1] = zone
            else:
                merged[-1]["top"] = max(merged[-1]["top"], zone["top"])
                merged[-1]["bottom"] = min(merged[-1]["bottom"], zone["bottom"])
                merged[-1]["reaction_count"] = max(merged[-1]["reaction_count"],
                                                    zone["reaction_count"])
                if zone.get("liquidity_sweep"):
                    merged[-1]["liquidity_sweep"] = True
                if zone.get("structure_confirmed"):
                    merged[-1]["structure_confirmed"] = True
        else:
            merged.append(dict(zone))
    return merged


# ============================================================================
# MULTI-TIMEFRAME ZONE ANALYSIS
# ============================================================================

def analyze_zones_mtf(zones_by_tf: dict, price: float, atr_val: float) -> dict:
    """Analyze zones across multiple timeframes and cluster overlapping zones.
    zones_by_tf: {timeframe: {"demand": [...], "supply": [...]}}
    Returns clustered zones with MTF strength boost."""
    tf_priority = {"4H": 5, "1H": 4, "45M": 3, "15M": 2, "5M": 1}

    all_demand = []
    all_supply = []

    for tf, zones in zones_by_tf.items():
        priority = tf_priority.get(tf, 1)
        for z in zones.get("demand", []):
            z["timeframe"] = tf
            z["tf_priority"] = priority
            all_demand.append(z)
        for z in zones.get("supply", []):
            z["timeframe"] = tf
            z["tf_priority"] = priority
            all_supply.append(z)

    clustered_demand = _cluster_mtf_zones(all_demand, atr_val)
    clustered_supply = _cluster_mtf_zones(all_supply, atr_val)

    return {
        "demand": sorted(clustered_demand, key=lambda z: z["strength_score"], reverse=True),
        "supply": sorted(clustered_supply, key=lambda z: z["strength_score"], reverse=True),
    }


def _cluster_mtf_zones(zones: list, atr_val: float) -> list:
    """Cluster overlapping zones from different timeframes."""
    if not zones:
        return []

    zones.sort(key=lambda z: z["bottom"])

    clustered = []
    for z in zones:
        merged = False
        for c in clustered:
            overlap = min(z["top"], c["top"]) - max(z["bottom"], c["bottom"])
            if overlap > 0:
                if z["strength_score"] > c["strength_score"]:
                    z["strength_score"] = min(100, z["strength_score"] + 5)
                    z["mtf_confluence"] = c.get("mtf_timeframes", []) + [z.get("timeframe", "")]
                    c.update(z)
                else:
                    c["strength_score"] = min(100, c["strength_score"] + 5)
                    if z.get("timeframe"):
                        c.setdefault("mtf_timeframes", []).append(z["timeframe"])
                merged = True
                break
        if not merged:
            clustered.append(dict(z))

    return clustered
