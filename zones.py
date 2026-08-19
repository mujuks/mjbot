import numpy as np
import pandas as pd


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


def detect_zones(df: pd.DataFrame, cfg: dict) -> dict:
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

    for idx, kind, _ in pivots:
        if idx + right + 1 >= n:
            continue
        start = max(0, idx - left)
        base_high = float(high.iloc[start:idx].max()) if idx > start else float(high.iloc[idx])
        base_low = float(low.iloc[start:idx].min()) if idx > start else float(low.iloc[idx])

        if kind == "low":
            ahead = close.iloc[min(idx + 1 + right, n - 1)]
            move = ahead - low.iloc[idx]
            if move > move_atr * atr.iloc[idx]:
                touches = _count_touches(low, idx + 1, n, base_high)
                mitigated = _check_mitigated(close, idx, n, base_high, direction="demand")
                impulse_strength = min(1.0, move / (2 * atr.iloc[idx]))
                avg_vol = volume.iloc[max(0, idx - 20):idx].mean() if idx > 0 else 0
                vol_at_zone = volume.iloc[idx]
                high_volume = vol_at_zone > avg_vol * 1.3 if avg_vol > 0 else False

                zone_mid = (base_high + base_low) / 2
                in_discount = zone_mid < range_mid

                demand.append({
                    "type": "demand",
                    "top": base_high,
                    "bottom": base_low,
                    "index": idx,
                    "strength": 1.0 / (1 + touches),
                    "touches": touches,
                    "mitigated": mitigated,
                    "impulse_strength": round(impulse_strength, 3),
                    "high_volume": high_volume,
                    "in_discount": in_discount,
                    "in_premium": not in_discount,
                })

        elif kind == "high":
            ahead = close.iloc[min(idx + 1 + right, n - 1)]
            move = high.iloc[idx] - ahead
            if move > move_atr * atr.iloc[idx]:
                touches = _count_touches(high, idx + 1, n, base_low)
                mitigated = _check_mitigated(close, idx, n, base_low, direction="supply")
                impulse_strength = min(1.0, move / (2 * atr.iloc[idx]))
                avg_vol = volume.iloc[max(0, idx - 20):idx].mean() if idx > 0 else 0
                vol_at_zone = volume.iloc[idx]
                high_volume = vol_at_zone > avg_vol * 1.3 if avg_vol > 0 else False

                zone_mid = (base_high + base_low) / 2
                in_premium = zone_mid > range_mid

                supply.append({
                    "type": "supply",
                    "top": base_high,
                    "bottom": base_low,
                    "index": idx,
                    "strength": 1.0 / (1 + touches),
                    "touches": touches,
                    "mitigated": mitigated,
                    "impulse_strength": round(impulse_strength, 3),
                    "high_volume": high_volume,
                    "in_premium": in_premium,
                    "in_discount": not in_premium,
                })

    demand = _dedupe(demand)
    supply = _dedupe(supply)
    return {"demand": sorted(demand, key=lambda z: z["index"], reverse=True),
            "supply": sorted(supply, key=lambda z: z["index"], reverse=True),
            "range_mid": range_mid}


def _check_mitigated(close: pd.Series, zone_idx: int, n: int, level: float, direction: str) -> bool:
    for i in range(zone_idx + 1, min(zone_idx + 100, n)):
        if direction == "demand" and close.iloc[i] < level:
            return True
        if direction == "supply" and close.iloc[i] > level:
            return True
    return False


def _count_touches(series: pd.Series, start: int, end: int, level: float) -> int:
    count = 0
    for i in range(start, end):
        if series.iloc[i] <= level:
            count += 1
    return count


def _dedupe(zones: list) -> list:
    merged = []
    for zone in sorted(zones, key=lambda z: z["bottom"]):
        if merged and zone["bottom"] < merged[-1]["top"]:
            merged[-1]["top"] = max(merged[-1]["top"], zone["top"])
            merged[-1]["bottom"] = min(merged[-1]["bottom"], zone["bottom"])
            merged[-1]["strength"] = max(merged[-1]["strength"], zone["strength"])
            merged[-1]["touches"] = min(merged[-1]["touches"], zone["touches"])
            merged[-1]["mitigated"] = merged[-1]["mitigated"] and zone.get("mitigated", False)
            merged[-1]["impulse_strength"] = max(merged[-1].get("impulse_strength", 0),
                                                  zone.get("impulse_strength", 0))
        else:
            merged.append(dict(zone))
    return merged


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()
