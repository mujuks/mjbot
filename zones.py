import pandas as pd


def find_pivots(high: pd.Series, low: pd.Series, left: int = 3, right: int = 3) -> list:
    pivots = []
    highs = high.values
    lows = low.values
    n = len(high)
    for i in range(left, n - right):
        if highs[i] == max(highs[i - left:i + right + 1]):
            pivots.append((i, "high", highs[i]))
        if lows[i] == min(lows[i - left:i + right + 1]):
            pivots.append((i, "low", lows[i]))
    return pivots


def detect_zones(df: pd.DataFrame, cfg: dict) -> dict:
    left = cfg.get("zone_left", 3)
    right = cfg.get("zone_right", 3)
    move_atr = cfg.get("zone_move_atr", 1.5)

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    atr = _atr(df, cfg.get("atr_period", 14))
    pivots = find_pivots(high, low, left, right)
    n = len(df)

    supply = []
    demand = []

    for idx, kind, _ in pivots:
        if idx + right + 1 >= n:
            continue
        start = max(0, idx - left)
        base_high = float(high.iloc[start:idx].max())
        base_low = float(low.iloc[start:idx].min())

        if kind == "low":
            ahead = close.iloc[min(idx + 1 + right, n - 1)]
            move = ahead - low.iloc[idx]
            if move > move_atr * atr.iloc[idx]:
                touches = _count_touches(low, idx + 1, n, base_high)
                demand.append(
                    {
                        "type": "demand",
                        "top": float(base_high),
                        "bottom": float(base_low),
                        "index": idx,
                        "strength": 1.0 / (1 + touches),
                        "touches": touches,
                    }
                )
        elif kind == "high":
            ahead = close.iloc[min(idx + 1 + right, n - 1)]
            move = high.iloc[idx] - ahead
            if move > move_atr * atr.iloc[idx]:
                touches = _count_touches(high, idx + 1, n, base_low)
                supply.append(
                    {
                        "type": "supply",
                        "top": float(base_high),
                        "bottom": float(base_low),
                        "index": idx,
                        "strength": 1.0 / (1 + touches),
                        "touches": touches,
                    }
                )

    demand = _dedupe(demand)
    supply = _dedupe(supply)
    return {"demand": sorted(demand, key=lambda z: z["index"], reverse=True),
            "supply": sorted(supply, key=lambda z: z["index"], reverse=True)}


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
