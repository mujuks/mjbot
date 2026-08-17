import numpy as np
import pandas as pd


def detect_order_blocks(df: pd.DataFrame, cfg: dict) -> dict:
    """Detect bullish and bearish order blocks.

    A bullish order block is the last bearish candle before a strong
    bullish impulse move.  A bearish order block is the last bullish
    candle before a strong bearish impulse move.

    Returns dict with keys:
        bullish: list of dicts with {index, top, bottom, strength, touches}
        bearish: list of dicts with {index, top, bottom, strength, touches}
    """
    atr_mult = cfg.get("ob_move_atr", 2.0)
    lookback = cfg.get("ob_lookback", 50)
    min_move = cfg.get("ob_min_move", 2)

    from zones import _atr

    atr = _atr(df, cfg.get("atr_period", 14)).values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)

    bullish_obs = []
    bearish_obs = []

    start = max(3, n - lookback)

    for i in range(start, n - min_move):
        body = closes[i] - opens[i]
        is_bearish = body < 0
        is_bullish = body > 0

        if is_bearish:
            move_up = 0
            max_move = 0
            for j in range(i + 1, min(i + 1 + 6, n)):
                move_up += closes[j] - opens[j]
                max_move = max(max_move, closes[j] - lows[i])
            if max_move > atr_mult * atr[i] and move_up > 0:
                bullish_obs.append({
                    "index": i,
                    "top": float(highs[i]),
                    "bottom": float(lows[i]),
                    "strength": min(1.0, max_move / (atr_mult * atr[i])),
                    "touches": 0,
                })

        if is_bullish:
            move_down = 0
            max_move = 0
            for j in range(i + 1, min(i + 1 + 6, n)):
                move_down += opens[j] - closes[j]
                max_move = max(max_move, highs[i] - closes[j])
            if max_move > atr_mult * atr[i] and move_down > 0:
                bearish_obs.append({
                    "index": i,
                    "top": float(highs[i]),
                    "bottom": float(lows[i]),
                    "strength": min(1.0, max_move / (atr_mult * atr[i])),
                    "touches": 0,
                })

    bullish_obs = _merge_obs(bullish_obs)
    bearish_obs = _merge_obs(bearish_obs)

    _count_ob_touches(bullish_obs, lows, n)
    _count_ob_touches(bearish_obs, highs, n, supply=True)

    bullish_obs.sort(key=lambda z: z["index"], reverse=True)
    bearish_obs.sort(key=lambda z: z["index"], reverse=True)

    return {"bullish": bullish_obs, "bearish": bearish_obs}


def find_nearest_ob(obs: list, price: float, atr_val: float, cfg: dict) -> dict | None:
    """Find the nearest order block to the current price within ATR range."""
    max_dist = cfg.get("ob_max_dist_atr", 1.5) * atr_val
    best = None
    for ob in obs:
        dist = price - ob["bottom"] if ob in [o for o in obs] else price - ob["top"]
        dist = price - ((ob["bottom"] + ob["top"]) / 2)
        if abs(dist) <= max_dist:
            max_touches = cfg.get("ob_max_touches", 5)
            if ob["touches"] <= max_touches:
                if best is None or abs(dist) < abs(best["_dist"]):
                    ob["_dist"] = dist
                    best = ob
    if best is not None:
        best.pop("_dist", None)
    return best


def _merge_obs(obs: list) -> list:
    if not obs:
        return obs
    obs.sort(key=lambda z: z["bottom"])
    merged = [obs[0]]
    for ob in obs[1:]:
        if ob["bottom"] <= merged[-1]["top"]:
            merged[-1]["top"] = max(merged[-1]["top"], ob["top"])
            merged[-1]["bottom"] = min(merged[-1]["bottom"], ob["bottom"])
            merged[-1]["strength"] = max(merged[-1]["strength"], ob["strength"])
        else:
            merged.append(dict(ob))
    return merged


def _count_ob_touches(obs: list, levels: np.ndarray, n: int, supply: bool = False):
    for ob in obs:
        idx = ob["index"]
        count = 0
        for i in range(idx + 1, n):
            if supply:
                if levels[i] >= ob["bottom"]:
                    count += 1
            else:
                if levels[i] <= ob["top"]:
                    count += 1
        ob["touches"] = count
