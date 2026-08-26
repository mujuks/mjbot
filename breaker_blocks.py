import numpy as np
import pandas as pd


def detect_breaker_blocks(df: pd.DataFrame, order_blocks: dict, structure: dict, cfg: dict) -> dict:
    """
    Detect breaker blocks — OBs that were invalidated (broken through in the
    opposite direction) and now act as reversal zones.

    Bullish breaker: a bearish OB (supply) that was broken above — price failed
    to hold below it, so it becomes a demand zone for longs.

    Bearish breaker: a bullish OB (demand) that was broken below — price failed
    to hold above it, so it becomes a supply zone for shorts.

    Key difference from mitigation blocks:
    - Mitigation block = OB was mitigated (traded back through) and now holds as S/R
    - Breaker block = OB was *invalidated* by a strong move against it (BOS/CHoCH),
      meaning institutional orders reversed, and the failed level becomes reversal fuel.
    """
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values
    n = len(df)
    atr_period = cfg.get("atr_period", 14)

    from zones import _atr
    atr_vals = _atr(df, atr_period).values

    max_displacement_atr = cfg.get("breaker_displacement_atr", 1.0)
    max_lookback = cfg.get("breaker_lookback", 80)

    bullish_breakers = []
    bearish_breakers = []

    swing_highs = structure.get("swing_highs", [])
    swing_lows = structure.get("swing_lows", [])

    for ob in order_blocks.get("bullish", []):
        if ob.get("mitigated", False):
            idx = ob["index"]
            ob_top = ob["top"]
            ob_bottom = ob["bottom"]
            ob_mid = (ob_top + ob_bottom) / 2

            for i in range(idx + 1, min(idx + max_lookback, n)):
                displacement = opens[i] - closes[i]
                candle_range = highs[i] - lows[i]
                if candle_range <= 0:
                    continue
                if closes[i] < ob_bottom and displacement > max_displacement_atr * atr_vals[i]:
                    bos_event = False
                    for sl in swing_lows:
                        if sl["index"] > i and sl["index"] <= i + 6:
                            if closes[min(sl["index"], n - 1)] < sl["price"]:
                                bos_event = True
                                break

                    vol_avg = np.mean(df["Volume"].values[max(0, i - 20):i]) if i > 0 else 0
                    vol_now = df["Volume"].values[i] if "Volume" in df.columns else 0
                    high_volume = vol_now > vol_avg * 1.3 if vol_avg > 0 else False

                    bearish_breakers.append({
                        "index": idx,
                        "break_index": i,
                        "top": ob_top,
                        "bottom": ob_bottom,
                        "mid": float(ob_mid),
                        "direction": "bearish",
                        "strength": min(1.0, displacement / (max_displacement_atr * atr_vals[i])) if atr_vals[i] > 0 else 0.5,
                        "bos_confirmed": bos_event,
                        "high_volume": high_volume,
                        "mitigated": False,
                        "touches": 0,
                    })
                    break

    for ob in order_blocks.get("bearish", []):
        if ob.get("mitigated", False):
            idx = ob["index"]
            ob_top = ob["top"]
            ob_bottom = ob["bottom"]
            ob_mid = (ob_top + ob_bottom) / 2

            for i in range(idx + 1, min(idx + max_lookback, n)):
                displacement = closes[i] - opens[i]
                candle_range = highs[i] - lows[i]
                if candle_range <= 0:
                    continue
                if closes[i] > ob_top and displacement > max_displacement_atr * atr_vals[i]:
                    bos_event = False
                    for sh in swing_highs:
                        if sh["index"] > i and sh["index"] <= i + 6:
                            if closes[min(sh["index"], n - 1)] > sh["price"]:
                                bos_event = True
                                break

                    vol_avg = np.mean(df["Volume"].values[max(0, i - 20):i]) if i > 0 else 0
                    vol_now = df["Volume"].values[i] if "Volume" in df.columns else 0
                    high_volume = vol_now > vol_avg * 1.3 if vol_avg > 0 else False

                    bullish_breakers.append({
                        "index": idx,
                        "break_index": i,
                        "top": ob_top,
                        "bottom": ob_bottom,
                        "mid": float(ob_mid),
                        "direction": "bullish",
                        "strength": min(1.0, displacement / (max_displacement_atr * atr_vals[i])) if atr_vals[i] > 0 else 0.5,
                        "bos_confirmed": bos_event,
                        "high_volume": high_volume,
                        "mitigated": False,
                        "touches": 0,
                    })
                    break

    _count_breaker_touches(bullish_breakers, lows, n, supply=False)
    _count_breaker_touches(bearish_breakers, highs, n, supply=True)
    _mark_breaker_mitigated(bullish_breakers, closes, n, direction="bullish")
    _mark_breaker_mitigated(bearish_breakers, closes, n, direction="bearish")

    bullish_breakers.sort(key=lambda z: z["index"], reverse=True)
    bearish_breakers.sort(key=lambda z: z["index"], reverse=True)

    return {"bullish": bullish_breakers, "bearish": bearish_breakers}


def find_nearest_breaker(breakers: list, price: float, atr_val: float, cfg: dict) -> dict | None:
    max_dist = cfg.get("breaker_max_dist_atr", 2.0) * atr_val
    best = None
    for bb in breakers:
        dist = price - bb["mid"]
        if abs(dist) <= max_dist and not bb.get("mitigated", False):
            max_touches = cfg.get("breaker_max_touches", 5)
            if bb["touches"] <= max_touches:
                if best is None or abs(dist) < abs(best["_dist"]):
                    bb["_dist"] = dist
                    best = bb
    if best is not None:
        best.pop("_dist", None)
    return best


def _count_breaker_touches(breakers: list, levels: np.ndarray, n: int, supply: bool):
    for bb in breakers:
        idx = bb["break_index"]
        count = 0
        for i in range(idx + 1, n):
            if supply:
                if levels[i] >= bb["bottom"]:
                    count += 1
            else:
                if levels[i] <= bb["top"]:
                    count += 1
        bb["touches"] = count


def _mark_breaker_mitigated(breakers: list, closes: np.ndarray, n: int, direction: str):
    for bb in breakers:
        idx = bb["break_index"]
        mitigated = False
        for i in range(idx + 1, min(idx + 100, n)):
            if direction == "bullish" and closes[i] < bb["bottom"]:
                mitigated = True
                break
            if direction == "bearish" and closes[i] > bb["top"]:
                mitigated = True
                break
        bb["mitigated"] = mitigated
