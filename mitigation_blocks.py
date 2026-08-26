import numpy as np
import pandas as pd


def detect_mitigation_blocks(df: pd.DataFrame, order_blocks: dict, cfg: dict) -> dict:
    """
    Detect mitigation blocks — OBs that were broken through and then price
    returns to retest them from the other side. These become institutional
    reference points (support/resistance).

    A bullish mitigation block was a bearish OB that was broken above, now
    acts as support on retest.
    A bearish mitigation block was a bullish OB that was broken below, now
    acts as resistance on retest.
    """
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    atr_period = cfg.get("atr_period", 14)
    max_atr = cfg.get("mitigation_max_dist_atr", 1.5)

    from zones import _atr
    atr_vals = _atr(df, atr_period).values

    mitigation_bullish = []
    mitigation_bearish = []

    for ob in order_blocks.get("bullish", []):
        if ob.get("mitigated", False):
            idx = ob["index"]
            ob_top = ob["top"]
            ob_bottom = ob["bottom"]
            ob_mid = (ob_top + ob_bottom) / 2

            for i in range(idx + 1, min(idx + 80, n)):
                if closes[i] < ob_bottom:
                    retest_price = ob_mid
                    for j in range(i + 1, min(i + 30, n)):
                        dist = abs(lows[j] - retest_price)
                        if dist <= max_atr * atr_vals[j]:
                            mitigation_bullish.append({
                                "index": idx,
                                "break_index": i,
                                "retest_index": j,
                                "top": ob_top,
                                "bottom": ob_bottom,
                                "mid": float(ob_mid),
                                "direction": "bullish",
                                "strength": ob.get("strength", 0.5),
                                "bos_confirmed": ob.get("bos_confirmed", False),
                                "mitigated": False,
                                "touches": 0,
                            })
                            break
                    break

    for ob in order_blocks.get("bearish", []):
        if ob.get("mitigated", False):
            idx = ob["index"]
            ob_top = ob["top"]
            ob_bottom = ob["bottom"]
            ob_mid = (ob_top + ob_bottom) / 2

            for i in range(idx + 1, min(idx + 80, n)):
                if closes[i] > ob_top:
                    retest_price = ob_mid
                    for j in range(i + 1, min(i + 30, n)):
                        dist = abs(highs[j] - retest_price)
                        if dist <= max_atr * atr_vals[j]:
                            mitigation_bearish.append({
                                "index": idx,
                                "break_index": i,
                                "retest_index": j,
                                "top": ob_top,
                                "bottom": ob_bottom,
                                "mid": float(ob_mid),
                                "direction": "bearish",
                                "strength": ob.get("strength", 0.5),
                                "bos_confirmed": ob.get("bos_confirmed", False),
                                "mitigated": False,
                                "touches": 0,
                            })
                            break
                    break

    _count_mitigation_touches(mitigation_bullish, lows, n, supply=False)
    _count_mitigation_touches(mitigation_bearish, highs, n, supply=True)
    _mark_mitigation_touched(mitigation_bullish, closes, n)
    _mark_mitigation_touched(mitigation_bearish, closes, n)

    mitigation_bullish.sort(key=lambda z: z["index"], reverse=True)
    mitigation_bearish.sort(key=lambda z: z["index"], reverse=True)

    return {"bullish": mitigation_bullish, "bearish": mitigation_bearish}


def find_nearest_mitigation(blocks: list, price: float, atr_val: float, cfg: dict) -> dict | None:
    max_dist = cfg.get("mitigation_max_dist_atr", 1.5) * atr_val
    best = None
    for mb in blocks:
        dist = price - mb["mid"]
        if abs(dist) <= max_dist and not mb.get("mitigated", False):
            if best is None or abs(dist) < abs(best["_dist"]):
                mb["_dist"] = dist
                best = mb
    if best is not None:
        best.pop("_dist", None)
    return best


def _count_mitigation_touches(blocks: list, levels: np.ndarray, n: int, supply: bool):
    for mb in blocks:
        idx = mb["retest_index"]
        count = 0
        for i in range(idx + 1, n):
            if supply:
                if levels[i] >= mb["bottom"]:
                    count += 1
            else:
                if levels[i] <= mb["top"]:
                    count += 1
        mb["touches"] = count


def _mark_mitigation_touched(blocks: list, closes: np.ndarray, n: int):
    for mb in blocks:
        idx = mb["retest_index"]
        touched = False
        for i in range(idx + 1, min(idx + 60, n)):
            if mb["direction"] == "bullish" and closes[i] < mb["bottom"]:
                touched = True
                break
            if mb["direction"] == "bearish" and closes[i] > mb["top"]:
                touched = True
                break
        mb["mitigated"] = touched
