import numpy as np
import pandas as pd


def detect_rejection_blocks(df: pd.DataFrame, swings: dict, cfg: dict) -> dict:
    """
    Detect rejection blocks — wick extremes at swing points that represent
    institutional rejection of price. These act as strong support/resistance.

    A rejection block is the extreme wick level (high or low) at a swing point
    where price made a sharp reversal with a long wick, indicating institutional
    rejection.

    Bullish rejection block: swing low with a long lower wick — institutions
    rejected lower prices.
    Bearish rejection block: swing high with a long upper wick — institutions
    rejected higher prices.
    """
    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values
    closes = df["Close"].values
    n = len(df)

    from zones import _atr
    atr_vals = _atr(df, cfg.get("atr_period", 14)).values

    min_wick_ratio = cfg.get("rejection_wick_ratio", 1.5)
    max_lookback = cfg.get("rejection_lookback", 50)
    start = max(0, n - max_lookback)

    bullish_rejections = []
    bearish_rejections = []

    for sl in swings.get("lows", []):
        idx = sl["index"]
        if idx < start:
            continue
        candle_high = highs[idx]
        candle_low = lows[idx]
        candle_open = opens[idx]
        candle_close = closes[idx]
        candle_range = candle_high - candle_low
        if candle_range <= 0:
            continue

        lower_wick = min(candle_open, candle_close) - candle_low
        upper_wick = candle_high - max(candle_open, candle_close)
        body = abs(candle_close - candle_open)

        if lower_wick > body * min_wick_ratio and lower_wick > upper_wick * min_wick_ratio:
            wick_ratio = lower_wick / candle_range
            vol_avg = np.mean(df["Volume"].values[max(0, idx - 20):idx]) if idx > 0 else 0
            vol_now = df["Volume"].values[idx] if "Volume" in df.columns else 0
            high_volume = vol_now > vol_avg * 1.2 if vol_avg > 0 else False

            bullish_rejections.append({
                "index": idx,
                "price": float(candle_low),
                "wick_depth": float(lower_wick),
                "wick_ratio": float(wick_ratio),
                "strength": min(1.0, wick_ratio),
                "high_volume": high_volume,
                "mitigated": False,
                "touches": 0,
            })

    for sh in swings.get("highs", []):
        idx = sh["index"]
        if idx < start:
            continue
        candle_high = highs[idx]
        candle_low = lows[idx]
        candle_open = opens[idx]
        candle_close = closes[idx]
        candle_range = candle_high - candle_low
        if candle_range <= 0:
            continue

        upper_wick = candle_high - max(candle_open, candle_close)
        lower_wick = min(candle_open, candle_close) - candle_low
        body = abs(candle_close - candle_open)

        if upper_wick > body * min_wick_ratio and upper_wick > lower_wick * min_wick_ratio:
            wick_ratio = upper_wick / candle_range
            vol_avg = np.mean(df["Volume"].values[max(0, idx - 20):idx]) if idx > 0 else 0
            vol_now = df["Volume"].values[idx] if "Volume" in df.columns else 0
            high_volume = vol_now > vol_avg * 1.2 if vol_avg > 0 else False

            bearish_rejections.append({
                "index": idx,
                "price": float(candle_high),
                "wick_depth": float(upper_wick),
                "wick_ratio": float(wick_ratio),
                "strength": min(1.0, wick_ratio),
                "high_volume": high_volume,
                "mitigated": False,
                "touches": 0,
            })

    _count_rejection_touches(bullish_rejections, lows, n, supply=False)
    _count_rejection_touches(bearish_rejections, highs, n, supply=True)
    _mark_rejection_mitigated(bullish_rejections, closes, n, "bullish")
    _mark_rejection_mitigated(bearish_rejections, closes, n, "bearish")

    bullish_rejections.sort(key=lambda z: z["index"], reverse=True)
    bearish_rejections.sort(key=lambda z: z["index"], reverse=True)

    return {"bullish": bullish_rejections, "bearish": bearish_rejections}


def find_nearest_rejection(rejections: list, price: float, atr_val: float, cfg: dict) -> dict | None:
    max_dist = cfg.get("rejection_max_dist_atr", 1.5) * atr_val
    best = None
    for rb in rejections:
        dist = price - rb["price"]
        if abs(dist) <= max_dist and not rb.get("mitigated", False):
            if rb["touches"] <= cfg.get("rejection_max_touches", 5):
                if best is None or abs(dist) < abs(best["_dist"]):
                    rb["_dist"] = dist
                    best = rb
    if best is not None:
        best.pop("_dist", None)
    return best


def _count_rejection_touches(rejections: list, levels: np.ndarray, n: int, supply: bool):
    for rb in rejections:
        idx = rb["index"]
        count = 0
        for i in range(idx + 1, n):
            if supply:
                if levels[i] >= rb["price"]:
                    count += 1
            else:
                if levels[i] <= rb["price"]:
                    count += 1
        rb["touches"] = count


def _mark_rejection_mitigated(rejections: list, closes: np.ndarray, n: int, direction: str):
    for rb in rejections:
        idx = rb["index"]
        mitigated = False
        for i in range(idx + 1, min(idx + 60, n)):
            if direction == "bullish" and closes[i] < rb["price"]:
                mitigated = True
                break
            if direction == "bearish" and closes[i] > rb["price"]:
                mitigated = True
                break
        rb["mitigated"] = mitigated
