import numpy as np
import pandas as pd


def detect_order_blocks(df: pd.DataFrame, cfg: dict) -> dict:
    atr_mult = cfg.get("ob_move_atr", 1.5)
    lookback = cfg.get("ob_lookback", 100)
    min_move = cfg.get("ob_min_move", 2)

    from zones import _atr

    atr = _atr(df, cfg.get("atr_period", 14)).values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values if "Volume" in df.columns else np.zeros(len(df))
    n = len(df)

    swing_left = cfg.get("swing_left", 3)
    pivot_highs = []
    pivot_lows = []
    for i in range(swing_left, n - swing_left):
        wh = highs[i - swing_left:i + swing_left + 1]
        wl = lows[i - swing_left:i + swing_left + 1]
        if highs[i] == np.max(wh) and np.sum(wh == highs[i]) == 1:
            pivot_highs.append((i, highs[i]))
        if lows[i] == np.min(wl) and np.sum(wl == lows[i]) == 1:
            pivot_lows.append((i, lows[i]))

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
            impulse_bars = 0
            for j in range(i + 1, min(i + 1 + 6, n)):
                move_up += closes[j] - opens[j]
                max_move = max(max_move, closes[j] - lows[i])
                if closes[j] > opens[j]:
                    impulse_bars += 1
            if max_move > atr_mult * atr[i] and move_up > 0:
                bos_confirmed = False
                for ph_idx, ph_price in pivot_highs:
                    if ph_idx > i and ph_idx <= i + 6 and closes[min(ph_idx, n - 1)] > ph_price:
                        bos_confirmed = True
                        break

                speed = max_move / max(impulse_bars, 1) / max(atr[i], 0.01)

                vol_avg = np.mean(volumes[max(0, i - 20):i]) if i > 0 else 0
                vol_at_ob = volumes[i]
                high_volume = vol_at_ob > vol_avg * 1.3 if vol_avg > 0 else False

                bullish_obs.append({
                    "index": i,
                    "top": float(highs[i]),
                    "bottom": float(lows[i]),
                    "strength": min(1.0, max_move / (atr_mult * atr[i])),
                    "touches": 0,
                    "bos_confirmed": bos_confirmed,
                    "speed": round(float(speed), 3),
                    "impulse_bars": impulse_bars,
                    "high_volume": high_volume,
                    "mitigated": False,
                })

        if is_bullish:
            move_down = 0
            max_move = 0
            impulse_bars = 0
            for j in range(i + 1, min(i + 1 + 6, n)):
                move_down += opens[j] - closes[j]
                max_move = max(max_move, highs[i] - closes[j])
                if closes[j] < opens[j]:
                    impulse_bars += 1
            if max_move > atr_mult * atr[i] and move_down > 0:
                bos_confirmed = False
                for sl_idx, sl_price in pivot_lows:
                    if sl_idx > i and sl_idx <= i + 6 and closes[min(sl_idx, n - 1)] < sl_price:
                        bos_confirmed = True
                        break

                speed = max_move / max(impulse_bars, 1) / max(atr[i], 0.01)

                vol_avg = np.mean(volumes[max(0, i - 20):i]) if i > 0 else 0
                vol_at_ob = volumes[i]
                high_volume = vol_at_ob > vol_avg * 1.3 if vol_avg > 0 else False

                bearish_obs.append({
                    "index": i,
                    "top": float(highs[i]),
                    "bottom": float(lows[i]),
                    "strength": min(1.0, max_move / (atr_mult * atr[i])),
                    "touches": 0,
                    "bos_confirmed": bos_confirmed,
                    "speed": round(float(speed), 3),
                    "impulse_bars": impulse_bars,
                    "high_volume": high_volume,
                    "mitigated": False,
                })

    bullish_obs = _merge_obs(bullish_obs)
    bearish_obs = _merge_obs(bearish_obs)
    _count_ob_touches(bullish_obs, lows, n)
    _count_ob_touches(bearish_obs, highs, n, supply=True)
    _mark_mitigated(bullish_obs, closes, n, direction="bullish")
    _mark_mitigated(bearish_obs, closes, n, direction="bearish")

    bullish_obs.sort(key=lambda z: z["index"], reverse=True)
    bearish_obs.sort(key=lambda z: z["index"], reverse=True)

    return {"bullish": bullish_obs, "bearish": bearish_obs}


def find_nearest_ob(obs: list, price: float, atr_val: float, cfg: dict) -> dict | None:
    max_dist = cfg.get("ob_max_dist_atr", 2.0) * atr_val
    best = None
    for ob in obs:
        dist = price - ((ob["bottom"] + ob["top"]) / 2)
        if abs(dist) <= max_dist:
            max_touches = cfg.get("ob_max_touches", 8)
            if ob["touches"] <= max_touches and not ob.get("mitigated", False):
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
            merged[-1]["bos_confirmed"] = merged[-1]["bos_confirmed"] or ob["bos_confirmed"]
            merged[-1]["speed"] = max(merged[-1]["speed"], ob["speed"])
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


def _mark_mitigated(obs: list, closes: np.ndarray, n: int, direction: str):
    for ob in obs:
        idx = ob["index"]
        mitigated = False
        for i in range(idx + 1, min(idx + 100, n)):
            if direction == "bullish" and closes[i] < ob["bottom"]:
                mitigated = True
                break
            if direction == "bearish" and closes[i] > ob["top"]:
                mitigated = True
                break
        ob["mitigated"] = mitigated
