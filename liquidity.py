import numpy as np
import pandas as pd


def detect_liquidity_sweep(df: pd.DataFrame, cfg: dict) -> dict:
    left = cfg.get("sweep_pivot", 3)
    window = cfg.get("sweep_window", 4)

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values if "Volume" in df.columns else np.zeros(len(df))
    n = len(df)

    piv_high = []
    piv_low = []
    for i in range(left, n - left):
        window_h = highs[i - left:i + left + 1]
        window_l = lows[i - left:i + left + 1]
        if highs[i] == np.max(window_h) and np.sum(window_h == highs[i]) == 1:
            piv_high.append((i, highs[i]))
        if lows[i] == np.min(window_l) and np.sum(window_l == lows[i]) == 1:
            piv_low.append((i, lows[i]))

    sweeps = []
    last_sweep = {"bullish": False, "bearish": False, "level": None, "bars_ago": None,
                  "wick_depth": 0, "volume_confirmed": False}

    for i in range(max(1, n - window), n):
        for pi, pl in piv_low:
            if pi < i and lows[i] < pl and closes[i] > pl:
                wick_depth = pl - lows[i]
                avg_vol = np.mean(volumes[max(0, i - 20):i]) if i > 0 else 0
                vol_confirmed = volumes[i] > avg_vol * 1.2 if avg_vol > 0 else False
                sweep = {
                    "bullish": True, "bearish": False, "level": float(pl),
                    "bars_ago": n - 1 - i, "wick_depth": float(wick_depth),
                    "volume_confirmed": vol_confirmed, "index": i,
                }
                sweeps.append(sweep)
                last_sweep = sweep
                break

        for pi, ph in piv_high:
            if pi < i and highs[i] > ph and closes[i] < ph:
                wick_depth = highs[i] - ph
                avg_vol = np.mean(volumes[max(0, i - 20):i]) if i > 0 else 0
                vol_confirmed = volumes[i] > avg_vol * 1.2 if avg_vol > 0 else False
                sweep = {
                    "bullish": False, "bearish": True, "level": float(ph),
                    "bars_ago": n - 1 - i, "wick_depth": float(wick_depth),
                    "volume_confirmed": vol_confirmed, "index": i,
                }
                sweeps.append(sweep)
                last_sweep = sweep
                break

    return last_sweep, sweeps


def find_equal_levels(prices: list, tolerance_pct: float = 0.05) -> list:
    if len(prices) < 2:
        return []
    clusters = []
    sorted_prices = sorted(enumerate(prices), key=lambda x: x[1])
    used = set()
    for i in range(len(sorted_prices)):
        if i in used:
            continue
        cluster = [sorted_prices[i]]
        used.add(i)
        for j in range(i + 1, len(sorted_prices)):
            if j in used:
                continue
            p1 = sorted_prices[i][1]
            p2 = sorted_prices[j][1]
            if p1 > 0 and abs(p2 - p1) / p1 <= tolerance_pct / 100:
                cluster.append(sorted_prices[j])
                used.add(j)
            elif p2 - p1 > p1 * tolerance_pct / 100:
                break
        if len(cluster) >= 2:
            avg_price = np.mean([c[1] for c in cluster])
            clusters.append({
                "price": float(avg_price),
                "count": len(cluster),
                "indices": [c[0] for c in cluster],
            })
    return clusters


def detect_liquidity_pools(df: pd.DataFrame, cfg: dict) -> dict:
    eqh_tol = cfg.get("eqh_tolerance_pct", 0.05)
    eqh_lookback = cfg.get("eqh_lookback", 30)
    n = len(df)
    start = max(0, n - eqh_lookback)

    highs = df["High"].iloc[start:].values.tolist()
    lows = df["Low"].iloc[start:].values.tolist()

    equal_highs = find_equal_levels(highs, eqh_tol)
    equal_lows = find_equal_levels(lows, eqh_tol)

    last_price = float(df["Close"].iloc[-1])

    bsl_targets = []
    ssl_targets = []

    for eq in equal_highs:
        bsl_targets.append({
            "price": eq["price"],
            "strength": min(1.0, eq["count"] / 3.0),
            "type": "buy_side_liquidity",
        })

    for eq in equal_lows:
        ssl_targets.append({
            "price": eq["price"],
            "strength": min(1.0, eq["count"] / 3.0),
            "type": "sell_side_liquidity",
        })

    piv_high = []
    piv_low = []
    left = cfg.get("sweep_pivot", 3)
    for i in range(left, n - left):
        if df["High"].iloc[i] == df["High"].iloc[i - left:i + left + 1].max():
            piv_high.append({"index": i, "price": float(df["High"].iloc[i])})
        if df["Low"].iloc[i] == df["Low"].iloc[i - left:i + left + 1].min():
            piv_low.append({"index": i, "price": float(df["Low"].iloc[i])})

    for ph in piv_high[-5:]:
        if ph["price"] > last_price:
            bsl_targets.append({
                "price": ph["price"],
                "strength": 0.5,
                "type": "external_bsl",
            })

    for pl in piv_low[-5:]:
        if pl["price"] < last_price:
            ssl_targets.append({
                "price": pl["price"],
                "strength": 0.5,
                "type": "external_ssl",
            })

    bsl_targets.sort(key=lambda x: x["price"])
    ssl_targets.sort(key=lambda x: x["price"], reverse=True)

    nearest_bsl = None
    nearest_ssl = None
    for t in bsl_targets:
        if t["price"] > last_price:
            nearest_bsl = t
            break
    for t in ssl_targets:
        if t["price"] < last_price:
            nearest_ssl = t
            break

    return {
        "bsl_targets": bsl_targets,
        "ssl_targets": ssl_targets,
        "nearest_bsl": nearest_bsl,
        "nearest_ssl": nearest_ssl,
        "equal_highs": equal_highs,
        "equal_lows": equal_lows,
    }


def compute_dynamic_tp(entry: float, sl: float, direction: str, pools: dict, atr_val: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return entry

    min_rr = 1.5
    targets = []
    if direction == "BUY":
        for t in pools.get("bsl_targets", []):
            if t["price"] > entry:
                targets.append(t["price"])
        if targets:
            tp = targets[0]
            if (tp - entry) >= min_rr * risk:
                return tp
        return entry + 2.0 * risk
    else:
        for t in pools.get("ssl_targets", []):
            if t["price"] < entry:
                targets.append(t["price"])
        if targets:
            tp = targets[0]
            if (entry - tp) >= min_rr * risk:
                return tp
        return entry - 2.0 * risk


def detect_fvg(df: pd.DataFrame, cfg: dict) -> dict:
    if df is None or df.empty or len(df) < 3:
        return {"bullish_fvgs": [], "bearish_fvgs": [], "nearest": None}

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)
    lookback = cfg.get("fvg_lookback", 50)
    start = max(2, n - lookback)

    bullish_fvgs = []
    bearish_fvgs = []

    for i in range(start, n - 1):
        if lows[i + 1] > highs[i - 1]:
            gap = lows[i + 1] - highs[i - 1]
            bullish_fvgs.append({
                "top": float(lows[i + 1]),
                "bottom": float(highs[i - 1]),
                "size": float(gap),
                "index": i,
                "mitigated": closes[i + 1] <= highs[i - 1] if i + 1 < n else False,
            })

        if highs[i + 1] < lows[i - 1]:
            gap = lows[i - 1] - highs[i + 1]
            bearish_fvgs.append({
                "top": float(lows[i - 1]),
                "bottom": float(highs[i + 1]),
                "size": float(gap),
                "index": i,
                "mitigated": closes[i + 1] >= lows[i - 1] if i + 1 < n else False,
            })

    last_price = closes[-1]
    atr = float(pd.Series(np.abs(np.diff(closes, prepend=closes[0]))).ewm(span=14, adjust=False).mean().iloc[-1])
    if atr <= 0:
        atr = 1.0

    nearest = None
    for fvg in reversed(bullish_fvgs):
        if not fvg["mitigated"] and abs(last_price - (fvg["top"] + fvg["bottom"]) / 2) < 3 * atr:
            nearest = {**fvg, "direction": "bullish"}
            break

    if nearest is None:
        for fvg in reversed(bearish_fvgs):
            if not fvg["mitigated"] and abs(last_price - (fvg["top"] + fvg["bottom"]) / 2) < 3 * atr:
                nearest = {**fvg, "direction": "bearish"}
                break

    return {
        "bullish_fvgs": bullish_fvgs,
        "bearish_fvgs": bearish_fvgs,
        "nearest": nearest,
        "bullish_count": sum(1 for f in bullish_fvgs if not f["mitigated"]),
        "bearish_count": sum(1 for f in bearish_fvgs if not f["mitigated"]),
    }
