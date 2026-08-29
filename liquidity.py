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
    if direction == "BUY":
        for t in pools.get("bsl_targets", []):
            tp = t["price"]
            if tp > entry and (tp - entry) >= min_rr * risk:
                return tp
        return entry + 2.0 * risk
    else:
        for t in pools.get("ssl_targets", []):
            tp = t["price"]
            if tp < entry and (entry - tp) >= min_rr * risk:
                return tp
        return entry - 2.0 * risk


def compute_multi_tp(entry: float, sl: float, direction: str, pools: dict, atr_val: float) -> tuple:
    risk = abs(entry - sl)
    if risk <= 0:
        return entry, entry

    targets_key = "bsl_targets" if direction == "BUY" else "ssl_targets"
    targets = pools.get(targets_key, [])

    tp1_rr = 2.0
    tp2_rr = 4.0

    tp1 = None
    tp2 = None

    if direction == "BUY":
        candidates = [t["price"] for t in targets if t["price"] > entry]
        candidates.sort()
        for p in candidates:
            rr = (p - entry) / risk
            if tp1 is None and rr >= tp1_rr:
                tp1 = p
            elif tp1 is not None and tp2 is None and rr >= tp2_rr:
                tp2 = p
                break
        if tp1 is None:
            tp1 = entry + tp1_rr * risk
        if tp2 is None:
            tp2 = entry + tp2_rr * risk
    else:
        candidates = [t["price"] for t in targets if t["price"] < entry]
        candidates.sort(reverse=True)
        for p in candidates:
            rr = (entry - p) / risk
            if tp1 is None and rr >= tp1_rr:
                tp1 = p
            elif tp1 is not None and tp2 is None and rr >= tp2_rr:
                tp2 = p
                break
        if tp1 is None:
            tp1 = entry - tp1_rr * risk
        if tp2 is None:
            tp2 = entry - tp2_rr * risk

    return tp1, tp2


def detect_fvg(df: pd.DataFrame, cfg: dict) -> dict:
    if df is None or df.empty or len(df) < 3:
        return {"bullish_fvgs": [], "bearish_fvgs": [], "nearest": None}

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)
    lookback = cfg.get("fvg_lookback", 50)
    start = max(2, n - lookback)

    from zones import _atr
    atr_vals = _atr(df, cfg.get("atr_period", 14)).values

    bullish_fvgs = []
    bearish_fvgs = []

    for i in range(start, n - 1):
        if lows[i + 1] > highs[i - 1]:
            gap = lows[i + 1] - highs[i - 1]
            atr_ref = atr_vals[i] if atr_vals[i] > 0 else 1.0
            size_class = _classify_fvg_size(gap, atr_ref, cfg)
            mid = (lows[i + 1] + highs[i - 1]) / 2
            entry_price = _fvg_entry_price(size_class, mid, lows[i + 1], highs[i - 1])
            bullish_fvgs.append({
                "top": float(lows[i + 1]),
                "bottom": float(highs[i - 1]),
                "mid": float(mid),
                "size": float(gap),
                "size_class": size_class,
                "entry_price": float(entry_price),
                "index": i,
                "mitigated": closes[i + 1] <= highs[i - 1] if i + 1 < n else False,
            })

        if highs[i + 1] < lows[i - 1]:
            gap = lows[i - 1] - highs[i + 1]
            atr_ref = atr_vals[i] if atr_vals[i] > 0 else 1.0
            size_class = _classify_fvg_size(gap, atr_ref, cfg)
            mid = (lows[i - 1] + highs[i + 1]) / 2
            entry_price = _fvg_entry_price(size_class, mid, lows[i - 1], highs[i + 1])
            bearish_fvgs.append({
                "top": float(lows[i - 1]),
                "bottom": float(highs[i + 1]),
                "mid": float(mid),
                "size": float(gap),
                "size_class": size_class,
                "entry_price": float(entry_price),
                "index": i,
                "mitigated": closes[i + 1] >= lows[i - 1] if i + 1 < n else False,
            })

    last_price = closes[-1]
    atr = float(pd.Series(np.abs(np.diff(closes, prepend=closes[0]))).ewm(span=14, adjust=False).mean().iloc[-1])
    if atr <= 0:
        atr = 1.0

    nearest = None
    for fvg in reversed(bullish_fvgs):
        if not fvg["mitigated"] and abs(last_price - fvg["mid"]) < 3 * atr:
            nearest = {**fvg, "direction": "bullish"}
            break

    if nearest is None:
        for fvg in reversed(bearish_fvgs):
            if not fvg["mitigated"] and abs(last_price - fvg["mid"]) < 3 * atr:
                nearest = {**fvg, "direction": "bearish"}
                break

    return {
        "bullish_fvgs": bullish_fvgs,
        "bearish_fvgs": bearish_fvgs,
        "nearest": nearest,
        "bullish_count": sum(1 for f in bullish_fvgs if not f["mitigated"]),
        "bearish_count": sum(1 for f in bearish_fvgs if not f["mitigated"]),
    }


def _classify_fvg_size(gap: float, atr: float, cfg: dict) -> str:
    big_threshold = cfg.get("fvg_big_atr", 1.5)
    small_threshold = cfg.get("fvg_small_atr", 0.3)
    ratio = gap / atr if atr > 0 else 0
    if ratio >= big_threshold:
        return "big"
    elif ratio <= small_threshold:
        return "small"
    return "medium"


def _fvg_entry_price(size_class: str, mid: float, top: float, bottom: float) -> float:
    if size_class == "big":
        return mid
    return bottom


def detect_liquidity_voids(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Detect liquidity voids — rapid one-sided moves that leave order flow gaps.
    A void is a sequence of 3+ consecutive same-direction candles with
    minimal retracement, creating an imbalance zone.

    Bullish void: rapid upward move leaving unfilled orders below.
    Bearish void: rapid downward move leaving unfilled orders above.
    """
    if df is None or df.empty or len(df) < 5:
        return {"bullish_voids": [], "bearish_voids": [], "nearest": None}

    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values
    closes = df["Close"].values
    n = len(df)

    from zones import _atr
    atr_vals = _atr(df, cfg.get("atr_period", 14)).values

    lookback = cfg.get("void_lookback", 60)
    start = max(3, n - lookback)
    min_candles = cfg.get("void_min_candles", 3)
    max_retrace_pct = cfg.get("void_max_retrace_pct", 0.3)

    bullish_voids = []
    bearish_voids = []

    for i in range(start, n - min_candles):
        bull_count = 0
        bear_count = 0
        for j in range(i, min(i + min_candles + 3, n)):
            if closes[j] > opens[j]:
                bull_count += 1
            elif closes[j] < opens[j]:
                bear_count += 1

        if bull_count >= min_candles:
            move_high = max(highs[i:min(i + min_candles + 2, n)])
            move_low = min(lows[i:min(i + min_candles + 2, n)])
            move_size = move_high - move_low
            retrace = 0
            for j in range(i + min_candles, min(i + min_candles + 3, n)):
                retrace = max(retrace, move_high - lows[j])

            atr_ref = atr_vals[i] if i < len(atr_vals) and atr_vals[i] > 0 else 1.0
            if move_size > atr_ref * 1.5:
                retrace_ok = retrace <= move_size * max_retrace_pct if move_size > 0 else True
                if retrace_ok:
                    bullish_voids.append({
                        "index": i,
                        "top": float(move_high),
                        "bottom": float(move_low),
                        "mid": float((move_high + move_low) / 2),
                        "size": float(move_size),
                        "candles": bull_count,
                        "strength": min(1.0, move_size / (2.0 * atr_ref)),
                        "mitigated": False,
                    })

        if bear_count >= min_candles:
            move_high = max(highs[i:min(i + min_candles + 2, n)])
            move_low = min(lows[i:min(i + min_candles + 2, n)])
            move_size = move_high - move_low
            retrace = 0
            for j in range(i + min_candles, min(i + min_candles + 3, n)):
                retrace = max(retrace, highs[j] - move_low)

            atr_ref = atr_vals[i] if i < len(atr_vals) and atr_vals[i] > 0 else 1.0
            if move_size > atr_ref * 1.5:
                retrace_ok = retrace <= move_size * max_retrace_pct if move_size > 0 else True
                if retrace_ok:
                    bearish_voids.append({
                        "index": i,
                        "top": float(move_high),
                        "bottom": float(move_low),
                        "mid": float((move_high + move_low) / 2),
                        "size": float(move_size),
                        "candles": bear_count,
                        "strength": min(1.0, move_size / (2.0 * atr_ref)),
                        "mitigated": False,
                    })

    for v in bullish_voids:
        for j in range(v["index"] + min_candles, n):
            if closes[j] < v["bottom"]:
                v["mitigated"] = True
                break

    for v in bearish_voids:
        for j in range(v["index"] + min_candles, n):
            if closes[j] > v["top"]:
                v["mitigated"] = True
                break

    bullish_voids.sort(key=lambda z: z["index"], reverse=True)
    bearish_voids.sort(key=lambda z: z["index"], reverse=True)

    last_price = closes[-1]
    atr_val = atr_vals[-1] if len(atr_vals) > 0 and atr_vals[-1] > 0 else 1.0
    nearest = None
    for v in reversed(bullish_voids):
        if not v["mitigated"] and abs(last_price - v["mid"]) < 3 * atr_val:
            nearest = {**v, "direction": "bullish"}
            break
    if nearest is None:
        for v in reversed(bearish_voids):
            if not v["mitigated"] and abs(last_price - v["mid"]) < 3 * atr_val:
                nearest = {**v, "direction": "bearish"}
                break

    return {
        "bullish_voids": bullish_voids,
        "bearish_voids": bearish_voids,
        "nearest": nearest,
        "bullish_count": sum(1 for v in bullish_voids if not v["mitigated"]),
        "bearish_count": sum(1 for v in bearish_voids if not v["mitigated"]),
    }


def detect_volume_profile(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Detect volume profile — Point of Control (POC) and Value Area (VA).

    The POC is the price level with the highest traded volume. It acts as a
    magnet for price and a key institutional reference point.

    Value Area (VA) contains ~70% of traded volume (1 standard deviation).
    Prices above VA = premium, below VA = discount.

    Returns POC, VA high/low, and whether price is near POC.
    """
    if df is None or df.empty or len(df) < 20:
        return {"poc": None, "va_high": None, "va_low": None, "near_poc": False, "score": 0}

    cfg_vp = cfg.get("volume_profile", {})
    if not cfg_vp.get("enabled", True):
        return {"poc": None, "va_high": None, "va_low": None, "near_poc": False, "score": 0}

    lookback = cfg_vp.get("lookback", 100)
    n = len(df)
    start = max(0, n - lookback)

    highs = df["High"].iloc[start:].values
    lows = df["Low"].iloc[start:].values
    closes = df["Close"].iloc[start:].values
    volumes = df["Volume"].iloc[start:].values

    if np.sum(volumes) <= 0:
        return {"poc": None, "va_high": None, "va_low": None, "near_poc": False, "score": 0}

    # Create price bins
    price_min = float(np.min(lows))
    price_max = float(np.max(highs))
    if price_max <= price_min:
        return {"poc": None, "va_high": None, "va_low": None, "near_poc": False, "score": 0}

    num_bins = cfg_vp.get("bins", 50)
    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_volume = np.zeros(num_bins)

    # Distribute volume into price bins
    for i in range(len(closes)):
        typical = (highs[i] + lows[i] + closes[i]) / 3.0
        bin_idx = int((typical - price_min) / (price_max - price_min) * (num_bins - 1))
        bin_idx = max(0, min(num_bins - 1, bin_idx))
        bin_volume[bin_idx] += volumes[i]

    # POC = bin with highest volume
    poc_idx = int(np.argmax(bin_volume))
    poc = float(bin_mids[poc_idx])

    # Value Area: expand from POC until ~70% of volume is included
    total_vol = np.sum(bin_volume)
    va_target = total_vol * 0.70
    accumulated = bin_volume[poc_idx]
    va_low_idx = poc_idx
    va_high_idx = poc_idx

    while accumulated < va_target and (va_low_idx > 0 or va_high_idx < num_bins - 1):
        expand_up = bin_volume[va_high_idx + 1] if va_high_idx < num_bins - 1 else 0
        expand_down = bin_volume[va_low_idx - 1] if va_low_idx > 0 else 0

        if expand_up >= expand_down and va_high_idx < num_bins - 1:
            va_high_idx += 1
            accumulated += bin_volume[va_high_idx]
        elif va_low_idx > 0:
            va_low_idx -= 1
            accumulated += bin_volume[va_low_idx]
        else:
            break

    va_high = float(bin_mids[va_high_idx])
    va_low = float(bin_mids[va_low_idx])

    # Check if current price is near POC
    last_price = float(closes[-1])
    atr_ref = float(np.mean(np.abs(np.diff(closes, prepend=closes[0]))[-14:]))
    if atr_ref <= 0:
        atr_ref = 1.0
    near_poc = abs(last_price - poc) < cfg_vp.get("near_poc_atr", 0.5) * atr_ref

    # Score: near POC = momentum stall zone (neutral), near VA edges = reversal zone
    score = 0
    if near_poc:
        score = 0  # POC is a magnet, not directional
    elif last_price > va_high:
        score = 1  # In premium, possible reversal down
    elif last_price < va_low:
        score = 1  # In discount, possible reversal up

    return {
        "poc": round(poc, 2),
        "va_high": round(va_high, 2),
        "va_low": round(va_low, 2),
        "near_poc": near_poc,
        "in_premium": last_price > va_high,
        "in_discount": last_price < va_low,
        "volume_at_poc": float(bin_volume[poc_idx]),
        "score": score,
    }
