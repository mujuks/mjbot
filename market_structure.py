import numpy as np
import pandas as pd


def detect_swings(df: pd.DataFrame, cfg: dict) -> dict:
    left = cfg.get("swing_left", 3)
    right = cfg.get("swing_right", 3)
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    swing_highs = []
    swing_lows = []

    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        window_l = lows[i - left:i + right + 1]
        if highs[i] == np.max(window_h) and np.sum(window_h == highs[i]) == 1:
            swing_highs.append({"index": i, "price": float(highs[i])})
        if lows[i] == np.min(window_l) and np.sum(window_l == lows[i]) == 1:
            swing_lows.append({"index": i, "price": float(lows[i])})

    return {"highs": swing_highs, "lows": swing_lows}


def classify_structure(swings: dict) -> dict:
    sh_list = swings["highs"]
    sl_list = swings["lows"]
    n = len(sh_list)
    m = len(sl_list)

    if n < 2 and m < 2:
        return {"trend": "flat", "events": [], "last_bos": None, "last_choch": None}

    events = []
    prev_sh = None
    prev_sl = None

    for i, sh in enumerate(sh_list):
        label = "H"
        if prev_sh is not None:
            if sh["price"] > prev_sh["price"]:
                label = "HH"
            elif sh["price"] < prev_sh["price"]:
                label = "LH"
            else:
                label = "EH"
        sh["label"] = label
        prev_sh = sh

    for i, sl in enumerate(sl_list):
        label = "L"
        if prev_sl is not None:
            if sl["price"] > prev_sl["price"]:
                label = "HL"
            elif sl["price"] < prev_sl["price"]:
                label = "LL"
            else:
                label = "EL"
        sl["label"] = label
        prev_sl = sl

    trend = "flat"
    last_bos = None
    last_choch = None

    hh_count = sum(1 for s in sh_list if s.get("label") == "HH")
    hl_count = sum(1 for s in sl_list if s.get("label") == "HL")
    lh_count = sum(1 for s in sh_list if s.get("label") == "LH")
    ll_count = sum(1 for s in sl_list if s.get("label") == "LL")

    if hh_count > 0 and hl_count > 0 and lh_count == 0 and ll_count == 0:
        trend = "bullish"
    elif lh_count > 0 and ll_count > 0 and hh_count == 0 and hl_count == 0:
        trend = "bearish"
    elif hh_count + hl_count > lh_count + ll_count:
        trend = "bullish"
    elif lh_count + ll_count > hh_count + hl_count:
        trend = "bearish"

    for i in range(1, len(sh_list)):
        prev = sh_list[i - 1]
        curr = sh_list[i]
        if curr["label"] == "HH":
            events.append({"type": "HH", "index": curr["index"], "price": curr["price"],
                           "prev_price": prev["price"]})
        elif curr["label"] == "LH":
            events.append({"type": "LH", "index": curr["index"], "price": curr["price"],
                           "prev_price": prev["price"]})

    for i in range(1, len(sl_list)):
        prev = sl_list[i - 1]
        curr = sl_list[i]
        if curr["label"] == "HL":
            events.append({"type": "HL", "index": curr["index"], "price": curr["price"],
                           "prev_price": prev["price"]})
        elif curr["label"] == "LL":
            events.append({"type": "LL", "index": curr["index"], "price": curr["price"],
                           "prev_price": prev["price"]})

    events.sort(key=lambda e: e["index"])

    return {
        "trend": trend,
        "swing_highs": sh_list,
        "swing_lows": sl_list,
        "events": events,
        "counts": {"HH": hh_count, "HL": hl_count, "LH": lh_count, "LL": ll_count},
    }


def detect_bos_choch(df: pd.DataFrame, structure: dict, cfg: dict) -> dict:
    close = df["Close"].values
    n = len(close)
    lookback = cfg.get("structure_lookback", 3)
    swing_highs = structure.get("swing_highs", [])
    swing_lows = structure.get("swing_lows", [])

    recent_highs = [s for s in swing_highs if s["index"] < n - 1][-lookback:]
    recent_lows = [s for s in swing_lows if s["index"] < n - 1][-lookback:]

    last_close = close[-1]
    prev_close = close[-2] if n > 1 else last_close

    bos_bull = False
    bos_bear = False
    choch_bull = False
    choch_bear = False
    bos_level = None
    choch_level = None
    event = None

    if recent_highs:
        last_sh = recent_highs[-1]
        if prev_close <= last_sh["price"] and last_close > last_sh["price"]:
            if structure["trend"] == "bearish":
                choch_bull = True
                choch_level = last_sh["price"]
                event = "CHoCH_BULL"
            else:
                bos_bull = True
                bos_level = last_sh["price"]
                event = "BOS_BULL"

    if recent_lows:
        last_sl = recent_lows[-1]
        if prev_close >= last_sl["price"] and last_close < last_sl["price"]:
            if structure["trend"] == "bullish":
                choch_bear = True
                choch_level = last_sl["price"]
                event = "CHoCH_BEAR"
            else:
                bos_bear = True
                bos_level = last_sl["price"]
                event = "BOS_BEAR"

    in_discount = False
    in_premium = False
    if recent_highs and recent_lows:
        range_high = max(s["price"] for s in recent_highs[-5:])
        range_low = min(s["price"] for s in recent_lows[-5:])
        range_mid = (range_high + range_low) / 2
        if last_close < range_mid:
            in_discount = True
        elif last_close > range_mid:
            in_premium = True

    return {
        "bos_bull": bos_bull,
        "bos_bear": bos_bear,
        "choch_bull": choch_bull,
        "choch_bear": choch_bear,
        "bos_level": bos_level,
        "choch_level": choch_level,
        "event": event,
        "trend": structure["trend"],
        "in_discount": in_discount,
        "in_premium": in_premium,
        "range_high": max(s["price"] for s in recent_highs[-5:]) if recent_highs else 0,
        "range_low": min(s["price"] for s in recent_lows[-5:]) if recent_lows else 0,
    }


def compute_premium_discount(df: pd.DataFrame, cfg: dict) -> dict:
    lookback = cfg.get("pd_lookback", 50)
    highs = df["High"].values
    lows = df["Low"].values
    close = df["Close"].values
    n = len(df)
    start = max(0, n - lookback)
    range_high = float(np.max(highs[start:]))
    range_low = float(np.min(lows[start:]))
    range_size = range_high - range_low
    last = float(close[-1])

    if range_size <= 0:
        return {"zone": "equilibrium", "level": 0.5, "range_high": range_high,
                "range_low": range_low}

    pct = (last - range_low) / range_size
    if pct < 0.3:
        zone = "deep_discount"
    elif pct < 0.5:
        zone = "discount"
    elif pct < 0.7:
        zone = "premium"
    else:
        zone = "deep_premium"

    return {
        "zone": zone,
        "level": round(pct, 4),
        "range_high": range_high,
        "range_low": range_low,
        "midpoint": (range_high + range_low) / 2,
    }


def analyze_structure(df: pd.DataFrame, cfg: dict) -> dict:
    if df is None or df.empty or len(df) < cfg.get("swing_left", 3) * 3:
        return {
            "trend": "flat", "event": None, "bos_bull": False, "bos_bear": False,
            "choch_bull": False, "choch_bear": False,
            "in_discount": False, "in_premium": False,
            "range_high": 0, "range_low": 0, "midpoint": 0,
            "swing_highs": [], "swing_lows": [], "events": [],
            "pd_zone": "equilibrium", "pd_level": 0.5,
        }

    swings = detect_swings(df, cfg)
    structure = classify_structure(swings)
    bos_choch = detect_bos_choch(df, structure, cfg)
    pd_info = compute_premium_discount(df, cfg)

    return {
        "trend": structure["trend"],
        "event": bos_choch["event"],
        "bos_bull": bos_choch["bos_bull"],
        "bos_bear": bos_choch["bos_bear"],
        "choch_bull": bos_choch["choch_bull"],
        "choch_bear": bos_choch["choch_bear"],
        "bos_level": bos_choch["bos_level"],
        "choch_level": bos_choch["choch_level"],
        "in_discount": bos_choch["in_discount"],
        "in_premium": bos_choch["in_premium"],
        "range_high": bos_choch["range_high"],
        "range_low": bos_choch["range_low"],
        "midpoint": pd_info["midpoint"],
        "swing_highs": structure.get("swing_highs", []),
        "swing_lows": structure.get("swing_lows", []),
        "events": structure.get("events", []),
        "counts": structure.get("counts", {}),
        "pd_zone": pd_info["zone"],
        "pd_level": pd_info["level"],
    }
