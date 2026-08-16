import numpy as np
import pandas as pd

from liquidity import detect_liquidity_sweep
from zones import detect_zones, _atr
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int, slow: int, signal: int):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def _momentum(close: pd.Series, atr_series: pd.Series, cfg: dict) -> int:
    fast = ema(close, cfg["ma_fast"])
    slow = ema(close, cfg["ma_slow"])
    trend_up = fast.iloc[-1] > slow.iloc[-1]
    trend_down = fast.iloc[-1] < slow.iloc[-1]

    last = close.iloc[-1]
    body = abs(last - close.iloc[-2])
    strong_up = close.iloc[-1] > close.iloc[-2] and body > cfg.get("market_mover_atr", 1.0) * atr_series.iloc[-1]
    strong_dn = close.iloc[-1] < close.iloc[-2] and body > cfg.get("market_mover_atr", 1.0) * atr_series.iloc[-1]

    score = 0
    score += 1 if trend_up else -1
    score += 1 if strong_up else (-1 if strong_dn else 0)
    return score


def _volume_signal(volume: pd.Series, cfg: dict) -> int:
    if volume.empty or len(volume) < 2:
        return 0
    period = cfg.get("volume_period", 20)
    mult = cfg.get("volume_mult", 1.5)
    avg = volume.iloc[-min(period, len(volume)):].mean()
    if avg <= 0:
        return 0
    if volume.iloc[-1] > mult * avg:
        return 1
    return 0


def _find_nearest_zone(zones: list, price: float, atr_val: float) -> dict | None:
    best = None
    for zone in zones:
        distance = zone["bottom"] - price if zone["type"] == "demand" else price - zone["top"]
        if distance <= 1.5 * atr_val and distance >= -1.5 * atr_val:
            if best is None or abs(distance) < abs(best["_distance"]):
                zone["_distance"] = distance
                best = zone
    if best is not None:
        best.pop("_distance", None)
    return best


def analyze(df: pd.DataFrame, cfg: dict) -> dict:
    empty = {
        "signal": "NONE", "score": 0, "price": 0.0, "entry": 0.0,
        "stop_loss": 0.0, "take_profit": 0.0, "details": {},
    }
    if df is None or df.empty or len(df) < 60:
        return empty

    close = df["Close"]
    price = float(close.iloc[-1])
    atr_val = float(_atr(df, cfg.get("atr_period", 14)).iloc[-1])

    momentum = _momentum(close, _atr(df, cfg.get("atr_period", 14)), cfg)
    zones = detect_zones(df, cfg)
    sweep = detect_liquidity_sweep(df, cfg)
    vol_score = _volume_signal(df["Volume"], cfg)

    rsi_val = rsi(close, cfg["rsi_period"]).iloc[-1]
    macd_line, macd_sig, _ = macd(close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])

    demand_zone = _find_nearest_zone(zones["demand"], price, atr_val)
    supply_zone = _find_nearest_zone(zones["supply"], price, atr_val)

    buy_score = 0
    sell_score = 0
    details = {}

    if demand_zone:
        details["zone"] = f"demand {demand_zone['bottom']:.2f}-{demand_zone['top']:.2f} (touches {demand_zone['touches']})"
        buy_score += 2
        if demand_zone["touches"] == 0:
            buy_score += 1
    if sweep["bullish"]:
        details["sweep"] = f"bullish sweep of low {sweep['level']:.2f} ({sweep['bars_ago']} bars ago)"
        buy_score += 2

    if supply_zone:
        details["zone"] = f"supply {supply_zone['bottom']:.2f}-{supply_zone['top']:.2f} (touches {supply_zone['touches']})"
        sell_score += 2
        if supply_zone["touches"] == 0:
            sell_score += 1
    if sweep["bearish"]:
        details["sweep"] = f"bearish sweep of high {sweep['level']:.2f} ({sweep['bars_ago']} bars ago)"
        sell_score += 2

    if momentum > 0:
        buy_score += momentum
        details["momentum"] = "up"
    elif momentum < 0:
        sell_score += -momentum
        details["momentum"] = "down"

    if vol_score:
        details["volume"] = "spike"
        if momentum > 0:
            buy_score += 1
        elif momentum < 0:
            sell_score += 1

    if macd_line.iloc[-1] > macd_sig.iloc[-1] and rsi_val < cfg["rsi_overbought"]:
        buy_score += 1
    elif macd_line.iloc[-1] < macd_sig.iloc[-1] and rsi_val > cfg["rsi_oversold"]:
        sell_score += 1

    details["rsi"] = round(float(rsi_val), 2)

    threshold = cfg.get("signal_threshold", 3)
    strong = cfg.get("strong_threshold", 5)
    sl_buffer = cfg.get("sl_atr_buffer", 1.0)
    rr = cfg.get("risk_reward", 2.0)

    signal = "NONE"
    entry = sl = tp = price

    if buy_score >= threshold and buy_score > sell_score:
        if buy_score >= strong:
            signal = "STRONG_BUY"
        else:
            signal = "BUY"
        entry = price
        if sweep["bullish"]:
            sl = min(sweep["level"], demand_zone["bottom"] if demand_zone else price) - sl_buffer * atr_val
        elif demand_zone:
            sl = demand_zone["bottom"] - sl_buffer * atr_val
        else:
            sl = price - 2 * sl_buffer * atr_val
        tp = entry + rr * (entry - sl)
    elif sell_score >= threshold and sell_score > buy_score:
        if sell_score >= strong:
            signal = "STRONG_SELL"
        else:
            signal = "SELL"
        entry = price
        if sweep["bearish"]:
            sl = max(sweep["level"], supply_zone["top"] if supply_zone else price) + sl_buffer * atr_val
        elif supply_zone:
            sl = supply_zone["top"] + sl_buffer * atr_val
        else:
            sl = price + 2 * sl_buffer * atr_val
        tp = entry - rr * (sl - entry)

    return {
        "signal": signal,
        "score": max(buy_score, sell_score),
        "price": price,
        "entry": round(entry, 2),
        "stop_loss": round(sl, 2),
        "take_profit": round(tp, 2),
        "details": details,
        "rsi": round(float(rsi_val), 2),
    }


def _build_zones_arrays(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray, cfg: dict):
    n = len(close)
    left = cfg.get("zone_left", 3)
    right = cfg.get("zone_right", 3)
    move_atr = cfg.get("zone_move_atr", 1.5)

    piv_low_idx = []
    piv_high_idx = []
    for i in range(left, n - left):
        if low[i] == min(low[i - left:i + left + 1]):
            piv_low_idx.append(i)
        if high[i] == max(high[i - left:i + left + 1]):
            piv_high_idx.append(i)

    def build(kind, piv_idx):
        zones = []
        for idx in piv_idx:
            if idx + right + 1 >= n:
                continue
            start = max(0, idx - left)
            base_high = float(max(high[start:idx]))
            base_low = float(min(low[start:idx]))
            if kind == "demand":
                touch_idx = np.where(low[idx + 1:] <= base_high)[0] + idx + 1
            else:
                touch_idx = np.where(high[idx + 1:] <= base_low)[0] + idx + 1
            zones.append(
                {"kind": kind, "top": base_high, "bottom": base_low, "index": idx, "touch": touch_idx}
            )
        return zones

    demand = build("demand", piv_low_idx)
    supply = build("supply", piv_high_idx)
    return demand, supply


def build_signal_series(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    n = len(df)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    vol = df["Volume"].to_numpy(float)
    atr_s = _atr(df, cfg.get("atr_period", 14)).to_numpy(float)

    fast = ema(df["Close"], cfg["ma_fast"]).to_numpy(float)
    slow = ema(df["Close"], cfg["ma_slow"]).to_numpy(float)
    diff = np.diff(close, prepend=close[0])
    body = np.abs(diff)
    atr_ma = cfg.get("market_mover_atr", 1.0)
    strong_up = (diff > 0) & (body > atr_ma * atr_s)
    strong_dn = (diff < 0) & (body > atr_ma * atr_s)
    momentum = np.where(fast > slow, 1, -1) + np.where(strong_up, 1, np.where(strong_dn, -1, 0))

    vol_period = cfg.get("volume_period", 20)
    vol_avg = pd.Series(vol).rolling(vol_period, min_periods=1).mean().to_numpy(float)
    vol_spike = (vol > cfg.get("volume_mult", 1.5) * vol_avg) & (vol_avg > 0)

    rsi_val = rsi(df["Close"], cfg["rsi_period"]).to_numpy(float)
    macd_line, macd_sig, _ = macd(df["Close"], cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])
    ml = macd_line.to_numpy(float)
    ms = macd_sig.to_numpy(float)
    bullish_f = (ml > ms) & (rsi_val < cfg["rsi_overbought"])
    bearish_f = (ml < ms) & (rsi_val > cfg["rsi_oversold"])

    window = cfg.get("sweep_window", 3)
    sweep_event_bull = np.zeros(n, bool)
    sweep_event_bear = np.zeros(n, bool)
    sweep_event_bull_level = np.full(n, np.nan)
    sweep_event_bear_level = np.full(n, np.nan)
    piv_low_level = np.full(n, np.nan)
    piv_high_level = np.full(n, np.nan)
    pl_lo, pl_hi = [], []
    left_p = cfg.get("sweep_pivot", 3)
    for i in range(left_p, n - left_p):
        if low[i] == min(low[i - left_p:i + left_p + 1]):
            piv_low_level[i] = low[i]
            pl_lo.append(i)
        if high[i] == max(high[i - left_p:i + left_p + 1]):
            piv_high_level[i] = high[i]
            pl_hi.append(i)

    for i in range(1, n):
        for j in pl_lo:
            if j >= i:
                break
            level = piv_low_level[j]
            if low[i] < level and close[i] > level:
                sweep_event_bull[i] = True
                sweep_event_bull_level[i] = level
                break
        for j in pl_hi:
            if j >= i:
                break
            level = piv_high_level[j]
            if high[i] > level and close[i] < level:
                sweep_event_bear[i] = True
                sweep_event_bear_level[i] = level
                break

    sweep_bull = np.zeros(n, bool)
    sweep_bear = np.zeros(n, bool)
    sweep_bull_level = np.full(n, np.nan)
    sweep_bear_level = np.full(n, np.nan)
    for i in range(n):
        for k in range(max(0, i - window + 1), i + 1):
            if sweep_event_bull[k]:
                sweep_bull[i] = True
                sweep_bull_level[i] = sweep_event_bull_level[k]
                break
        for k in range(max(0, i - window + 1), i + 1):
            if sweep_event_bear[k]:
                sweep_bear[i] = True
                sweep_bear_level[i] = sweep_event_bear_level[k]
                break

    demand_cands, supply_cands = _build_zones_arrays(high, low, close, atr_s, cfg)
    right = cfg.get("zone_right", 3)
    move_atr = cfg.get("zone_move_atr", 1.5)

    near_demand = np.full(n, np.nan)
    near_supply = np.full(n, np.nan)
    zone_demand_bottom = np.full(n, np.nan)
    zone_supply_top = np.full(n, np.nan)
    demand_fresh = np.zeros(n, bool)
    supply_fresh = np.zeros(n, bool)

    def nearest_zone(cands, i, kind):
        active = []
        for z in cands:
            idx = z["index"]
            if idx >= i:
                break
            if i <= idx + right:
                continue
            ahead_i = min(idx + 1 + right, i)
            move = (close[ahead_i] - low[idx]) if kind == "demand" else (high[idx] - close[ahead_i])
            if move <= move_atr * atr_s[idx]:
                continue
            touch_count = int(np.searchsorted(z["touch"], i, side="right"))
            active.append((z["bottom"], z["top"], touch_count))
        if not active:
            return None
        active.sort(key=lambda t: t[0])
        merged = []
        for bottom, top, tc in active:
            if merged and bottom < merged[-1][1]:
                merged[-1] = (min(merged[-1][0], bottom), max(merged[-1][1], top), min(merged[-1][2], tc))
            else:
                merged.append((bottom, top, tc))
        price = close[i]
        best = None
        for bottom, top, tc in merged:
            dist = (bottom - price) if kind == "demand" else (price - top)
            if abs(dist) > 1.5 * atr_s[i]:
                continue
            if best is None or abs(dist) < abs(best[0]):
                best = (dist, bottom, top, tc)
        return best

    for i in range(1, n):
        best_d = nearest_zone(demand_cands, i, "demand")
        if best_d is not None:
            near_demand[i] = best_d[0]
            zone_demand_bottom[i] = best_d[1]
            demand_fresh[i] = best_d[3] == 0
        best_s = nearest_zone(supply_cands, i, "supply")
        if best_s is not None:
            near_supply[i] = best_s[0]
            zone_supply_top[i] = best_s[2]
            supply_fresh[i] = best_s[3] == 0

    buy_score = np.zeros(n, int)
    sell_score = np.zeros(n, int)

    has_demand = ~np.isnan(near_demand)
    has_supply = ~np.isnan(near_supply)
    buy_score[has_demand] += 2
    sell_score[has_supply] += 2
    buy_score[has_demand & demand_fresh] += 1
    sell_score[has_supply & supply_fresh] += 1

    buy_score[sweep_bull] += 2
    sell_score[sweep_bear] += 2
    buy_score += np.maximum(momentum, 0)
    sell_score += np.maximum(-momentum, 0)
    buy_score[vol_spike & (momentum > 0)] += 1
    sell_score[vol_spike & (momentum < 0)] += 1
    buy_score[bullish_f] += 1
    sell_score[bearish_f] += 1

    threshold = cfg.get("signal_threshold", 3)
    strong = cfg.get("strong_threshold", 5)
    sl_buffer = cfg.get("sl_atr_buffer", 1.0)
    rr = cfg.get("risk_reward", 2.0)

    signal_buy = (buy_score >= threshold) & (buy_score > sell_score)
    signal_sell = (sell_score >= threshold) & (sell_score > buy_score)

    signals = np.full(n, "NONE", dtype=object)
    signals[signal_buy & (buy_score >= strong)] = "STRONG_BUY"
    signals[signal_buy & (buy_score < strong)] = "BUY"
    signals[signal_sell & (sell_score >= strong)] = "STRONG_SELL"
    signals[signal_sell & (sell_score < strong)] = "SELL"

    sl = np.full(n, np.nan)
    tp = np.full(n, np.nan)

    for i in np.where(signal_buy)[0]:
        price = close[i]
        sweep_level = sweep_bull_level[i]
        if not np.isnan(sweep_level):
            zl = zone_demand_bottom[i]
            ref = min(sweep_level, zl if not np.isnan(zl) else price)
            sl[i] = ref - sl_buffer * atr_s[i]
        elif not np.isnan(zone_demand_bottom[i]):
            sl[i] = zone_demand_bottom[i] - sl_buffer * atr_s[i]
        else:
            sl[i] = price - 2 * sl_buffer * atr_s[i]
        tp[i] = price + rr * (price - sl[i])

    for i in np.where(signal_sell)[0]:
        price = close[i]
        sweep_level = sweep_bear_level[i]
        if not np.isnan(sweep_level):
            zt = zone_supply_top[i]
            ref = max(sweep_level, zt if not np.isnan(zt) else price)
            sl[i] = ref + sl_buffer * atr_s[i]
        elif not np.isnan(zone_supply_top[i]):
            sl[i] = zone_supply_top[i] + sl_buffer * atr_s[i]
        else:
            sl[i] = price + 2 * sl_buffer * atr_s[i]
        tp[i] = price - rr * (sl[i] - price)

    return pd.DataFrame({
        "signal": signals,
        "entry": close,
        "stop_loss": sl,
        "take_profit": tp,
        "buy_score": buy_score,
        "sell_score": sell_score,
    }, index=df.index)
