"""MJBot Pattern Detection — Sweep-Retest-Entry and advanced ICT patterns.

Detects when price:
1. Sweeps liquidity (grabs stops above/below a key level)
2. Reverses with displacement (big body candle)
3. Retraces to the FVG/OB left by the sweep — the entry zone

This catches the high-probability retest after a liquidity grab.
"""
import numpy as np
import pandas as pd


def detect_sweep_retest(df: pd.DataFrame, cfg: dict, fvg: dict = None,
                        obs: dict = None, pools: dict = None) -> dict:
    """Detect sweep-retest-entry pattern.

    Returns dict with:
        detected: bool
        direction: "bearish" | "bullish" | None
        sweep_price: float — the price that was swept
        retrace_zone: (lo, hi) — the FVG/OB zone where retest happens
        current_in_zone: bool — is price currently in the retrace zone
        displacement_size: float — how big was the reversal candle
        confidence: float 0-1
    """
    result = {
        "detected": False,
        "direction": None,
        "sweep_price": None,
        "retrace_zone": None,
        "current_in_zone": False,
        "displacement_size": 0,
        "confidence": 0,
    }

    if df is None or len(df) < 20:
        return result

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    opn = df["Open"].values
    n = len(close)

    atr_period = cfg.get("atr_period", 14)
    lookback = cfg.get("sweep_window", 4)
    pivot_left = cfg.get("sweep_pivot", 3)
    pivot_right = cfg.get("sweep_pivot", 3)

    # Compute ATR
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)),
                   np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(atr_period).mean().values
    if np.isnan(atr[-1]) or atr[-1] <= 0:
        return result
    atr_val = atr[-1]
    current_price = close[-1]

    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    for i in range(pivot_left, n - pivot_right):
        if high[i] == max(high[i - pivot_left:i + pivot_left + 1]):
            swing_highs.append((i, high[i]))
        if low[i] == min(low[i - pivot_left:i + pivot_left + 1]):
            swing_lows.append((i, low[i]))

    # --- BEARISH PATTERN: Sweep high, drop, retrace to sell ---
    # Look for a recent swing high that swept a previous swing high
    recent_highs = [sh for sh in swing_highs if sh[0] >= n - 60]
    if len(recent_highs) >= 2:
        # Check if the latest swing high swept above an earlier one
        for i in range(len(recent_highs) - 1, 0, -1):
            sweep_idx, sweep_price = recent_highs[i]
            prev_idx, prev_price = recent_highs[i - 1]

            # Sweep: current high above previous high
            if sweep_price > prev_price:
                # Check for displacement after the sweep — big bear candle
                after_sweep_start = sweep_idx + 1
                if after_sweep_start >= n:
                    continue

                # Look for a big bearish candle within 5 bars of the sweep
                disp_found = False
                disp_size = 0
                disp_idx = None
                for j in range(after_sweep_start, min(after_sweep_start + 5, n)):
                    body = opn[j] - close[j]  # bear candle: open > close
                    if body > atr_val * 1.0:
                        disp_found = True
                        disp_size = body
                        disp_idx = j
                        break

                if not disp_found:
                    continue

                # Check if price broke below a swing low after the sweep
                swing_low_after = [sl for sl in swing_lows
                                   if sl[0] > sweep_idx and sl[0] < n - 1]
                if not swing_low_after:
                    continue

                # Price dropped below the nearest swing low after sweep
                drop_target = swing_low_after[0][1]
                dropped_below = np.any(close[after_sweep_start:] < drop_target)
                if not dropped_below:
                    continue

                # Now check: is current price retracing toward the sweep zone?
                # The retrace zone is the FVG or OB area between the sweep high and the drop
                retrace_lo = sweep_price - 2 * atr_val
                retrace_hi = sweep_price + 0.5 * atr_val

                # If we have FVG data, use the bearish FVG near the sweep
                if fvg:
                    for fv in fvg.get("bearish_fvgs", []):
                        fv_top = fv.get("top", 0)
                        fv_bottom = fv.get("bottom", 0)
                        if abs(fv_top - sweep_price) < 3 * atr_val:
                            retrace_lo = fv_bottom
                            retrace_hi = fv_top
                            break

                # If we have OB data, use the bearish OB near the sweep
                if obs:
                    for ob in obs.get("bearish", []):
                        ob_top = ob.get("top", 0)
                        ob_bottom = ob.get("bottom", 0)
                        if abs(ob_top - sweep_price) < 3 * atr_val:
                            retrace_lo = min(retrace_lo, ob_bottom)
                            retrace_hi = max(retrace_hi, ob_top)
                            break

                # Check if price is currently in the retrace zone
                in_zone = retrace_lo <= current_price <= retrace_hi

                # Confidence scoring
                conf = 0.5  # base
                if in_zone:
                    conf += 0.2
                if disp_size > atr_val * 1.5:
                    conf += 0.1
                bars_since = n - 1 - sweep_idx
                if 3 <= bars_since <= 30:
                    conf += 0.1
                if in_zone and current_price > close[-2]:
                    conf += 0.1  # slight bounce in zone

                result.update({
                    "detected": True,
                    "direction": "bearish",
                    "sweep_price": round(sweep_price, 2),
                    "retrace_zone": (round(retrace_lo, 2), round(retrace_hi, 2)),
                    "current_in_zone": in_zone,
                    "displacement_size": round(disp_size, 2),
                    "confidence": round(min(conf, 1.0), 2),
                    "bars_since_sweep": bars_since,
                })
                return result

    # --- BULLISH PATTERN: Sweep low, rally, retrace to buy ---
    recent_lows = [sl for sl in swing_lows if sl[0] >= n - 60]
    if len(recent_lows) >= 2:
        for i in range(len(recent_lows) - 1, 0, -1):
            sweep_idx, sweep_price = recent_lows[i]
            prev_idx, prev_price = recent_lows[i - 1]

            # Sweep: current low below previous low
            if sweep_price < prev_price:
                after_sweep_start = sweep_idx + 1
                if after_sweep_start >= n:
                    continue

                disp_found = False
                disp_size = 0
                for j in range(after_sweep_start, min(after_sweep_start + 5, n)):
                    body = close[j] - opn[j]  # bull candle: close > open
                    if body > atr_val * 1.0:
                        disp_found = True
                        disp_size = body
                        break

                if not disp_found:
                    continue

                swing_high_after = [sh for sh in swing_highs
                                    if sh[0] > sweep_idx and sh[0] < n - 1]
                if not swing_high_after:
                    continue

                rally_target = swing_high_after[0][1]
                rallied_above = np.any(close[after_sweep_start:] > rally_target)
                if not rallied_above:
                    continue

                retrace_lo = sweep_price - 0.5 * atr_val
                retrace_hi = sweep_price + 2 * atr_val

                if fvg:
                    for fv in fvg.get("bullish_fvgs", []):
                        fv_top = fv.get("top", 0)
                        fv_bottom = fv.get("bottom", 0)
                        if abs(fv_bottom - sweep_price) < 3 * atr_val:
                            retrace_lo = fv_bottom
                            retrace_hi = fv_top
                            break

                if obs:
                    for ob in obs.get("bullish", []):
                        ob_top = ob.get("top", 0)
                        ob_bottom = ob.get("bottom", 0)
                        if abs(ob_bottom - sweep_price) < 3 * atr_val:
                            retrace_lo = min(retrace_lo, ob_bottom)
                            retrace_hi = max(retrace_hi, ob_top)
                            break

                in_zone = retrace_lo <= current_price <= retrace_hi

                conf = 0.5
                if in_zone:
                    conf += 0.2
                if disp_size > atr_val * 1.5:
                    conf += 0.1
                bars_since = n - 1 - sweep_idx
                if 3 <= bars_since <= 30:
                    conf += 0.1
                if in_zone and current_price < close[-2]:
                    conf += 0.1

                result.update({
                    "detected": True,
                    "direction": "bullish",
                    "sweep_price": round(sweep_price, 2),
                    "retrace_zone": (round(retrace_lo, 2), round(retrace_hi, 2)),
                    "current_in_zone": in_zone,
                    "displacement_size": round(disp_size, 2),
                    "confidence": round(min(conf, 1.0), 2),
                    "bars_since_sweep": bars_since,
                })
                return result

    return result
