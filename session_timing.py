import numpy as np
import pandas as pd
from datetime import datetime


# --- ICT Killzone Definitions (UTC) ---
KILLZONES = {
    "asian":       {"start": 0,  "end": 8,  "label": "Asian (Accumulation)"},
    "london":      {"start": 7,  "end": 10, "label": "London (Manipulation)"},
    "ny_am":       {"start": 12, "end": 15, "label": "NY AM (Distribution)"},
    "ny_pm":       {"start": 15, "end": 17, "label": "NY PM / London Close"},
}

# Silver Bullet windows (UTC) — 1-hour high-probability scalping windows
SILVER_BULLET_WINDOWS = [
    (10, 11),   # 10:00-11:00 UTC (London mid-session)
    (15, 16),   # 15:00-16:00 UTC (NY session)
]


def _minutes_of(ts) -> int:
    return ts.hour * 60 + ts.minute


def get_current_killzone(now_utc) -> str | None:
    """Return the name of the current ICT killzone, or None."""
    t = _minutes_of(now_utc)
    for name, kz in KILLZONES.items():
        start_m = kz["start"] * 60
        end_m = kz["end"] * 60
        if start_m <= t < end_m:
            return name
    return None


def get_killzone_label(kz_name: str) -> str:
    return KILLZONES.get(kz_name, {}).get("label", "Unknown")


def is_silver_bullet(now_utc) -> bool:
    """Check if current time is within a Silver Bullet window."""
    t = _minutes_of(now_utc)
    for start_h, end_h in SILVER_BULLET_WINDOWS:
        if start_h * 60 <= t < end_h * 60:
            return True
    return False


def detect_power_of_3(df: pd.DataFrame, now_utc, cfg: dict) -> dict:
    """
    Detect Power of 3 (AMD) session phases:
    - Accumulation: Asian session range formation (00:00-08:00 UTC)
    - Manipulation: London open sweeps Asian range (07:00-10:00 UTC)
    - Distribution: NY session real move (12:00-15:00 UTC)

    Returns phase info, Asian range, and whether manipulation swept the range.
    """
    if df is None or df.empty or len(df) < 20:
        return {"phase": "unknown", "asian_range": None, "manipulation_sweep": False,
                "distribution_direction": 0, "score": 0}

    cfg_timeout = cfg.get("session_timing", {})
    if not cfg_timeout.get("power_of_3_enabled", True):
        return {"phase": "unknown", "asian_range": None, "manipulation_sweep": False,
                "distribution_direction": 0, "score": 0}

    idx = df.index
    if hasattr(idx, 'tz') and idx.tz is not None:
        try:
            from datetime import timezone as _tz
            idx = idx.tz_convert("UTC")
        except Exception:
            pass

    current_t = _minutes_of(now_utc)
    current_hour = now_utc.hour

    # Identify today's date for Asian range
    today = now_utc.date()

    # Collect Asian session candles (00:00 - 08:00 UTC) from today
    asian_start_m = 0
    asian_end_m = 8 * 60

    asian_highs = []
    asian_lows = []
    asian_closes = []

    for i in range(len(df) - 1, -1, -1):
        ts = idx[i]
        try:
            if hasattr(ts, 'date'):
                candle_date = ts.date()
            else:
                continue
            if candle_date != today:
                break
            candle_t = ts.hour * 60 + ts.minute
            if asian_start_m <= candle_t < asian_end_m:
                asian_highs.append(float(df["High"].iloc[i]))
                asian_lows.append(float(df["Low"].iloc[i]))
                asian_closes.append(float(df["Close"].iloc[i]))
        except Exception:
            continue

    asian_range = None
    if asian_highs and asian_lows:
        ah = max(asian_highs)
        al = min(asian_lows)
        asian_range = {"high": ah, "low": al, "size": ah - al,
                       "mid": (ah + al) / 2}

    # Determine current phase
    phase = "unknown"
    manipulation_sweep = False
    distribution_direction = 0
    score = 0

    if 0 <= current_hour < 8:
        phase = "accumulation"
    elif 7 <= current_hour < 10:
        phase = "manipulation"
    elif 12 <= current_hour < 15:
        phase = "distribution"
    elif 15 <= current_hour < 17:
        phase = "distribution_late"
    else:
        phase = "off_session"

    # Detect manipulation sweep of Asian range
    if asian_range is not None and asian_range["size"] > 0:
        last_price = float(df["Close"].iloc[-1])
        last_high = float(df["High"].iloc[-1])
        last_low = float(df["Low"].iloc[-1])

        # Check if recent price action swept Asian high or low
        lookback = min(12, len(df))
        recent = df.iloc[-lookback:]

        swept_high = any(float(recent["High"].iloc[j]) > asian_range["high"]
                         for j in range(len(recent)))
        swept_low = any(float(recent["Low"].iloc[j]) < asian_range["low"]
                        for j in range(len(recent)))

        # Manipulation confirmed if price swept one side and reversed
        if swept_high and last_price < asian_range["high"]:
            manipulation_sweep = True
        elif swept_low and last_price > asian_range["low"]:
            manipulation_sweep = True

    # Distribution phase scoring
    if phase in ("distribution", "distribution_late") and asian_range is not None:
        last_price = float(df["Close"].iloc[-1])
        if last_price > asian_range["high"]:
            distribution_direction = 1  # bullish distribution
            score = 3
        elif last_price < asian_range["low"]:
            distribution_direction = -1  # bearish distribution
            score = 3
        elif manipulation_sweep:
            # After sweep, direction is opposite of the sweep
            last_high = float(df["High"].iloc[-1])
            last_low = float(df["Low"].iloc[-1])
            mid = asian_range["mid"]
            if last_price > mid:
                distribution_direction = 1
                score = 2
            elif last_price < mid:
                distribution_direction = -1
                score = 2

    # Accumulation phase: note the range but no directional score
    if phase == "accumulation" and asian_range is not None:
        score = 1  # Acknowledging the setup is forming

    return {
        "phase": phase,
        "asian_range": asian_range,
        "manipulation_sweep": manipulation_sweep,
        "distribution_direction": distribution_direction,
        "score": score,
        "killzone": get_current_killzone(now_utc),
    }


def detect_judas_swing(df: pd.DataFrame, now_utc, cfg: dict) -> dict:
    """
    Detect Judas Swing pattern:
    1. Price sweeps Asian session range (or recent swing high/low)
    2. Reversal with displacement (CHoCH + strong candle)
    3. This happens at London open (07:00-09:00 UTC)

    A Judas Swing is a false directional move to trap traders before the real move.
    """
    if df is None or df.empty or len(df) < 20:
        return {"detected": False, "direction": 0, "score": 0}

    cfg_js = cfg.get("session_timing", {})
    if not cfg_js.get("judas_swing_enabled", True):
        return {"detected": False, "direction": 0, "score": 0}

    current_hour = now_utc.hour
    # Judas Swing is most valid at London open
    if not (6 <= current_hour <= 10):
        return {"detected": False, "direction": 0, "score": 0}

    idx = df.index
    if hasattr(idx, 'tz') and idx.tz is not None:
        try:
            from datetime import timezone as _tz
            idx = idx.tz_convert("UTC")
        except Exception:
            pass

    today = now_utc.date()
    atr_mult = cfg_js.get("judas_displacement_atr", 1.0)

    # Get Asian range for today
    asian_highs = []
    asian_lows = []
    for i in range(len(df) - 1, -1, -1):
        ts = idx[i]
        try:
            if hasattr(ts, 'date') and ts.date() == today:
                candle_t = ts.hour * 60 + ts.minute
                if 0 <= candle_t < 8 * 60:
                    asian_highs.append(float(df["High"].iloc[i]))
                    asian_lows.append(float(df["Low"].iloc[i]))
        except Exception:
            continue

    if not asian_highs or not asian_lows:
        return {"detected": False, "direction": 0, "score": 0}

    asian_high = max(asian_highs)
    asian_low = min(asian_lows)
    asian_mid = (asian_high + asian_low) / 2

    # Check recent candles for sweep + reversal
    from zones import _atr
    atr_val = float(_atr(df, cfg.get("atr_period", 14)).iloc[-1])

    lookback = min(8, len(df))
    recent = df.iloc[-lookback:]

    # Detect sweep above Asian high then close below
    for j in range(len(recent) - 1, -1, -1):
        h = float(recent["High"].iloc[j])
        c = float(recent["Close"].iloc[j])
        o = float(recent["Open"].iloc[j])
        body = abs(c - o)

        if h > asian_high and c < asian_high and body > atr_mult * atr_val * 0.5:
            # Sweep above + close below = bearish Judas Swing
            return {
                "detected": True,
                "direction": -1,
                "sweep_level": asian_high,
                "score": 3,
                "type": "bearish_judas",
            }

    # Detect sweep below Asian low then close above
    for j in range(len(recent) - 1, -1, -1):
        l = float(recent["Low"].iloc[j])
        c = float(recent["Close"].iloc[j])
        o = float(recent["Open"].iloc[j])
        body = abs(c - o)

        if l < asian_low and c > asian_low and body > atr_mult * atr_val * 0.5:
            return {
                "detected": True,
                "direction": 1,
                "sweep_level": asian_low,
                "score": 3,
                "type": "bullish_judas",
            }

    return {"detected": False, "direction": 0, "score": 0}


def detect_silver_bullet(df: pd.DataFrame, now_utc, cfg: dict) -> dict:
    """
    Detect Silver Bullet setup:
    - Time window: 10:00-11:00 UTC or 15:00-16:00 UTC
    - Requires: recent liquidity sweep + displacement + FVG
    - Entry: in the FVG created by the displacement
    """
    if df is None or df.empty or len(df) < 10:
        return {"in_window": False, "setup_ready": False, "score": 0}

    cfg_sb = cfg.get("session_timing", {})
    if not cfg_sb.get("silver_bullet_enabled", True):
        return {"in_window": False, "setup_ready": False, "score": 0}

    in_window = is_silver_bullet(now_utc)
    if not in_window:
        return {"in_window": False, "setup_ready": False, "score": 0}

    from zones import _atr
    atr_val = float(_atr(df, cfg.get("atr_period", 14)).iloc[-1])

    # Check for recent liquidity sweep (last 6 candles)
    lookback = min(6, len(df))
    recent = df.iloc[-lookback:]

    highs_arr = recent["High"].values
    lows_arr = recent["Low"].values
    closes_arr = recent["Close"].values
    opens_arr = recent["Open"].values

    # Find swing points before the recent window
    swing_high_before = float(df["High"].iloc[-lookback - 5:-lookback].max()) if len(df) > lookback + 5 else None
    swing_low_before = float(df["Low"].iloc[-lookback - 5:-lookback].min()) if len(df) > lookback + 5 else None

    swept_high = False
    swept_low = False

    if swing_high_before is not None:
        swept_high = any(h > swing_high_before for h in highs_arr)
    if swing_low_before is not None:
        swept_low = any(l < swing_low_before for l in lows_arr)

    # Check for displacement after sweep
    has_displacement = False
    displacement_dir = 0

    for j in range(len(recent) - 1, max(0, len(recent) - 4), -1):
        body = closes_arr[j] - opens_arr[j]
        abs_body = abs(body)
        avg_body = np.mean(np.abs(np.diff(closes_arr[:j + 1], prepend=closes_arr[0]))) if j > 0 else atr_val * 0.3

        if abs_body > 1.5 * max(avg_body, atr_val * 0.3):
            has_displacement = True
            displacement_dir = 1 if body > 0 else -1
            break

    # Setup is valid if: sweep happened + displacement in same direction
    setup_ready = False
    score = 0

    if swept_low and displacement_dir == 1 and has_displacement:
        setup_ready = True
        score = 3
    elif swept_high and displacement_dir == -1 and has_displacement:
        setup_ready = True
        score = 3

    return {
        "in_window": True,
        "setup_ready": setup_ready,
        "swept_high": swept_high,
        "swept_low": swept_low,
        "displacement": has_displacement,
        "displacement_dir": displacement_dir,
        "score": score,
    }


def validate_displacement(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Validate displacement using 5-criterion framework (institutional standard):
    1. Body size: 1.5x+ average of prior 5 candle bodies
    2. Wick ratio: each wick < 20% of body
    3. One-sided pressure: close in top 20% (bullish) or bottom 20% (bearish) of range
    4. Structure break: price must break a recent swing high/low
    5. Volume: 1.5x+ average volume

    Returns displacement quality (0-5) and direction.
    """
    if df is None or df.empty or len(df) < 10:
        return {"quality": 0, "direction": 0, "criteria_met": 0, "details": []}

    cfg_d = cfg.get("session_timing", {})
    if not cfg_d.get("displacement_validation_enabled", True):
        return {"quality": 0, "direction": 0, "criteria_met": 0, "details": []}

    last = df.iloc[-1]
    curr_o = float(last["Open"])
    curr_h = float(last["High"])
    curr_l = float(last["Low"])
    curr_c = float(last["Close"])
    curr_body = abs(curr_c - curr_o)
    curr_range = curr_h - curr_l if curr_h > curr_l else 0.001

    if curr_body == 0:
        return {"quality": 0, "direction": 0, "criteria_met": 0, "details": []}

    direction = 1 if curr_c > curr_o else -1

    # Average body of prior 5 candles
    prior = df.iloc[-6:-1]
    prior_bodies = np.abs(prior["Close"].values - prior["Open"].values)
    avg_body = np.mean(prior_bodies) if len(prior_bodies) > 0 else curr_body

    # Criterion 1: Body 1.5x+ average
    body_ratio = curr_body / max(avg_body, 0.001)
    c1 = body_ratio >= cfg_d.get("displacement_body_ratio", 1.5)

    # Criterion 2: Wicks < 20% of body each side
    upper_wick = curr_h - max(curr_o, curr_c)
    lower_wick = min(curr_o, curr_c) - curr_l
    upper_wick_ratio = upper_wick / max(curr_body, 0.001)
    lower_wick_ratio = lower_wick / max(curr_body, 0.001)
    c2 = upper_wick_ratio < cfg_d.get("displacement_max_wick_pct", 0.20) and \
         lower_wick_ratio < cfg_d.get("displacement_max_wick_pct", 0.20)

    # Criterion 3: One-sided pressure (close in top/bottom 20% of range)
    if curr_range > 0:
        close_pct = (curr_c - curr_l) / curr_range
        if direction == 1:
            c3 = close_pct >= 0.80  # close in top 20%
        else:
            c3 = close_pct <= 0.20  # close in bottom 20%
    else:
        c3 = False

    # Criterion 4: Structure break (break a recent swing)
    from market_structure import detect_swings
    swings = detect_swings(df, cfg)
    c4 = False
    if direction == 1:
        for sh in reversed(swings.get("highs", [])):
            if sh["index"] < len(df) - 1 and curr_c > sh["price"]:
                c4 = True
                break
    else:
        for sl in reversed(swings.get("lows", [])):
            if sl["index"] < len(df) - 1 and curr_c < sl["price"]:
                c4 = True
                break

    # Criterion 5: Volume 1.5x+ average
    volumes = df["Volume"].values
    vol_avg = np.mean(volumes[max(0, len(volumes) - 20):-1]) if len(volumes) > 20 else np.mean(volumes[:-1])
    c5 = float(last["Volume"]) > cfg_d.get("displacement_volume_ratio", 1.5) * vol_avg if vol_avg > 0 else False

    criteria_met = sum([c1, c2, c3, c4, c5])
    details = []
    if c1:
        details.append(f"body {body_ratio:.1f}x")
    if c2:
        details.append("clean wicks")
    if c3:
        details.append("one-sided")
    if c4:
        details.append("broke structure")
    if c5:
        details.append("vol confirmed")

    return {
        "quality": criteria_met,
        "direction": direction,
        "criteria_met": criteria_met,
        "body_ratio": round(body_ratio, 2),
        "wick_clean": c2,
        "one_sided": c3,
        "structure_break": c4,
        "volume_confirmed": c5,
        "details": details,
        "score": criteria_met,  # 0-5 score
    }


def session_timing_score(df: pd.DataFrame, now_utc, cfg: dict) -> dict:
    """
    Master session timing scorer. Combines:
    - Current killzone
    - Power of 3 phase
    - Judas Swing detection
    - Silver Bullet window
    - Displacement validation

    Returns a consolidated score and all sub-scores.
    """
    cfg_st = cfg.get("session_timing", {})
    if not cfg_st.get("enabled", True):
        return {"total_score": 0, "phase": "unknown", "details": []}

    kz = get_current_killzone(now_utc)
    amd = detect_power_of_3(df, now_utc, cfg)
    judas = detect_judas_swing(df, now_utc, cfg)
    sb = detect_silver_bullet(df, now_utc, cfg)
    disp = validate_displacement(df, cfg)

    total = 0
    details = []

    # Killzone alignment
    if kz:
        if kz in ("london", "ny_am"):
            total += 1
            details.append(f"KTZ: {get_killzone_label(kz)}")
        elif kz == "asian":
            details.append("KTZ: Asian accumulation")

    # Power of 3
    if amd["score"] > 0:
        total += amd["score"]
        details.append(f"AMD: {amd['phase']} (score {amd['score']})")
        if amd["asian_range"]:
            details.append(f"Asian: {amd['asian_range']['low']:.2f}-{amd['asian_range']['high']:.2f}")
        if amd["manipulation_sweep"]:
            details.append("Manipulation: sweep confirmed")

    # Judas Swing
    if judas["detected"]:
        total += judas["score"]
        details.append(f"Judas: {judas.get('type', 'unknown')}")

    # Silver Bullet
    if sb["in_window"]:
        total += 1
        details.append("Silver Bullet: window active")
        if sb["setup_ready"]:
            total += sb["score"]
            details.append("Silver Bullet: SETUP READY")

    # Displacement (score 0-5, add as bonus)
    if disp["quality"] >= 3:
        total += 1
        details.append(f"Displacement: {disp['quality']}/5 ({', '.join(disp['details'][:3])})")

    return {
        "total_score": total,
        "killzone": kz,
        "killzone_label": get_killzone_label(kz) if kz else None,
        "phase": amd["phase"],
        "asian_range": amd["asian_range"],
        "manipulation_sweep": amd["manipulation_sweep"],
        "distribution_direction": amd["distribution_direction"],
        "judas_swing": judas,
        "silver_bullet": sb,
        "displacement": disp,
        "details": details,
    }
