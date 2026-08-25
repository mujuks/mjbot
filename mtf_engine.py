"""Multi-timeframe confluence engine.

Runs the full SMC strategy on several timeframes and blends their directional
readings into one composite score. Higher timeframes carry more weight, the
entry timeframe is taken from the live analysis, and at most one cached
timeframe is refreshed per call so polling stays cheap.

Verdict semantics vs the entry direction:
  - composite opposes entry beyond -0.15  -> veto (demote firm signals)
  - composite agrees beyond +0.55         -> high-confluence tag
  - anything between                      -> neutral
"""
import logging
import math
import time

from data_fetcher import fetch_forex_data
from strategy import analyze

log = logging.getLogger("mtf")

# (timeframe, weight, cache TTL seconds); 5m is served live by the bot loop.
TIMEFRAMES = [
    ("4h", 0.30, 3600),
    ("1h", 0.26, 1200),
    ("45m", 0.24, 900),
    ("15m", 0.20, 600),
    ("5m", 0.18, 0),
]

_ENTRY_TF = "5m"
_VETO_LEVEL = -0.15
_STRONG_LEVEL = 0.55
_SCALP_MIN_15M_ALIGN = 0.30  # 15m must agree at least this strongly for firm scalp signals

_FIRM_DIR = {
    "STRONG_BUY": 1.0, "BUY": 1.0,
    "STRONG_SELL": -1.0, "SELL": -1.0,
}
_WATCH_DIR = {"WATCH_BUY": 0.5, "WATCH_SELL": -0.5}

_cache: dict[str, dict[str, tuple[float, dict]]] = {}


def _arrow(x: float) -> str:
    return "\u25b2" if x > 0.05 else ("\u25bc" if x < -0.05 else "\u25ac")


def _label(tf: str) -> str:
    return tf.upper() if len(tf) <= 2 else tf.rstrip("m").upper() + "M"


def _tf_direction(result: dict, threshold: float) -> float:
    """Continuous direction in [-1, 1]: firm > watch > score-edge."""
    sig = result.get("signal")
    if sig in _FIRM_DIR:
        d = _FIRM_DIR[sig]
    elif sig in _WATCH_DIR:
        d = _WATCH_DIR[sig]
    else:
        b = result.get("details", {}).get("buy_score", 0)
        s = result.get("details", {}).get("sell_score", 0)
        d = math.tanh((b - s) / max(float(threshold), 1.0))
    return d


def _run_tf(pair: str, tf: str, cfg: dict) -> tuple:
    days = {"4h": 45, "1h": 40, "45m": 35}.get(tf, int(cfg.get("lookback_days", 30)))
    df = fetch_forex_data(pair, tf, days)
    res = analyze(df, cfg)
    return df, res


def assess(pair: str, cfg: dict, entry_result: dict | None = None) -> dict | None:
    """Blend cached/live timeframe readings into a composite confluence verdict.

    Refreshes at most one stale higher-timeframe per call (all of them on a
    cold cache) so a poll cycle never blocks on several downloads.
    """
    if not cfg.get("mtf", {}).get("enabled", True):
        return None
    threshold = float(cfg.get("signal_threshold", 3))
    now = time.time()
    pc = _cache.setdefault(pair, {})
    table: list[tuple[str, float, float]] = []  # (tf, dir, weight)
    refresh_budget = 2 if not pc else 1

    for tf, weight, ttl in TIMEFRAMES:
        if tf == _ENTRY_TF:
            if entry_result is not None:
                pc[tf] = (now, entry_result)
                table.append((tf, _tf_direction(entry_result, threshold), weight))
            continue

        cached = pc.get(tf)
        if cached and now - cached[0] < ttl:
            table.append((tf, _tf_direction(cached[1], threshold), weight))
            continue
        if refresh_budget <= 0:
            if cached:
                table.append((tf, _tf_direction(cached[1], threshold), weight))
            continue

        try:
            _, res = _run_tf(pair, tf, cfg)
        except Exception as e:
            log.warning("%s %s refresh failed: %s", pair, tf, e)
            if cached:
                table.append((tf, _tf_direction(cached[1], threshold), weight))
            continue
        refresh_budget -= 1
        pc[tf] = (now, res)
        table.append((tf, _tf_direction(res, threshold), weight))

    if not table or entry_result is None:
        return None

    w_sum = sum(w for _, _, w in table)
    composite = sum(d * w for _, d, w in table) / w_sum if w_sum else 0.0

    entry_sig = entry_result.get("signal", "")
    entry_dir = 1.0 if "BUY" in entry_sig else -1.0 if "SELL" in entry_sig else 0.0
    aligned = composite * entry_dir if entry_dir else 0.0

    # Scalper-specific: check if 15m aligns with entry direction
    scalp_15m_aligned = True
    if entry_dir:
        for tf, d, _ in table:
            if tf == "15m":
                if abs(d) >= 0.10:
                    scalp_15m_aligned = (d * entry_dir) >= _SCALP_MIN_15M_ALIGN
                break

    parts = [f"{_label(tf)}{_arrow(d)}" for tf, d, _ in table]
    summary = f"{' \u00b7 '.join(parts)} -> {abs(composite) * 100:.0f}% {'LONG' if composite > 0 else 'SHORT' if composite < 0 else 'FLAT'}"

    return {
        "composite": round(composite, 3),
        "summary": summary,
        "rows": [(tf, round(d, 2)) for tf, d, _ in table],
        "veto": bool(entry_dir) and aligned < _VETO_LEVEL,
        "high_confluence": bool(entry_dir) and aligned >= _STRONG_LEVEL,
        "scalp_15m_aligned": scalp_15m_aligned,
    }


def reset_cache() -> None:
    _cache.clear()
