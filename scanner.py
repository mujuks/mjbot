"""MJBot Gold Scanner -- Structure analysis and daily intelligence briefings.

Scans gold across multiple timeframes and produces:
  - Daily structure briefing (key levels, HTF direction, session context)
  - Multi-TF structure map (4H/1H/45M/30M/15M zones and events)
  - Key levels dashboard (support/resistance, OBs, FVGs, liquidity)
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_fetcher import fetch_forex_data
from market_structure import classify_structure, compute_premium_discount, detect_bos_choch, detect_swings
from liquidity import detect_fvg, detect_liquidity_pools, detect_volume_profile, detect_liquidity_sweep
from order_blocks import detect_order_blocks
from zones import detect_zones
from strategy import _atr, analyze

_DATA_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
_DATA_DIR.mkdir(exist_ok=True)
_STRUCTURE_STATE_FILE = _DATA_DIR / "structure_state.json"

_SCANNER_TFS = ["4h", "1h", "45m", "30m", "15m"]


def _load_structure_state() -> dict:
    if _STRUCTURE_STATE_FILE.exists():
        try:
            return json.loads(_STRUCTURE_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_structure_state(state: dict):
    _STRUCTURE_STATE_FILE.write_text(
        json.dumps(state, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def _tf_label(tf: str) -> str:
    return tf.upper() if len(tf) <= 2 else tf.rstrip("m").upper() + "M"


def _arrow(x: float) -> str:
    return "▲" if x > 0.05 else ("▼" if x < -0.05 else "▬")


def scan_timeframe(pair: str, tf: str, cfg: dict) -> dict | None:
    days_map = {"4h": 45, "1h": 40, "45m": 35, "30m": 30, "15m": 25}
    days = days_map.get(tf, int(cfg.get("lookback_days", 30)))
    try:
        df = fetch_forex_data(pair, tf, days)
    except Exception as e:
        return {"error": str(e)}

    if df is None or len(df) < 30:
        return {"error": "insufficient data"}

    result = analyze(df, cfg)
    atr = float(_atr(df, cfg.get("atr_period", 14)).iloc[-1])
    price = float(df["Close"].iloc[-1])

    swings = detect_swings(df, cfg)
    structure = classify_structure(swings)
    events = detect_bos_choch(df, structure, cfg)
    pd_zone = compute_premium_discount(df, cfg.get("pd_lookback", 50))

    zones = detect_zones(df, cfg)
    obs = detect_order_blocks(df, cfg)
    fvgs = detect_fvg(df, cfg)
    pools = detect_liquidity_pools(df, cfg)
    vp = detect_volume_profile(df, cfg)

    direction = 0
    if result.get("signal", "") in ("BUY", "STRONG_BUY"):
        direction = 1
    elif result.get("signal", "") in ("SELL", "STRONG_SELL"):
        direction = -1

    latest_event = None
    if events:
        if isinstance(events, dict):
            for key in ("choch_bull", "choch_bear", "bos_bull", "bos_bear"):
                if events.get(key):
                    direction_label = "BULL" if "bull" in key else "BEAR"
                    event_type = "CHoCH" if "choch" in key else "BOS"
                    level = events.get(f"{key.split('_')[0]}_level", 0)
                    latest_event = {
                        "type": event_type,
                        "direction": direction_label,
                        "price": round(float(level), 2) if level else 0,
                    }
                    break
        elif isinstance(events, list) and events:
            last_evt = events[-1]
            latest_event = {
                "type": last_evt.get("type", ""),
                "direction": last_evt.get("direction", ""),
                "price": round(float(last_evt.get("price", 0)), 2),
            }

    buy_s = result.get("details", {}).get("buy_score", 0)
    sell_s = result.get("details", {}).get("sell_score", 0)

    key_zones = []
    all_zones = []
    if isinstance(zones, dict):
        all_zones = (zones.get("demand", []) or []) + (zones.get("supply", []) or [])
    elif isinstance(zones, list):
        all_zones = zones
    for z in all_zones[:5]:
        key_zones.append({
            "type": z.get("type", ""),
            "lo": round(float(z.get("lo", 0)), 2),
            "hi": round(float(z.get("hi", 0)), 2),
            "touches": z.get("touches", 0),
        })

    key_obs = []
    for ob in (obs or [])[:3]:
        key_obs.append({
            "type": ob.get("type", ""),
            "lo": round(float(ob.get("lo", 0)), 2),
            "hi": round(float(ob.get("hi", 0)), 2),
            "strength": ob.get("strength", 0),
        })

    fvgs_list = []
    for f in (fvgs or [])[:3]:
        fvgs_list.append({
            "type": f.get("type", ""),
            "lo": round(float(f.get("lo", 0)), 2),
            "hi": round(float(f.get("hi", 0)), 2),
        })

    nearest_bsl = None
    nearest_ssl = None
    if pools:
        bsl = [p for p in pools if p.get("type") == "bsl"]
        ssl = [p for p in pools if p.get("type") == "ssl"]
        if bsl:
            nearest_bsl = round(min(abs(float(b.get("price", 0)) - price) for b in bsl if float(b.get("price", 0)) > price) + price, 2) if any(float(b.get("price", 0)) > price for b in bsl) else None
        if ssl:
            nearest_ssl = round(price - min(abs(price - float(s.get("price", 0))) for s in ssl if float(s.get("price", 0)) < price), 2) if any(float(s.get("price", 0)) < price for s in ssl) else None

    vp_data = None
    if vp:
        vp_data = {
            "poc": round(float(vp.get("poc", 0)), 2),
            "vah": round(float(vp.get("vah", 0)), 2),
            "val": round(float(vp.get("val", 0)), 2),
        }

    return {
        "tf": tf,
        "price": round(price, 2),
        "atr": round(atr, 2),
        "signal": result.get("signal", "NONE"),
        "direction": direction,
        "structure": structure.get("trend", "unknown") if isinstance(structure, dict) else "unknown",
        "pd_zone": pd_zone,
        "buy_score": buy_s,
        "sell_score": sell_s,
        "latest_event": latest_event,
        "zones": key_zones,
        "order_blocks": key_obs,
        "fvgs": fvgs_list,
        "nearest_bsl": nearest_bsl,
        "nearest_ssl": nearest_ssl,
        "volume_profile": vp_data,
    }


def scan_all_timeframes(pair: str, cfg: dict) -> list[dict]:
    results = []
    for tf in _SCANNER_TFS:
        try:
            scan = scan_timeframe(pair, tf, cfg)
            if scan and "error" not in scan:
                results.append(scan)
        except Exception as e:
            pass
    return results


def format_structure_map(scans: list[dict]) -> str:
    if not scans:
        return "No structure data available"
    lines = ["Multi-TF Structure Map"]
    for s in scans:
        tf = _tf_label(s["tf"])
        arrow = _arrow(s["direction"]) if s["direction"] else _arrow(0)
        event = ""
        if s.get("latest_event"):
            evt = s["latest_event"]
            event = f" [{evt['type']} {evt['direction']}]"
        lines.append(
            f"  {tf} {arrow} {s['structure']:>8} | "
            f"Zone: {s['pd_zone']:>14} | "
            f"B{s['buy_score']}/S{s['sell_score']}"
            f"{event}"
        )
    return "\n".join(lines)


def format_key_levels(scans: list[dict], price: float) -> str:
    if not scans:
        return "No key levels available"
    lines = [f"Key Levels (Price: {price:.2f})"]
    htf = scans[0] if scans else {}
    if htf.get("nearest_bsl"):
        lines.append(f"  BSL Target: {htf['nearest_bsl']:.2f}")
    if htf.get("nearest_ssl"):
        lines.append(f"  SSL Target: {htf['nearest_ssl']:.2f}")
    if htf.get("volume_profile"):
        vp = htf["volume_profile"]
        lines.append(f"  POC: {vp['poc']:.2f} | VAH: {vp['vah']:.2f} | VAL: {vp['val']:.2f}")
    all_obs = []
    for s in scans:
        for ob in s.get("order_blocks", []):
            all_obs.append({**ob, "tf": s["tf"]})
    all_obs.sort(key=lambda x: abs((x["lo"] + x["hi"]) / 2 - price))
    for ob in all_obs[:5]:
        mid = (ob["lo"] + ob["hi"]) / 2
        dist = abs(mid - price)
        lines.append(f"  OB [{_tf_label(ob['tf'])}] {ob['type']} {ob['lo']:.2f}-{ob['hi']:.2f} (dist: {dist:.1f}pts)")
    all_fvgs = []
    for s in scans:
        for f in s.get("fvgs", []):
            all_fvgs.append({**f, "tf": s["tf"]})
    for f in all_fvgs[:3]:
        lines.append(f"  FVG [{_tf_label(f['tf'])}] {f['type']} {f['lo']:.2f}-{f['hi']:.2f}")
    return "\n".join(lines)


def format_daily_briefing(pair: str, scans: list[dict], cfg: dict) -> str:
    now = datetime.now(timezone.utc)
    if not scans:
        return f"Gold Scanner: No data available ({now:%Y-%m-%d %H:%M UTC})"
    htf = scans[0]
    price = htf["price"]
    lines = [
        f"Gold Daily Briefing - {now:%Y-%m-%d %H:%M UTC}",
        f"Price: {price:.2f} | ATR(14): {htf['atr']:.2f}",
        "",
        format_structure_map(scans),
        "",
        format_key_levels(scans, price),
    ]
    htf_trend = htf.get("structure", "unknown")
    htf_dir = "BULLISH" if htf["direction"] > 0 else "BEARISH" if htf["direction"] < 0 else "NEUTRAL"
    lines.extend([
        "",
        f"HTF Bias ({_tf_label(htf['tf'])}): {htf_trend} ({htf_dir})",
        f"Entry Zone: {htf.get('pd_zone', 'unknown')}",
    ])
    sessions_best = []
    try:
        from brain import Brain
        brain = Brain(cfg)
        sessions_best = brain.best_sessions(pair)
    except Exception:
        pass
    if sessions_best:
        lines.append(f"Best Sessions: {', '.join(sessions_best)}")
    return "\n".join(lines)


def run_periodic_scan(pair: str, cfg: dict, state: dict) -> str | None:
    scanner_cfg = cfg.get("scanner", {})
    if not scanner_cfg.get("enabled", True):
        return None
    interval = scanner_cfg.get("structure_report_interval_minutes", 240) * 60
    last_scan = state.get("last_structure_scan", 0)
    now = time.time()
    if now - last_scan < interval:
        return None
    state["last_structure_scan"] = now
    try:
        scans = scan_all_timeframes(pair, cfg)
        if not scans:
            return None
        report = format_daily_briefing(pair, scans, cfg)
        structure_state = _load_structure_state()
        structure_state["last_scan"] = datetime.now(timezone.utc).isoformat()
        structure_state["htf_trend"] = scans[0].get("structure", "unknown") if scans else "unknown"
        structure_state["htf_direction"] = scans[0].get("direction", 0) if scans else 0
        structure_state["price"] = scans[0].get("price", 0) if scans else 0
        structure_state["scans"] = {s["tf"]: {k: v for k, v in s.items() if k != "tf"} for s in scans}
        _save_structure_state(structure_state)
        return report
    except Exception as e:
        print(f"[scanner] error: {e}")
        return None
