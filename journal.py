"""Auto-log MJBot entries to the TradeDesk ERP journal.

Posts every confirmed signal to /api/trades as an open trade, then watches
price until SL or TP is touched and closes the trade via PUT - giving the
bot a measurable win-rate record over time.

API shape (verified against erp-gamma-eight.vercel.app):
    POST   /api/trades      flat trade object -> created row (with id)
    PUT    /api/trades/:id  partial update
"""
import os
import sys
import time
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_URL = "https://erp-gamma-eight.vercel.app"
_open: dict[str, list[dict]] = {}
_timeout = 20
_close_retry_limit = 5
_close_failures: dict[int, int] = {}
_last_eod_close_date: str | None = None


def _settings(cfg: dict):
    j = cfg.get("journal", {})
    return (
        j.get("url", DEFAULT_URL).rstrip("/"),
        int(j.get("account_id", 2)),
        float(j.get("lot_size", 1)),
        bool(j.get("enabled", False)),
        int(j.get("eod_close_hour_utc", 21)),
        int(j.get("max_hold_hours", 24)),
    )


def _note(result: dict) -> str:
    d = result.get("details", {})
    bits = []
    if d.get("htf_direction"):
        bits.append(f"Trend {d['htf_direction']}")
    if d.get("p_win") is not None:
        bits.append(f"WinProb {d['p_win'] * 100:.0f}%")
    if d.get("fib"):
        bits.append("Fib OTE")
    for c in d.get("context_lines", []):
        if c.startswith("Session:"):
            bits.append("Sess " + c.split(":", 1)[1].strip().split()[0])
            break
    for k in ("pd_zone", "zone", "sweep", "structure_event"):
        if d.get(k):
            bits.append(str(d[k]))
    return "MJBot | " + " | ".join(bits)


def _close_trade(pair: str, t: dict, exit_price: float, outcome: str,
                 base: str, cfg: dict, current_price: float | None = None) -> tuple[str | None, dict | None]:
    """PUT close a single trade. Returns (message, outcome_dict) or (None, None)."""
    trade_id = t.get("id")
    if not trade_id:
        return None, None
    close_payload = dict(t.get("payload") or {})
    close_payload.update({
        "status": "closed",
        "exit_price": round(exit_price, 2),
        "exit_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    })
    try:
        requests.put(
            f"{base}/api/trades/{trade_id}",
            json=close_payload,
            timeout=_timeout,
        ).raise_for_status()
    except Exception as e:
        _close_failures[trade_id] = _close_failures.get(trade_id, 0) + 1
        if _close_failures[trade_id] >= _close_retry_limit:
            print(f"[journal] close FAILED {_close_failures[trade_id]}x for #{trade_id}, removing from tracking: {e}",
                  file=sys.stderr)
            return None, None
        print(f"[journal] close failed for #{trade_id} (attempt {_close_failures[trade_id]}): {e}",
              file=sys.stderr)
        return None, None

    _close_failures.pop(trade_id, None)
    entry_price = t.get("payload", {}).get("entry_price") or t.get("payload", {}).get("entry", 0)
    sl = t["sl"]
    risk = abs(float(entry_price) - sl)
    r_mult = (exit_price - float(entry_price)) / risk if risk > 0 else 0
    if t["direction"] == "short":
        r_mult = -r_mult
    emoji = "[TP]" if outcome == "TP HIT" else "[SL]" if outcome == "SL HIT" else "[EOD]"
    msg = f"{emoji} Journal #{trade_id} closed - {outcome} ({t['signal']} XAUUSD)"
    oc = {
        "pair": pair,
        "signal": t.get("signal"),
        "outcome": "win" if outcome == "TP HIT" else "loss",
        "entry": float(entry_price),
        "exit": exit_price,
        "sl": sl,
        "tp": t["tp"],
        "pnl": round(exit_price - float(entry_price), 2) if t["direction"] == "long"
               else round(float(entry_price) - exit_price, 2),
        "r_multiple": round(r_mult, 2),
    }
    return msg, oc


def log_entry(cfg: dict, pair: str, result: dict) -> dict | None:
    """Create an open trade row for a fired signal. Returns created row or None."""
    url, account_id, lot, enabled, _, _ = _settings(cfg)
    if not enabled:
        return None
    sig = str(result.get("signal", ""))
    direction = "long" if "BUY" in sig else "short" if "SELL" in sig else None
    if not direction:
        return None
    try:
        entry = float(result.get("entry") or 0)
        sl = float(result.get("stop_loss") or 0)
        tp = float(result.get("take_profit") or 0)
        tp1 = float(result.get("tp1") or tp)
        tp2 = float(result.get("tp2") or tp)
    except (TypeError, ValueError):
        return None
    if not entry or not sl or not tp:
        return None

    payload = {
        "account_id": account_id,
        "symbol": "XAUUSD",
        "direction": direction,
        "status": "open",
        "entry_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "exit_date": None,
        "entry_price": round(entry, 2),
        "exit_price": None,
        "quantity": lot,
        "stop_loss": round(sl, 2),
        "take_profit": round(tp, 2),
        "fees": 0,
        "strategy": "MJBOT",
        "notes": _note(result)[:500] + f" | TP1:{round(tp1,2)} TP2:{round(tp2,2)}",
    }
    try:
        r = requests.post(f"{url}/api/trades", json=payload, timeout=_timeout)
        r.raise_for_status()
        row = r.json()
    except Exception as e:
        print(f"[journal] log failed: {e}", file=sys.stderr)
        return None

    trade_record = {
        "id": row.get("id"),
        "payload": payload,
        "direction": direction,
        "sl": sl,
        "tp": tp,
        "signal": sig,
        "opened_at": time.time(),
    }

    existing = _open.get(pair) or []
    existing.append(trade_record)
    _open[pair] = existing

    print(f"[journal] logged #{row.get('id')} {sig} @ {entry:.2f}")
    return row


def log_pending_trigger(cfg: dict, pair: str, plan: dict) -> dict | None:
    """Journal a pending-plan fill when its trigger level is touched."""
    result_like = {
        "signal": "STRONG_BUY" if plan.get("side") == "long" else "STRONG_SELL",
        "entry": plan.get("entry"),
        "stop_loss": plan.get("sl"),
        "take_profit": plan.get("tp"),
        "details": {"htf_direction": (plan.get("mtf") or "").split("->")[0].strip() or None,
                    "zone": f"{plan.get('zone_name')} pullback fill"},
    }
    return log_entry(cfg, pair, result_like)


def check_exits(cfg: dict, prices: dict[str, dict]) -> tuple[list[str], list[dict]]:
    """Close open bot trades whose TP or SL was touched.

    ``prices`` maps each pair to a dict with keys ``close``, ``high``, and ``low``
    (the latest candle OHLC values).  A trade is closed when the candle *range*
    reaches TP or SL, not just the close price.
    """
    url, _, _, enabled, _, _ = _settings(cfg)
    if not enabled or not _open:
        return [], []
    msgs = []
    outcomes = []
    for pair, trades in list(_open.items()):
        candle = prices.get(pair)
        if not candle:
            continue
        high = candle.get("high", candle.get("close", 0))
        low = candle.get("low", candle.get("close", 0))
        close_price = candle.get("close", 0)

        for t in list(trades):
            if not t.get("id"):
                continue

            hit_tp = high >= t["tp"] if t["direction"] == "long" else low <= t["tp"]
            hit_sl = low <= t["sl"] if t["direction"] == "long" else high >= t["sl"]

            if hit_tp and hit_sl:
                # Both hit in same candle – assume SL hit first (conservative)
                hit_tp = False

            if not (hit_tp or hit_sl):
                continue

            exit_price = t["tp"] if hit_tp else t["sl"]
            outcome_str = "TP HIT" if hit_tp else "SL HIT"
            msg, oc = _close_trade(pair, t, exit_price, outcome_str, url, cfg, close_price)
            if msg:
                msgs.append(msg)
                outcomes.append(oc)
                trades.remove(t)

        if not trades:
            _open.pop(pair, None)
    return msgs, outcomes


def check_timeout(cfg: dict) -> tuple[list[str], list[dict]]:
    """Force-close trades that have been open longer than max_hold_hours."""
    url, _, _, enabled, _, max_hold = _settings(cfg)
    if not enabled or not _open:
        return [], []
    msgs = []
    outcomes = []
    now = time.time()
    max_secs = max_hold * 3600

    for pair, trades in list(_open.items()):
        for t in list(trades):
            opened = t.get("opened_at", now)
            if now - opened < max_secs:
                continue
            msg, oc = _close_trade(pair, t, t.get("sl", 0), "TIMEOUT", url, cfg)
            if msg:
                msgs.append(msg)
                outcomes.append(oc)
                trades.remove(t)

        if not trades:
            _open.pop(pair, None)
    return msgs, outcomes


def check_eod_close(cfg: dict) -> tuple[list[str], list[dict]]:
    """Force-close all open trades at the configured EOD hour (once per day)."""
    global _last_eod_close_date
    url, _, _, enabled, eod_hour, _ = _settings(cfg)
    if not enabled or not _open:
        return [], []
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    if now_utc.hour < eod_hour:
        return [], []
    if _last_eod_close_date == today_str:
        return [], []

    _last_eod_close_date = today_str
    msgs = []
    outcomes = []

    for pair, trades in list(_open.items()):
        for t in list(trades):
            entry_price = t.get("payload", {}).get("entry_price") or 0
            msg, oc = _close_trade(pair, t, float(entry_price), "EOD CLOSE", url, cfg)
            if msg:
                msgs.append(msg)
                outcomes.append(oc)
                trades.remove(t)

        if not trades:
            _open.pop(pair, None)
    return msgs, outcomes


def force_close_all(cfg: dict, reason: str = "SHUTDOWN") -> tuple[list[str], list[dict]]:
    """Force-close every tracked open trade. Used on bot shutdown."""
    url, _, _, enabled, _, _ = _settings(cfg)
    if not enabled or not _open:
        return [], []
    msgs = []
    outcomes = []

    for pair, trades in list(_open.items()):
        for t in list(trades):
            entry_price = t.get("payload", {}).get("entry_price") or 0
            msg, oc = _close_trade(pair, t, float(entry_price), reason, url, cfg)
            if msg:
                msgs.append(msg)
                outcomes.append(oc)
                trades.remove(t)

        if not trades:
            _open.pop(pair, None)
    return msgs, outcomes


def restore_open(existing_rows: list[dict]) -> None:
    """Adopt trades already logged by MJBOT so exits keep being tracked."""
    fields = ("account_id", "symbol", "direction", "status", "entry_date", "exit_date",
              "entry_price", "exit_price", "quantity", "stop_loss", "take_profit",
              "fees", "strategy", "notes")
    restored = {}
    for row in existing_rows or []:
        if row.get("strategy") != "MJBOT" or row.get("status") != "open":
            continue
        symbol_pair = "GC=F"
        entry_date_str = row.get("entry_date", "")
        opened_at = time.time()
        if entry_date_str:
            try:
                entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                opened_at = entry_dt.timestamp()
            except (ValueError, TypeError):
                pass
        trade_record = {
            "id": row.get("id"),
            "payload": {k: row.get(k) for k in fields},
            "direction": row.get("direction"),
            "sl": float(row.get("stop_loss") or 0),
            "tp": float(row.get("take_profit") or 0),
            "signal": "BUY" if row.get("direction") == "long" else "SELL",
            "opened_at": opened_at,
        }
        restored.setdefault(symbol_pair, []).append(trade_record)

    _open.clear()
    _open.update(restored)


def fetch_open_bot_trades(cfg: dict) -> list[dict]:
    url, _, _, enabled, _, _ = _settings(cfg)
    if not enabled:
        return []
    try:
        r = requests.get(f"{url}/api/trades", timeout=_timeout)
        r.raise_for_status()
        rows = r.json()
        if isinstance(rows, dict):
            rows = rows.get("trades", [])
        return [x for x in rows if x.get("strategy") == "MJBOT"]
    except Exception as e:
        print(f"[journal] fetch failed: {e}", file=sys.stderr)
        return []


def open_count() -> int:
    return sum(len(v) for v in _open.values())


def open_trades_snapshot() -> list[dict]:
    result = []
    for pair, trades in _open.items():
        for t in trades:
            result.append({
                "id": t.get("id"),
                "pair": pair,
                "direction": t.get("direction"),
                "signal": t.get("signal"),
                "entry": t.get("payload", {}).get("entry_price"),
                "sl": t.get("sl"),
                "tp": t.get("tp"),
            })
    return result
