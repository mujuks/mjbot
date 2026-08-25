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
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_URL = "https://erp-gamma-eight.vercel.app"
_open: dict[str, dict] = {}
_timeout = 20


def _settings(cfg: dict):
    j = cfg.get("journal", {})
    return (
        j.get("url", DEFAULT_URL).rstrip("/"),
        int(j.get("account_id", 2)),
        float(j.get("lot_size", 1)),
        bool(j.get("enabled", False)),
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


def log_entry(cfg: dict, pair: str, result: dict) -> dict | None:
    """Create an open trade row for a fired signal. Returns created row or None."""
    base, account_id, lot, enabled = _settings(cfg)
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
        "notes": _note(result)[:500],
    }
    try:
        r = requests.post(f"{base}/api/trades", json=payload, timeout=_timeout)
        r.raise_for_status()
        row = r.json()
    except Exception as e:
        print(f"[journal] log failed: {e}", file=sys.stderr)
        return None

    _open[pair] = {
        "id": row.get("id"),
        "payload": payload,
        "direction": direction,
        "sl": sl,
        "tp": tp,
        "signal": sig,
    }
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


def check_exits(cfg: dict, prices: dict[str, float]) -> list[str]:
    """Close open bot trades whose TP or SL was touched. Returns messages."""
    base, _, _, enabled = _settings(cfg)
    if not enabled or not _open:
        return []
    msgs = []
    for pair, t in list(_open.items()):
        price = prices.get(pair)
        if not price or not t.get("id"):
            continue
        hit_tp = price >= t["tp"] if t["direction"] == "long" else price <= t["tp"]
        hit_sl = price <= t["sl"] if t["direction"] == "long" else price >= t["sl"]
        if not (hit_tp or hit_sl):
            continue
        outcome = "TP HIT" if hit_tp else "SL HIT"
        close_payload = dict(t.get("payload") or {})
        close_payload.update({
            "status": "closed",
            "exit_price": round(t["tp"] if hit_tp else t["sl"], 2),
            "exit_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        })
        try:
            requests.put(
                f"{base}/api/trades/{t['id']}",
                json=close_payload,
                timeout=_timeout,
            ).raise_for_status()
        except Exception as e:
            print(f"[journal] close failed for #{t['id']}: {e}", file=sys.stderr)
            continue
        emoji = "[TP]" if hit_tp else "[SL]"
        msgs.append(f"{emoji} Journal #{t['id']} closed - {outcome} ({t['signal']} XAUUSD)")
        del _open[pair]
    return msgs


def restore_open(existing_rows: list[dict]) -> None:
    """Adopt trades already logged by MJBOT so exits keep being tracked."""
    fields = ("account_id", "symbol", "direction", "status", "entry_date", "exit_date",
              "entry_price", "exit_price", "quantity", "stop_loss", "take_profit",
              "fees", "strategy", "notes")
    for row in existing_rows or []:
        if row.get("strategy") != "MJBOT" or row.get("status") != "open":
            continue
        symbol_pair = "GC=F"  # single-pair bot today
        _open[symbol_pair] = {
            "id": row.get("id"),
            "payload": {k: row.get(k) for k in fields},
            "direction": row.get("direction"),
            "sl": float(row.get("stop_loss") or 0),
            "tp": float(row.get("take_profit") or 0),
            "signal": "BUY" if row.get("direction") == "long" else "SELL",
        }


def fetch_open_bot_trades(cfg: dict) -> list[dict]:
    base, _, _, enabled = _settings(cfg)
    if not enabled:
        return []
    try:
        r = requests.get(f"{base}/api/trades", timeout=_timeout)
        r.raise_for_status()
        rows = r.json()
        if isinstance(rows, dict):
            rows = rows.get("trades", [])
        return [x for x in rows if x.get("strategy") == "MJBOT"]
    except Exception as e:
        print(f"[journal] fetch failed: {e}", file=sys.stderr)
        return []
