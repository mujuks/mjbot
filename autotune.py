"""MJBot self-tuner: reads its own journal results and adjusts the gates.

Runs once a week (Sunday 21:00 UTC, called from the bot polling loop).

What it measures (from closed MJBOT trades in the ERP):
  - overall rolling win rate          -> nudges quant.min_probability
  - Fib OTE-tagged trade win rate     -> can disable strategy.use_fib
  - session-tagged performance        -> report only (no auto change yet)

Rules of engagement:
  - needs >= MIN_TRADES samples before touching anything
  - moves one notch (0.05) per week, bounded [0.40, 0.60]
  - every finding + action is messaged to Telegram and appended to
    data/autotune_log.json so nothing changes silently
"""
import json
import os
import sys
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_TRADES = 10
MIN_FIB_TRADES = 8
FLOOR, CAP, STEP = 0.40, 0.60, 0.05
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "autotune_log.json")


def _closed_bot_trades(cfg: dict) -> list[dict]:
    j = cfg.get("journal", {})
    if not j.get("enabled"):
        return []
    base = j.get("url", "https://erp-gamma-eight.vercel.app").rstrip("/")
    try:
        r = requests.get(f"{base}/api/trades", timeout=20)
        r.raise_for_status()
        rows = r.json()
        if isinstance(rows, dict):
            rows = rows.get("trades", [])
        return [x for x in rows
                if x.get("strategy") == "MJBOT" and x.get("status") == "closed"]
    except Exception as e:
        print(f"[autotune] fetch failed: {e}", file=sys.stderr)
        return []


def _is_win(row: dict) -> bool | None:
    try:
        pnl = row.get("pnl")
        if pnl is not None:
            return float(pnl) > 0
        ep, xp = float(row["entry_price"]), float(row["exit_price"])
        if row.get("direction") == "long":
            return xp > ep
        if row.get("direction") == "short":
            return xp < ep
    except (TypeError, ValueError, KeyError):
        pass
    return None


def _win_rate(rows: list[dict]) -> tuple[int, float]:
    outcomes = [_is_win(r) for r in rows]
    decided = [o for o in outcomes if o is not None]
    if not decided:
        return 0, 0.0
    return len(decided), sum(decided) / len(decided)


def _tagged(rows: list[dict], tag: str) -> list[dict]:
    return [r for r in rows if tag.lower() in str(r.get("notes", "")).lower()]


def _log(entry: dict) -> None:
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        log = []
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, encoding="utf-8") as f:
                log = json.load(f)
        log.append(entry)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log[-200:], f, indent=1)
    except Exception as e:
        print(f"[autotune] log write failed: {e}", file=sys.stderr)


def run_weekly_tune(cfg: dict) -> str:
    """Inspect journal results; adjust config gates. Returns Telegram summary."""
    now = datetime.now(timezone.utc)
    trades = _closed_bot_trades(cfg)
    lines = ["[AUTO] Weekly self-review"]

    if len(trades) < MIN_TRADES:
        msg = (f"{lines[0]}\n"
               f"Only {len(trades)} closed MJBOT trades - need {MIN_TRADES} "
               f"before tuning. Gates unchanged.")
        _log({"ts": now.isoformat(), "action": "skip", "reason": "insufficient data",
              "closed_trades": len(trades)})
        return msg

    n, wr = _win_rate(trades)
    lines.append(f"Closed trades: {n} | win rate {wr * 100:.0f}%")

    # --- overall gate adjustment ---
    changed = []
    qcfg = cfg.setdefault("quant", {})
    cur_p = float(qcfg.get("min_probability", 0.45))
    if wr < 0.45 and cur_p < CAP:
        new_p = min(CAP, cur_p + STEP)
        qcfg["min_probability"] = round(new_p, 2)
        changed.append(f"market sagging -> probability floor {cur_p:.2f} -> {new_p:.2f}")
    elif wr > 0.55 and cur_p > FLOOR:
        new_p = max(FLOOR, cur_p - STEP)
        qcfg["min_probability"] = round(new_p, 2)
        changed.append(f"performing well -> probability floor {cur_p:.2f} -> {new_p:.2f}")
    else:
        lines.append(f"Probability floor stays {cur_p:.2f}")

    # --- fib bonus evaluation ---
    fib_rows = _tagged(trades, "fib ote")
    nf, wfr = _win_rate(fib_rows)
    if cfg.get("use_fib", True) and nf:
        lines.append(f"Fib OTE trades: {nf} | win rate {wfr * 100:.0f}%")
        if nf >= MIN_FIB_TRADES and wfr < 0.45:
            cfg["use_fib"] = False
            changed.append("fib signals underperforming -> OTE bonus disabled")
    elif not cfg.get("use_fib", True):
        lines.append("Fib OTE bonus currently disabled")

    # --- session breakdown (report-only for now) ---
    for label in ("PEAK", "SLOW"):
        rows = _tagged(trades, f"sess {label}")
        if rows:
            ns, ws = _win_rate(rows)
            lines.append(f"{label} session: {ns} trades @ {ws * 100:.0f}%")

    if changed:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        lines.append("\n[APPLIED]:")
        lines.extend(f"  - {c}" for c in changed)
    else:
        lines.append("No changes needed this week.")

    entry = {"ts": now.isoformat(), "closed": n, "win_rate": round(wr, 3),
             "changes": changed,
             "min_probability": cfg.get("quant", {}).get("min_probability"),
             "use_fib": cfg.get("use_fib", True)}
    _log(entry)
    return "\n".join(lines)


def maybe_run(cfg: dict, state: dict) -> str | None:
    """Fire run_weekly_tune once per ISO week, Sunday 21:00-22:00 UTC window."""
    now = datetime.now(timezone.utc)
    if now.weekday() != 6 or now.hour != 21:
        return None
    week = now.isocalendar()[1]
    if state.get("last_tune_week") == week:
        return None
    state["last_tune_week"] = week
    return run_weekly_tune(cfg)
