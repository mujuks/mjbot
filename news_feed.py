"""Economic news awareness via the free ForexFactory weekly calendar feed.

Fetches https://nfs.faireconomy.media/ff_calendar_thisweek.json (no API key,
cached 30 min) and exposes:
  - upcoming high/medium impact events filtered by currency
  - a blackout check so signals are suppressed around red-folder releases
  - Telegram-friendly daily brief / next-event formatting

Gold reacts overwhelmingly to USD data, so USD is the default filter.
"""
import logging
import time
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger("news")

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

_cache: dict = {"events": [], "epoch": 0.0, "ok": False}
_TTL_SECS = 1800


def _fetch_events() -> list[dict]:
    now = time.time()
    if _cache["ok"] and now - _cache["epoch"] < _TTL_SECS:
        return _cache["events"]
    try:
        r = requests.get(FEED_URL, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        })
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        log.warning("news feed fetch failed (%s); using stale cache if any", e)
        _cache["epoch"] = now
        return _cache["events"]

    events = []
    for item in raw if isinstance(raw, list) else []:
        try:
            dt = datetime.fromisoformat(item["date"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            events.append({
                "title": str(item.get("title", "")).strip(),
                "currency": str(item.get("country", "")).strip(),
                "impact": str(item.get("impact", "")).strip(),
                "dt": dt.astimezone(timezone.utc),
                "forecast": item.get("forecast", ""),
                "previous": item.get("previous", ""),
            })
        except Exception:
            continue
    events.sort(key=lambda e: e["dt"])

    if events:
        _cache.update(events=events, epoch=now, ok=True)
    else:
        _cache["epoch"] = now
    return _cache["events"]


def upcoming(cfg: dict, hours_ahead: float = 36,
             include_past_mins: float = 0.0) -> list[dict]:
    """Relevant events within [-include_past_mins, +hours_ahead] of now."""
    ncfg = cfg.get("news", {})
    currencies = set(ncfg.get("currencies", ["USD"]))
    impacts = set(ncfg.get("impacts", ["High", "Medium"]))
    now = datetime.now(timezone.utc)
    lo = now - timedelta(minutes=include_past_mins)
    hi = now + timedelta(hours=hours_ahead)
    return [
        e for e in _fetch_events()
        if e["currency"] in currencies and e["impact"] in impacts
        and lo <= e["dt"] <= hi
    ]


def blackout_check(cfg: dict, now: datetime | None = None) -> tuple[bool, dict | None]:
    """(True, event) when inside a no-trade window around an impact event."""
    ncfg = cfg.get("news", {})
    if not ncfg.get("enabled", True):
        return False, None
    currencies = set(ncfg.get("currencies", ["USD"]))
    impacts = set(ncfg.get("impacts", ["High"]))
    before = int(ncfg.get("minutes_before", 30))
    after = int(ncfg.get("minutes_after", 15))
    now = now or datetime.now(timezone.utc)

    for e in _fetch_events():
        if e["currency"] not in currencies or e["impact"] not in impacts:
            continue
        start = e["dt"] - timedelta(minutes=before)
        end = e["dt"] + timedelta(minutes=after)
        if start <= now <= end:
            return True, e
    return False, None


def next_event_line(cfg: dict) -> str | None:
    """Compact 'next release' hint for status/hourly messages."""
    evts = upcoming(cfg, hours_ahead=48, include_past_mins=10)
    if not evts:
        return None
    e = evts[0]
    mins = int((e["dt"] - datetime.now(timezone.utc)).total_seconds() // 60)
    when = f"in {mins}m" if mins >= 0 else f"{-mins}m ago"
    return f"{e['title']} ({e['currency']} {e['impact']}) {e['dt']:%a %H:%M} UTC [{when}]"


def daily_brief(cfg: dict) -> str | None:
    """Telegram schedule of today's relevant releases; None when nothing on."""
    ncfg = cfg.get("news", {})
    currencies = set(ncfg.get("currencies", ["USD"]))
    impacts = set(ncfg.get("impacts", ["High", "Medium"]))
    now = datetime.now(timezone.utc)
    todays = [
        e for e in _fetch_events()
        if e["currency"] in currencies and e["impact"] in impacts
        and e["dt"].date() == now.date()
    ]
    if not todays:
        return None

    lines = [f"News today ({now:%a %d %b}) - gold-relevant releases:"]
    for e in todays:
        fc = f" fc {e['forecast']}" if e["forecast"] else ""
        prev = f" prev {e['previous']}" if e["previous"] else ""
        flag = "[!]" if e["impact"] == "High" else "[-]"
        lines.append(f"  {flag} {e['dt']:%H:%M} UTC - {e['title']}{fc}{prev}")
    lines.append("Signals pause +/-30min around High-impact releases.")
    return "\n".join(lines)
