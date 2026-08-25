import json
import os
from datetime import datetime, timezone

import requests


def load_config(path: str = "config.json") -> dict:
    paths_to_try = [path, "config.example.json"]
    found = None
    for p in paths_to_try:
        full = os.path.join(os.path.dirname(os.path.abspath(__file__)), p)
        if os.path.exists(full):
            found = full
            break
        if os.path.exists(p):
            found = p
            break
    if found is None:
        raise FileNotFoundError(f"No config found (tried: {paths_to_try})")
    with open(found, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        cfg.setdefault("telegram", {})["enabled"] = True
        cfg["telegram"]["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN")
    if os.getenv("TELEGRAM_CHAT_ID"):
        cfg.setdefault("telegram", {})["chat_id"] = os.getenv("TELEGRAM_CHAT_ID")
    if os.getenv("WEBHOOK_HOST"):
        cfg.setdefault("webhook_server", {})["host"] = os.getenv("WEBHOOK_HOST")
    if os.getenv("WEBHOOK_PORT"):
        cfg.setdefault("webhook_server", {})["port"] = int(os.getenv("WEBHOOK_PORT"))
    if os.getenv("PORT"):
        cfg.setdefault("webhook_server", {})["port"] = int(os.getenv("PORT"))
    return cfg


def send_telegram(cfg: dict, message: str) -> bool:
    tg = cfg.get("telegram", {})
    if not tg.get("enabled") or not tg.get("bot_token") or not tg.get("chat_id"):
        return False
    url = f"https://api.telegram.org/bot{tg['bot_token']}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": tg["chat_id"], "text": message}, timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def send_telegram_photo(cfg: dict, image_bytes: bytes, caption: str = "") -> bool:
    tg = cfg.get("telegram", {})
    if not tg.get("enabled") or not tg.get("bot_token") or not tg.get("chat_id"):
        return False
    url = f"https://api.telegram.org/bot{tg['bot_token']}/sendPhoto"
    try:
        resp = requests.post(
            url,
            data={"chat_id": tg["chat_id"], "caption": caption},
            files={"photo": ("chart.png", image_bytes, "image/png")},
            timeout=30,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def send_webhook(cfg: dict, pair: str, result: dict) -> bool:
    hook = cfg.get("webhook", {})
    url = hook.get("url", "")
    if not url:
        return False
    payload = {
        "bot": "mjbot",
        "symbol": pair,
        "signal": result["signal"],
        "price": result["price"],
        "entry": result["entry"],
        "stop_loss": result["stop_loss"],
        "take_profit": result["take_profit"],
        "score": result["score"],
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code < 400
    except requests.RequestException:
        return False


def format_alert(pair: str, result: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    details = result.get("details", {})
    signal = result.get("signal", "NONE")

    emoji = "[BUY]" if "BUY" in signal else "[SELL]" if "SELL" in signal else "[-]"
    strength = "STRONG " if "STRONG" in signal else ""
    direction = "BUY" if "BUY" in signal else "SELL"

    lines = [
        f"{emoji} {pair} | {strength}{direction} SIGNAL",
        f"Time: {ts}",
        f"Entry: {result['entry']}",
        f"Stop Loss: {result['stop_loss']}",
        f"Take Profit: {result['take_profit']}",
    ]

    if details.get("htf_direction"):
        lines.append(f"Trend: {details['htf_direction']} on 4H/1H/45M/15M, entry on 5M")

    risk = abs(result["entry"] - result["stop_loss"])
    if risk > 0:
        rr = abs(result["take_profit"] - result["entry"]) / risk
        lines.append(f"Risk: {risk:.2f} pts | R:R 1:{rr:.1f}")

    if details.get("structure"):
        lines.append(f"Structure: {details['structure']}")
    if details.get("structure_event"):
        lines.append(f"Event: {details['structure_event']}")
    if details.get("pd_zone"):
        lines.append(f"Zone: {details['pd_zone']}")
    if details.get("zone"):
        lines.append(f"S/D: {details['zone']}")
    if details.get("sweep"):
        lines.append(f"Sweep: {details['sweep']}")
    if details.get("order_block"):
        lines.append(f"OB: {details['order_block']}")
    if details.get("fvg"):
        lines.append(f"FVG: {details['fvg']}")
    if details.get("fib"):
        lines.append(f"Fib: {details['fib']}")
    if details.get("momentum"):
        lines.append(f"Momentum: {details['momentum']}")
    if details.get("volume"):
        lines.append(f"Volume: {details['volume']}")
    if details.get("nearest_bsl"):
        lines.append(f"BSL Target: {details['nearest_bsl']}")
    if details.get("nearest_ssl"):
        lines.append(f"SSL Target: {details['nearest_ssl']}")
    if details.get("bias"):
        lines.append(f"Bias: {details['bias']}")
    if details.get("p_win") is not None:
        lines.append(f"Win Prob: {details['p_win'] * 100:.0f}%")
    if details.get("risk_suggested"):
        lines.append(f"Suggested Risk: {details['risk_suggested']}")
    if details.get("mtf_high_confluence"):
        lines.append("MTF Confluence: HIGH - all timeframes agree")
    if details.get("mtf_summary"):
        lines.append(f"MTF: {details['mtf_summary']}")
    for ctx in details.get("context_lines", []):
        lines.append(f"Context: {ctx}")

    if result.get("live_source"):
        lines.append(f"Source: {result['live_source']}")

    buy_s = details.get("buy_score", 0)
    sell_s = details.get("sell_score", 0)
    lines.append(f"Score: BUY {buy_s} | SELL {sell_s} | Confluence: {details.get('confluence', 0)} factors")

    return "\n".join(line for line in lines if line)
