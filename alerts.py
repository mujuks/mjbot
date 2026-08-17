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
    lines = [
        f"[{ts}] {pair} -> {result['signal']}",
        f"  Entry: {result['entry']}",
        f"  Stop Loss: {result['stop_loss']}",
        f"  Take Profit: {result['take_profit']}",
    ]
    if result.get("live_source"):
        lines.append(f"  Price source: {result['live_source']}")
    for key, value in result.get("details", {}).items():
        lines.append(f"  {key}: {value}")
    lines.append(f"  Score: {result['score']}")
    return "\n".join(lines)
