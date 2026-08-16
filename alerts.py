import json
from datetime import datetime

import requests


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code < 400
    except requests.RequestException:
        return False


def format_alert(pair: str, result: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"[{ts}] {pair} -> {result['signal']}",
        f"  Entry: {result['entry']}",
        f"  Stop Loss: {result['stop_loss']}",
        f"  Take Profit: {result['take_profit']}",
    ]
    for key, value in result.get("details", {}).items():
        lines.append(f"  {key}: {value}")
    lines.append(f"  Score: {result['score']}")
    return "\n".join(lines)
