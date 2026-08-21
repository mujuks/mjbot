import json
import logging
import os
import sys
from datetime import datetime, timezone

from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alerts import load_config, format_alert, send_telegram, send_telegram_photo, send_webhook
from chart import generate_signal_chart
from data_fetcher import fetch_forex_data
from live_price import fetch_gold_spot_price
from strategy import compute_bias

app = Flask(__name__)
log = logging.getLogger("webhook")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

PAIRS_MAP = {
    "XAUUSD": "GC=F",
    "GOLD": "GC=F",
    "GC=F": "GC=F",
}


def _load_cfg():
    return load_config()


def _format_tv_alert(tv: dict) -> dict:
    raw_signal = tv.get("signal", tv.get("action", "NONE")).upper().replace(" ", "_")
    signal_map = {
        "LONG": "BUY", "BUY": "BUY",
        "SHORT": "SELL", "SELL": "SELL",
        "STRONG_LONG": "STRONG_BUY", "STRONG_BUY": "STRONG_BUY",
        "STRONG_SHORT": "STRONG_SELL", "STRONG_SELL": "STRONG_SELL",
        "CLOSE": "NONE", "EXIT": "NONE", "FLAT": "NONE",
    }
    signal = signal_map.get(raw_signal, raw_signal)
    symbol = tv.get("symbol", tv.get("ticker", "GC=F")).upper()
    pair = PAIRS_MAP.get(symbol, symbol)

    def _float(key, default=0.0):
        val = tv.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _int(key, default=0):
        try:
            return int(float(tv.get(key, default)))
        except (TypeError, ValueError):
            return default

    return {
        "signal": signal,
        "pair": pair,
        "entry": _float("entry", _float("close")),
        "stop_loss": _float("sl", _float("stop_loss")),
        "take_profit": _float("tp", _float("take_profit")),
        "price": _float("close", _float("entry")),
        "score": _int("score"),
        "bias": tv.get("bias", "unknown"),
        "timeframe": tv.get("timeframe", tv.get("interval", "15m")),
        "raw": tv,
    }


def _send_to_telegram(sig: dict, cfg: dict):
    pair = sig["pair"]
    signal = sig["signal"]

    if signal == "NONE":
        log.info("Signal NONE for %s, skipping", pair)
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    bias_label = sig.get("bias", "unknown")
    message = (
        f"[{ts}] {pair} -> {signal}\n"
        f"  Entry: {sig['entry']:.2f}\n"
        f"  Stop Loss: {sig['stop_loss']:.2f}\n"
        f"  Take Profit: {sig['take_profit']:.2f}\n"
        f"  Bias: {bias_label}\n"
        f"  Source: TradingView\n"
        f"  Score: {sig['score']}"
    )

    log.info("Sending %s %s to Telegram", pair, signal)
    send_telegram(cfg, message)

    try:
        bias = 0
        if cfg.get("bias_enabled", True) and cfg.get("bias_timeframe"):
            bias_df = fetch_forex_data(pair, cfg["bias_timeframe"], cfg.get("lookback_days", 30))
            bias = compute_bias(bias_df, cfg)
        df = fetch_forex_data(pair, cfg["timeframe"], cfg.get("lookback_days", 30))
        live_price, _, _ = fetch_gold_spot_price()
        result = {
            "signal": signal,
            "entry": sig["entry"],
            "stop_loss": sig["stop_loss"],
            "take_profit": sig["take_profit"],
            "price": sig["price"],
            "score": sig["score"],
            "details": {"bias": bias_label},
            "live_price": live_price,
        }
        chart = generate_signal_chart(df, cfg, result, pair)
        if chart:
            send_telegram_photo(cfg, chart, message)
            log.info("Chart sent for %s", pair)
    except Exception as e:
        log.error("Chart generation failed: %s", e)

    hook_result = send_webhook(cfg, pair, {
        "signal": signal,
        "price": sig["price"],
        "entry": sig["entry"],
        "stop_loss": sig["stop_loss"],
        "take_profit": sig["take_profit"],
        "score": sig["score"],
    })
    if hook_result:
        log.info("Outbound webhook delivered for %s", pair)


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        if request.is_json:
            tv = request.get_json(force=True)
        elif request.form:
            tv = dict(request.form)
        else:
            raw = request.get_data(as_text=True)
            try:
                tv = json.loads(raw)
            except json.JSONDecodeError:
                tv = {"raw": raw}

        log.info("Received: %s", json.dumps(tv, default=str)[:500])
        cfg = _load_cfg()
        sig = _format_tv_alert(tv)
        _send_to_telegram(sig, cfg)
        return jsonify({"status": "ok", "signal": sig["signal"], "pair": sig["pair"]}), 200

    except Exception as e:
        log.error("Webhook error: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "time": datetime.now(timezone.utc).isoformat()}), 200


def run_server(host: str = "127.0.0.1", port: int = 8080):
    log.info("Starting webhook server on %s:%d", host, port)
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    run_server(args.host, args.port)
