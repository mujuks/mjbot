import json
import logging
import os
import sys
import time
import threading
from datetime import datetime, timezone

from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alerts import load_config, format_alert, send_telegram, send_telegram_photo, send_webhook
from chart import generate_signal_chart
from data_fetcher import fetch_forex_data
from live_price import fetch_gold_spot_price, fetch_xauusd_from_tradingview
from strategy import compute_bias

app = Flask(__name__)
log = logging.getLogger("webhook")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

PAIRS_MAP = {
    "XAUUSD": "GC=F",
    "GOLD": "GC=F",
    "GC=F": "GC=F",
}

_realtime_price_cache = {"price": None, "timestamp": None, "lock": threading.Lock()}


def _load_cfg():
    return load_config()


def _map_symbol(symbol: str) -> str:
    s = symbol.upper()
    if s in PAIRS_MAP:
        return PAIRS_MAP[s]
    if s.endswith("XAUUSD") or s.endswith(":GOLD"):
        return "GC=F"
    return s


def _get_realtime_price():
    """Get real-time XAUUSD price from cache or fetch."""
    with _realtime_price_cache["lock"]:
        if _realtime_price_cache["price"] and _realtime_price_cache["timestamp"]:
            age = (datetime.now(timezone.utc) - _realtime_price_cache["timestamp"]).total_seconds()
            if age < 10:
                return _realtime_price_cache["price"]
    
    try:
        data = fetch_xauusd_from_tradingview()
        if data and data["mid"] > 1000:
            with _realtime_price_cache["lock"]:
                _realtime_price_cache["price"] = data["mid"]
                _realtime_price_cache["timestamp"] = datetime.now(timezone.utc)
            return data["mid"]
    except Exception as e:
        log.warning("Real-time price fetch failed: %s", e)
    
    return None


def _validate_signal_with_realtime(sig: dict) -> dict:
    """Validate and enhance signal with real-time price data."""
    realtime_price = _get_realtime_price()
    if realtime_price:
        sig["realtime_price"] = realtime_price
        sig["price_source"] = "realtime"
        
        if sig["entry"] > 0:
            sig["entry_distance_pts"] = round(realtime_price - sig["entry"], 2)
            sig["entry_distance_pct"] = round((realtime_price - sig["entry"]) / sig["entry"] * 100, 3)
        
        if sig["stop_loss"] > 0:
            sig["risk_pts"] = round(abs(realtime_price - sig["stop_loss"]), 2)
        
        if sig["take_profit"] > 0:
            sig["reward_pts"] = round(abs(sig["take_profit"] - realtime_price), 2)
            
        if sig["risk_pts"] > 0:
            sig["rr_ratio"] = round(sig["reward_pts"] / sig["risk_pts"], 2)
    
    return sig


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
    pair = _map_symbol(symbol)

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
    
    if sig.get("realtime_price"):
        message += f"\n  Real-time Price: {sig['realtime_price']:.2f}"
        if sig.get("entry_distance_pts") is not None:
            message += f"\n  Entry Distance: {sig['entry_distance_pts']:+.2f} pts ({sig['entry_distance_pct']:+.3f}%)"
        if sig.get("rr_ratio"):
            message += f"\n  R:R Ratio: 1:{sig['rr_ratio']:.1f}"

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
        
        if cfg.get("realtime_price_check", False):
            sig = _validate_signal_with_realtime(sig)
        
        _send_to_telegram(sig, cfg)
        return jsonify({"status": "ok", "signal": sig["signal"], "pair": sig["pair"]}), 200

    except Exception as e:
        log.error("Webhook error: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "time": datetime.now(timezone.utc).isoformat()}), 200


@app.route("/price", methods=["GET"])
def price():
    """Real-time XAUUSD price endpoint."""
    try:
        data = fetch_xauusd_from_tradingview()
        if data:
            return jsonify({
                "status": "ok",
                "price": data["mid"],
                "bid": data["bid"],
                "ask": data["ask"],
                "spread": data["spread"],
                "exchange": data["exchange"],
                "feeds_count": data["feeds_count"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }), 200
        return jsonify({"status": "error", "message": "Price data unavailable"}), 503
    except Exception as e:
        log.error("Price endpoint error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/signal", methods=["POST"])
def signal():
    """Direct signal endpoint with real-time validation."""
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

        cfg = _load_cfg()
        sig = _format_tv_alert(tv)
        
        if cfg.get("realtime_price_check", False):
            sig = _validate_signal_with_realtime(sig)
        
        _send_to_telegram(sig, cfg)
        
        return jsonify({
            "status": "ok",
            "signal": sig["signal"],
            "pair": sig["pair"],
            "entry": sig["entry"],
            "stop_loss": sig["stop_loss"],
            "take_profit": sig["take_profit"],
            "realtime_price": sig.get("realtime_price"),
            "rr_ratio": sig.get("rr_ratio"),
        }), 200

    except Exception as e:
        log.error("Signal endpoint error: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


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
