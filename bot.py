import os
import sys
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from alerts import format_alert, load_config, send_telegram, send_telegram_photo, send_webhook
from chart import generate_signal_chart
from data_fetcher import fetch_forex_data
from live_price import fetch_gold_spot_price, validate_data_freshness, calibrate_to_spot
from strategy import analyze, compute_bias, trading_allowed


def analyze_pair(pair: str, cfg: dict) -> tuple[dict, pd.DataFrame]:
    df = fetch_forex_data(pair, cfg["timeframe"], cfg.get("lookback_days", 30))
    fresh, age_msg = validate_data_freshness(df, cfg.get("max_data_age_minutes", 60))
    if not fresh:
        print(f"  [{pair}] WARNING: {age_msg}")

    live_price, ts, source = fetch_gold_spot_price()

    bias = 0
    if cfg.get("bias_enabled", True) and cfg.get("bias_timeframe"):
        bias_df = fetch_forex_data(pair, cfg["bias_timeframe"], cfg.get("lookback_days", 30))
        bias = compute_bias(bias_df, cfg)
    result = analyze(df, cfg, bias=bias, live_price=live_price)
    result["bias"] = bias
    if live_price:
        result["live_price"] = live_price
        result["live_source"] = source
    return result, df


def format_digest(pair: str, result: dict, now_utc: datetime, cfg: dict) -> str:
    session = "open" if trading_allowed(now_utc, cfg) else "closed"
    bias_label = {1: "bullish", -1: "bearish", 0: "flat"}.get(result.get("bias", 0), "flat")
    ts = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    signal = result.get("signal", "NONE")
    lines = [
        f"[{ts}] Hourly report {pair} | Market {session} | Bias {bias_label}",
        f"  Signal: {signal}",
    ]
    if signal in ("STRONG_BUY", "BUY", "STRONG_SELL", "SELL"):
        lines.extend([
            f"  Entry: {result['entry']}",
            f"  Stop Loss: {result['stop_loss']}",
            f"  Take Profit: {result['take_profit']}",
            f"  Score: {result['score']}",
        ])
    else:
        lines.append("  No active setup this hour.")
    return "\n".join(lines)


def _polling_loop(cfg):
    print(f"Starting Forex signal bot. Pairs: {', '.join(cfg['pairs'])}")
    print(f"Entry timeframe: {cfg['timeframe']} | Bias timeframe: {cfg.get('bias_timeframe', 'none')}")
    print(f"Check every {cfg['interval_seconds']}s | Hourly digest: {cfg.get('hourly_digest', False)}")
    print("Signals: BUY / SELL / STRONG_BUY / STRONG_SELL\n")

    last_signals: dict[str, str] = {}
    last_results: dict[str, dict] = {}
    last_dfs: dict[str, pd.DataFrame] = {}
    last_digest_hour: int | None = None

    while True:
        now_utc = datetime.now(timezone.utc)
        active = trading_allowed(now_utc, cfg)

        for pair in cfg["pairs"]:
            try:
                if not active:
                    if last_signals.get(pair) != "NONE":
                        print(f"[{pair}] Market closed - pausing signals")
                        last_signals[pair] = "NONE"
                    continue

                result, df = analyze_pair(pair, cfg)
                last_results[pair] = result
                last_dfs[pair] = df
                signal = result["signal"]

                if signal in ("STRONG_BUY", "BUY", "STRONG_SELL", "SELL"):
                    if last_signals.get(pair) != signal:
                        message = format_alert(pair, result)
                        print(message)
                        send_telegram(cfg, message)
                        send_webhook(cfg, pair, result)
                        chart = generate_signal_chart(df, cfg, result, pair)
                        if chart:
                            send_telegram_photo(cfg, chart, message)
                        last_signals[pair] = signal
                else:
                    last_signals[pair] = "NONE"

            except Exception as e:
                print(f"[{pair}] Error: {e}", file=sys.stderr)

        if cfg.get("hourly_digest", False) and last_digest_hour != now_utc.hour:
            last_digest_hour = now_utc.hour
            for pair in cfg["pairs"]:
                result = last_results.get(pair) if active else None
                if result is None:
                    result = {"signal": "NONE", "bias": 0}
                message = format_digest(pair, result, now_utc, cfg)
                print(message)
                send_telegram(cfg, message)
                df = last_dfs.get(pair)
                if df is not None and result.get("signal", "NONE") != "NONE":
                    chart = generate_signal_chart(df, cfg, result, pair)
                    if chart:
                        send_telegram_photo(cfg, chart, message)

        time.sleep(cfg["interval_seconds"])


def main() -> None:
    cfg = load_config()

    port = int(os.environ.get("PORT", cfg.get("webhook_server", {}).get("port", 8080)))
    host = cfg.get("webhook_server", {}).get("host", "0.0.0.0")
    use_webhook_server = cfg.get("webhook_server", {}).get("enabled", True)

    if use_webhook_server:
        poll_thread = threading.Thread(target=_polling_loop, args=(cfg,), daemon=True)
        poll_thread.start()
        print(f"Polling loop started in background")
        print(f"Webhook server starting on {host}:{port}")
        print(f"  TradingView alert URL: http://YOUR_WEBHOOK_URL/webhook")
        print(f"  Health check: http://YOUR_WEBHOOK_URL/health")
        from webhook_server import run_server
        run_server(host, port)
    else:
        _polling_loop(cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped.")
