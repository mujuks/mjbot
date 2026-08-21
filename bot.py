import concurrent.futures
import os
import sys
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from alerts import format_alert, load_config, send_telegram, send_telegram_photo, send_webhook
from chart import generate_signal_chart
from data_fetcher import fetch_forex_data
from data_store import append_candles
from live_price import fetch_gold_spot_price, validate_data_freshness, calibrate_to_spot
from quant import assess_signal, suggested_risk_pct
from strategy import analyze, compute_bias, trading_allowed


def analyze_pair(pair: str, cfg: dict) -> tuple[dict, pd.DataFrame]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        df_future = executor.submit(fetch_forex_data, pair, cfg["timeframe"], cfg.get("lookback_days", 30))
        live_future = executor.submit(fetch_gold_spot_price)
        bias_future = None
        if cfg.get("bias_enabled", True) and cfg.get("bias_timeframe"):
            bias_future = executor.submit(fetch_forex_data, pair, cfg["bias_timeframe"], cfg.get("lookback_days", 30))

        df = df_future.result()
        live_price, ts, source = live_future.result()
        bias = 0
        if bias_future:
            bias_df = bias_future.result()
            bias = compute_bias(bias_df, cfg)

    fresh, age_msg = validate_data_freshness(df, cfg.get("max_data_age_minutes", 20))
    if not fresh:
        print(f"  [{pair}] WARNING: {age_msg}")

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
    details = result.get("details", {})
    lines = [
        f"[{ts}] Hourly report {pair} | Market {session}",
        f"  Signal: {signal} | Bias: {bias_label}",
    ]
    if details.get("structure"):
        lines.append(f"  Structure: {details['structure']}")
    if details.get("pd_zone"):
        lines.append(f"  Price Zone: {details['pd_zone']}")
    if signal in ("STRONG_BUY", "BUY", "STRONG_SELL", "SELL"):
        lines.extend([
            f"  Entry: {result['entry']}",
            f"  Stop Loss: {result['stop_loss']}",
            f"  Take Profit: {result['take_profit']}",
            f"  Score: {result['score']} | Buy: {details.get('buy_score', 0)} | Sell: {details.get('sell_score', 0)}",
        ])
    else:
        lines.append(f"  Buy: {details.get('buy_score', 0)} | Sell: {details.get('sell_score', 0)}")
    return "\n".join(lines)


def format_no_signal(pair: str, result: dict, now_utc: datetime, cfg: dict) -> str:
    bias_label = {1: "bullish", -1: "bearish", 0: "flat"}.get(result.get("bias", 0), "flat")
    ts = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    details = result.get("details", {})
    buy_score = details.get("buy_score", 0)
    sell_score = details.get("sell_score", 0)
    price = result.get("price", 0)
    lines = [
        f"[{ts}] {pair} | No signal yet",
        f"  Price: {price:.2f}",
        f"  Bias: {bias_label}",
        f"  Buy score: {buy_score} | Sell score: {sell_score}",
    ]
    if buy_score > sell_score and buy_score >= 2:
        lines.append(f"  Approaching BUY setup ({buy_score}/{cfg.get('signal_threshold', 3)})")
    elif sell_score > buy_score and sell_score >= 2:
        lines.append(f"  Approaching SELL setup ({sell_score}/{cfg.get('signal_threshold', 3)})")
    else:
        lines.append("  Market quiet - waiting for setup.")
    return "\n".join(lines)


def format_get_ready(pair: str, result: dict, direction: str, now_utc: datetime) -> str:
    ts = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    details = result.get("details", {})
    price = result.get("price", 0)
    bias_label = details.get("bias", "flat")

    emoji_dir = "BULLISH" if direction == "BUY" else "BEARISH"
    emoji = "🟢" if direction == "BUY" else "🔴"

    lines = [
        f"{emoji} {pair} | WATCH - {emoji_dir} SETUP FORMING",
        f"  Price: {price:.2f} | Bias: {bias_label}",
    ]
    if details.get("structure"):
        lines.append(f"  Structure: {details['structure']}")
    if details.get("structure_event"):
        lines.append(f"  Event: {details['structure_event']}")
    if details.get("pd_zone"):
        lines.append(f"  Price Zone: {details['pd_zone']}")
    if details.get("zone") and details["zone"] != "none":
        lines.append(f"  Zone: {details['zone']}")
    if details.get("order_block") and details["order_block"] != "none":
        lines.append(f"  OB: {details['order_block']}")
    if details.get("sweep"):
        lines.append(f"  Sweep: {details['sweep']}")
    buy_s = details.get("buy_score", 0)
    sell_s = details.get("sell_score", 0)
    lines.append(f"  Score: BUY {buy_s} | SELL {sell_s}")
    if details.get("p_win") is not None:
        lines.append(f"  Win Prob: {details['p_win'] * 100:.0f}%")
    if details.get("gated_from"):
        lines.append(f"  Quant gate demoted {details['gated_from']} (low confidence)")
    lines.append(f"  Waiting for {direction} confirmation...")
    return "\n".join(lines)


def format_status(pair: str, result: dict, now_utc: datetime, cfg: dict) -> str:
    bias_label = {1: "bullish", -1: "bearish", 0: "flat"}.get(result.get("bias", 0), "flat")
    ts = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    details = result.get("details", {})
    buy_score = details.get("buy_score", 0)
    sell_score = details.get("sell_score", 0)
    price = result.get("price", 0)
    threshold = cfg.get("signal_threshold", 2)
    lines = [
        f"[{ts}] {pair} | Status",
        f"  Price: {price:.2f} | Bias: {bias_label}",
    ]
    struct = details.get("structure", "")
    if struct:
        lines.append(f"  Structure: {struct}")
    pd_zone = details.get("pd_zone", "")
    if pd_zone:
        lines.append(f"  Price Zone: {pd_zone}")
    lines.append(f"  Buy: {buy_score}/{threshold} | Sell: {sell_score}/{threshold}")
    if buy_score >= threshold or sell_score >= threshold:
        lines.append("  Setup active - signal may fire soon")
    elif buy_score >= 2 or sell_score >= 2:
        lines.append("  Building momentum - watching closely")
    else:
        lines.append("  Market quiet")
    return "\n".join(lines)


def format_weekend_status(pair: str, now_utc: datetime, cfg: dict, live_price: float = None) -> str:
    ts = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    session = "weekend" if now_utc.weekday() >= 5 else "closed hours"
    price_str = f"{live_price:.2f}" if live_price else "unavailable"
    lines = [
        f"[{ts}] {pair} | {session.title()} - Market Closed",
        f"  Live price: {price_str}",
        f"  Gold futures (GC=F) are not trading right now.",
        f"  Auto-signals resume when market opens.",
    ]
    if live_price:
        lines.append(f"  Monitoring via TradingView XAU/USD spot.")
    return "\n".join(lines)


def _polling_loop(cfg):
    print(f"Starting SMC Gold Signal Bot. Pairs: {', '.join(cfg['pairs'])}")
    print(f"Entry: {cfg['timeframe']} | Bias: {cfg.get('bias_timeframe', 'none')}")
    print(f"Check every {cfg['interval_seconds']}s | Hourly digest: {cfg.get('hourly_digest', False)}")
    print("SMC: Structure | Supply/Demand | Liquidity Pools | Order Blocks | FVG | Premium/Discount")
    print("Signals: STRONG_BUY / BUY / STRONG_SELL / SELL / WATCH_BUY / WATCH_SELL\n")

    last_signals: dict[str, str] = {}
    last_results: dict[str, dict] = {}
    last_dfs: dict[str, pd.DataFrame] = {}
    last_candle_ts: dict[str, pd.Timestamp] = {}
    last_digest_hour: int | None = None
    last_status_minute: dict[str, int] = {}
    last_get_ready: dict[str, str] = {}
    ready_threshold = cfg.get("get_ready_threshold", 1)

    while True:
        now_utc = datetime.now(timezone.utc)
        active = trading_allowed(now_utc, cfg)
        current_minute = int(now_utc.timestamp() // 60)

        for pair in cfg["pairs"]:
            try:
                if not active:
                    if last_signals.get(pair) != "NONE":
                        print(f"[{pair}] Market closed - pausing signals")
                        last_signals[pair] = "NONE"
                    continue

                try:
                    result, df = analyze_pair(pair, cfg)
                except RuntimeError:
                    live_price, _, _ = fetch_gold_spot_price()
                    last_minute = last_status_minute.get(pair, -999)
                    if current_minute - last_minute >= cfg["interval_seconds"] // 60:
                        message = format_weekend_status(pair, now_utc, cfg, live_price)
                        print(message)
                        send_telegram(cfg, message)
                        last_status_minute[pair] = current_minute
                    continue

                current_candle_ts = df.index[-1]
                if last_candle_ts.get(pair) == current_candle_ts:
                    continue
                last_candle_ts[pair] = current_candle_ts

                last_results[pair] = result
                last_dfs[pair] = df
                try:
                    append_candles(pair, cfg["timeframe"], df)
                except Exception as se:
                    print(f"[{pair}] store error: {se}", file=sys.stderr)
                signal = result["signal"]
                details = result.get("details", {})
                buy_score = details.get("buy_score", 0)
                sell_score = details.get("sell_score", 0)
                threshold = cfg.get("signal_threshold", 3)

                qcfg = cfg.get("quant", {})
                if qcfg.get("enabled", True):
                    try:
                        assess = assess_signal(pair, df, result, cfg)
                    except Exception as qe:
                        print(f"[{pair}] quant error: {qe}", file=sys.stderr)
                        assess = None
                    if assess:
                        p_win = assess.get("p_win")
                        if p_win is not None and signal in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL"):
                            details["p_win"] = round(p_win, 3)
                            risk_pct = suggested_risk_pct(p_win, cfg)
                            if risk_pct is not None:
                                details["risk_suggested"] = f"{risk_pct}%"
                            min_p = float(qcfg.get("min_probability", 0.55))
                            if p_win < min_p:
                                details["gated_from"] = signal
                                signal = "WATCH_BUY" if "BUY" in signal else "WATCH_SELL"
                                result["signal"] = signal
                        elif p_win is None and assess.get("error"):
                            print(f"[{pair}] quant: {assess['error']}", file=sys.stderr)

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
                        last_get_ready.pop(pair, None)
                elif signal in ("WATCH_BUY", "WATCH_SELL"):
                    watch_key = f"watch_{pair}"
                    if last_signals.get(pair) != signal:
                        direction = "BUY" if signal == "WATCH_BUY" else "SELL"
                        message = format_get_ready(pair, result, direction, now_utc)
                        print(message)
                        send_telegram(cfg, message)
                        last_signals[pair] = signal
                else:
                    last_signals[pair] = "NONE"

                    dominant = "BUY" if buy_score > sell_score else "SELL" if sell_score > buy_score else None
                    dominant_score = max(buy_score, sell_score) if dominant else 0

                    if dominant and dominant_score >= ready_threshold and dominant_score < threshold:
                        if last_get_ready.get(pair) != dominant:
                            message = format_get_ready(pair, result, dominant, now_utc)
                            print(message)
                            send_telegram(cfg, message)
                            last_get_ready[pair] = dominant
                            last_status_minute[pair] = current_minute
                    else:
                        last_get_ready.pop(pair, None)

                    last_minute = last_status_minute.get(pair, -999)
                    if current_minute - last_minute >= cfg["interval_seconds"] // 60:
                        message = format_status(pair, result, now_utc, cfg)
                        print(message)
                        send_telegram(cfg, message)
                        last_status_minute[pair] = current_minute

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
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped.")
