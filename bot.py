import sys
import time

from alerts import format_alert, load_config, send_telegram, send_webhook
from data_fetcher import fetch_forex_data
from strategy import analyze


def main() -> None:
    cfg = load_config()
    print(f"Starting Forex signal bot. Pairs: {', '.join(cfg['pairs'])}")
    print(f"Timeframe: {cfg['timeframe']} | Check every {cfg['interval_seconds']}s")
    print("Signals: BUY / SELL / STRONG_BUY / STRONG_SELL\n")

    last_signals: dict[str, str] = {}

    while True:
        for pair in cfg["pairs"]:
            try:
                df = fetch_forex_data(pair, cfg["timeframe"])
                result = analyze(df, cfg)
                signal = result["signal"]

                if signal in ("STRONG_BUY", "BUY", "STRONG_SELL", "SELL"):
                    if last_signals.get(pair) != signal:
                        message = format_alert(pair, result)
                        print(message)
                        send_telegram(cfg, message)
                        send_webhook(cfg, pair, result)
                        last_signals[pair] = signal
                else:
                    last_signals[pair] = "NONE"

            except Exception as e:
                print(f"[{pair}] Error: {e}", file=sys.stderr)

        time.sleep(cfg["interval_seconds"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped.")
