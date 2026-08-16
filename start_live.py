import threading
import time
import sys
import os
import subprocess

sys.path.insert(0, r"C:\Users\Kiongozi Legit\Desktop\mjbot")

from webhook_server import run_server
from alerts import load_config, send_telegram, send_telegram_photo
from data_fetcher import fetch_forex_data
from strategy import analyze, compute_bias
from chart import generate_signal_chart, format_alert

PORT = 8080

def start_ssh_tunnel():
    print("Starting SSH tunnel via serveo.net...")
    proc = subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-R", f"80:localhost:{PORT}", "serveo.net"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    url = None
    start = time.time()
    while time.time() - start < 20:
        line = proc.stdout.readline()
        if not line:
            break
        print("SSH:", line.strip())
        if "Forwarding" in line:
            url = line.strip().split(" ")[-1].replace("http://", "https://")
            break
    if not url:
        print("SSH tunnel failed - try manual setup")
    return url

def start_localtunnel():
    print("Trying localtunnel via npx...")
    try:
        proc = subprocess.Popen(
            ["npx", "localtunnel", "--port", str(PORT)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        url = None
        start = time.time()
        while time.time() - start < 30:
            line = proc.stdout.readline()
            if not line:
                break
            print("LT:", line.strip())
            if "your url" in line.lower() or "https://" in line:
                url = line.strip().split()[-1]
                break
        return url
    except FileNotFoundError:
        print("npx not found")
        return None

def send_startup_signal(webhook_url):
    cfg = load_config()
    df = fetch_forex_data("GC=F", "15m", 30)
    bdf = fetch_forex_data("GC=F", "1h", 30)
    bias = compute_bias(bdf, cfg)
    result = analyze(df, cfg, bias=bias)
    result["bias"] = bias
    
    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    msg = (
        f"[{ts}] MJBot LIVE\n\n"
        f"TradingView Webhook URL:\n"
        f"  {webhook_url}/webhook\n\n"
        f"SETUP STEPS:\n"
        f"1. TradingView -> GC=F 15m chart\n"
        f"2. Alerts -> Create Alert\n"
        f"3. Webhook URL: {webhook_url}/webhook\n"
        f"4. Message: use the JSON from Pine Script\n"
        f"5. Click Create\n\n"
        f"NOTE: Gold futures are CLOSED (Sunday).\n"
        f"Signals flow automatically when market opens."
    )
    send_telegram(cfg, msg)
    chart = generate_signal_chart(df, cfg, result, "GC=F")
    if chart:
        send_telegram_photo(cfg, chart, msg)
    print("Startup message sent to Telegram")

if __name__ == "__main__":
    print("Starting MJBot with TradingView webhook...\n")
    
    wh_thread = threading.Thread(target=run_server, args=("127.0.0.1", PORT), daemon=True)
    wh_thread.start()
    time.sleep(1)
    print(f"Webhook server running on port {PORT}")
    
    url = start_ssh_tunnel()
    if not url:
        url = start_localtunnel()
    
    if url:
        print(f"\nWebhook URL: {url}/webhook\n")
        send_startup_signal(url)
    else:
        print("Could not create tunnel. Bot running locally on port 8080")
        cfg = load_config()
        send_telegram(cfg, "MJBot webhook server running on localhost:8080. ngrok needed for TradingView.")
    
    print("\nBot is LIVE. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopped.")
