"""Expose the local webhook server (port 8080) at a public HTTPS URL so
TradingView alerts can reach MJBot.

Tries several free tunnel providers in order until one works:
    1. localhost.run   (ssh, no account)
    2. serveo.net      (ssh, no account)
    3. pinggy.io       (ssh, no account)
    4. localtunnel     (npx)

Prints the public URL and sends it to Telegram. Keep this process running.
"""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PORT = int(os.environ.get("PORT", "8080"))
URL_RE = re.compile(r"https://[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}[a-zA-Z0-9/-]*")
_BAD_URL_HINTS = ("localhost.run/docs", "admin.localhost.run", "pinggy.io", "serveo.net")
_GOOD_HOST_HINTS = ("trycloudflare.com", "lhr.life", "serveo.net", "pinggy.link", "loca.lt")
CLOUDFLARED = os.path.join(os.environ.get("TEMP", r"C:\Temp"), "opencode", "cloudflared.exe")


def _pick_url(line: str) -> str | None:
    for m in URL_RE.finditer(line):
        u = m.group(0).rstrip(".")
        if any(h in u for h in _BAD_URL_HINTS):
            continue
        if any(h in u for h in _GOOD_HOST_HINTS):
            return u
    return None


def _doh_resolve(host):
    """Resolve A records via Cloudflare DoH - bypasses slow local DNS."""
    try:
        import requests
        r = requests.get(
            "https://1.1.1.1/dns-query",
            params={"name": host, "type": "A"},
            headers={"accept": "application/dns-json"}, timeout=10,
        )
        return [a["data"] for a in (r.json().get("Answer") or []) if a.get("type") == 1]
    except Exception:
        return []


def _verify(url: str) -> bool:
    from urllib.parse import urlparse
    host = urlparse(url).hostname
    import requests
    for attempt in range(2):
        try:
            r = requests.get(f"{url}/health", timeout=12)
            print(f"  verify {url}/health -> {r.status_code}", flush=True)
            return r.status_code == 200
        except Exception as e:
            print(f"  verify failed ({attempt + 1}): {type(e).__name__}", flush=True)
            time.sleep(4)
    # New subdomains often lag in local ISP DNS. Resolve via DoH and pin the IP.
    ips = _doh_resolve(host)
    for ip in ips[:2]:
        try:
            out = subprocess.run(
                ["curl.exe", "-sS", "--max-time", "15",
                 "--resolve", f"{host}:443:{ip}", f"{url}/health"],
                capture_output=True, text=True, timeout=25,
            )
            body = (out.stdout or "").strip()
            print(f"  verify(pinned {ip}) -> {body[:90]}", flush=True)
            if '"status"' in body and "running" in body:
                return True
        except Exception as e:
            print(f"  verify pinned failed: {type(e).__name__}", flush=True)
    return False


def _try_ssh(cmd, timeout=45):
    """cmd is a complete command line (list); used for both ssh and cloudflared."""
    prefix = os.path.basename(cmd[0]).lower()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except FileNotFoundError:
        return None, None
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            break
        stripped = line.strip()
        if len(stripped) > 4:  # skip control-sequence noise
            print(f"  {prefix}> {stripped[:140]}", flush=True)
        url = _pick_url(line)
        if url and _verify(url):
            return url, proc
    return None, proc


def get_tunnel():
    attempts = []
    if os.path.isfile(CLOUDFLARED):
        attempts.append((
            "trycloudflare",
            [CLOUDFLARED, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{PORT}"],
        ))
    attempts += [
        ("localhost.run", ["ssh", "-o", "StrictHostKeyChecking=no",
                           "-o", "ServerAliveInterval=30",
                           "-R", f"80:localhost:{PORT}", "nokey@localhost.run"]),
    ]
    procs = []
    url = None
    try:
        for name, args in attempts:
            print(f"Trying {name}...", flush=True)
            url, proc = _try_ssh(args)
            if proc:
                procs.append(proc)
            if url:
                return url, procs

        print("Trying localtunnel (npx)...", flush=True)
        try:
            proc = subprocess.Popen(
                ["npx", "-y", "localtunnel", "--port", str(PORT)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            procs.append(proc)
            start = time.time()
            while time.time() - start < 40:
                line = proc.stdout.readline()
                if not line:
                    break
                print(f"  lt> {line.strip()[:140]}", flush=True)
                url = _pick_url(line)
                if url and _verify(url):
                    return url, procs
        except FileNotFoundError:
            pass
    except KeyboardInterrupt:
        pass
    return None, procs


def main():
    from alerts import load_config, send_telegram

    cfg = load_config()
    url, procs = get_tunnel()
    if not url:
        msg = "MJBot: could not open a public tunnel for TradingView webhooks."
        print(msg)
        send_telegram(cfg, msg)
        sys.exit(1)

    hook = f"{url}/webhook"
    print(f"\nPUBLIC WEBHOOK URL:\n  {hook}\n")
    msg = (
        "\U0001f517 TradingView is now connected to MJBot\n\n"
        f"Webhook URL for your alert:\n{hook}\n\n"
        "TradingView setup:\n"
        "1. Open OANDA:XAUUSD 5m chart\n"
        "2. Add the gold_signals.pine v2 script\n"
        "3. Create Alert -> Condition: GoldSignals -> 'Any alert() function call'\n"
        f"4. Webhook URL: {hook}\n"
        "5. Notifications: tick 'Webhook URL'\n"
        "6. Create - TV signals now reach your bot + Telegram"
    )
    send_telegram(cfg, msg)
    print("Instructions sent to Telegram. Keeping tunnel alive...")
    try:
        while True:
            time.sleep(30)
            for p in list(procs):
                if p.poll() is not None:
                    procs.remove(p)
            alive = any(p.poll() is None for p in procs)
            if not alive:
                print("Tunnel died - reconnecting...", flush=True)
                url2, procs2 = get_tunnel()
                procs.extend(procs2)
                if url2 and url2 != url:
                    url = url2
                    send_telegram(cfg, f"\U0001f517 Tunnel changed. New webhook URL:\n{url}/webhook")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
