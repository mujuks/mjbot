import concurrent.futures
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from alerts import format_alert, load_config, send_telegram, send_telegram_photo, send_webhook
from chart import generate_signal_chart
from data_fetcher import fetch_forex_data
from data_store import append_candles, load_candles
import journal
from live_price import fetch_gold_spot_price, validate_data_freshness, calibrate_to_spot
import macro
import autotune
from mtf_engine import assess as mtf_assess
from news_feed import blackout_check, daily_brief, next_event_line
from quant import assess_signal, suggested_risk_pct
from strategy import _atr, analyze, compute_bias, trading_allowed

FIRM_SIGNALS = ("STRONG_BUY", "BUY", "STRONG_SELL", "SELL")

_realtime_price_cache = {"price": None, "timestamp": None, "lock": threading.Lock()}

_RANGE_RE = re.compile(r"(demand|supply|OB|FVG)\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)")


def _get_realtime_price():
    """Get real-time XAUUSD price from cache or fetch."""
    with _realtime_price_cache["lock"]:
        if _realtime_price_cache["price"] and _realtime_price_cache["timestamp"]:
            age = (datetime.now(timezone.utc) - _realtime_price_cache["timestamp"]).total_seconds()
            if age < 5:
                return _realtime_price_cache["price"]
    
    try:
        live_price, ts, source = fetch_gold_spot_price()
        if live_price and live_price > 1000:
            with _realtime_price_cache["lock"]:
                _realtime_price_cache["price"] = live_price
                _realtime_price_cache["timestamp"] = datetime.now(timezone.utc)
            return live_price
    except Exception as e:
        print(f"[realtime] Price fetch failed: {e}", file=sys.stderr)
    
    return None


def _parse_range(text):
    if not text:
        return None
    m = _RANGE_RE.search(str(text))
    if not m:
        return None
    try:
        return m.group(1), float(m.group(2)), float(m.group(3))
    except ValueError:
        return None


def build_watch_plan(result: dict, df, cfg: dict) -> list[str]:
    """Actionable 'where do I wait' levels for the current setup direction."""
    d = result.get("details", {})
    sig = result.get("signal", "")
    bull = "BUY" in sig
    price = float(result.get("price") or 0)
    if not price:
        return []
    thr = int(cfg.get("signal_threshold", 3))
    lines = []

    zones = []
    z = _parse_range(d.get("zone"))
    ob = _parse_range(d.get("order_block"))
    fv = _parse_range(d.get("fvg"))
    want = "demand" if bull else "supply"
    ob_tag = "bullish" if bull else "bearish"
    if z and z[0] == want:
        zones.append((f"{want.title()} zone", z[1], z[2]))
    if ob and ob[0] == "OB":
        raw_ob = str(d.get("order_block", ""))
        if ob_tag in raw_ob:
            zones.append((f"{ob_tag.title()} OB", ob[1], ob[2]))
    if fv and fv[0] == "FVG":
        raw_fv = str(d.get("fvg", ""))
        if ob_tag in raw_fv:
            zones.append((f"{ob_tag.title()} FVG", fv[1], fv[2]))

    for name, lo, hi in zones[:2]:
        mid = (lo + hi) / 2
        lines.append(f"{name} {lo:.2f}-{hi:.2f} (mid {mid - price:+.1f} pts from price)")

    liq = d.get("nearest_ssl" if bull else "nearest_bsl")
    if isinstance(liq, (int, float)) and liq:
        side = "SSL" if bull else "BSL"
        tag = "sweep of it into the zone = A+ entry" if bull else "sweep of it into the zone = A+ entry"
        lines.append(f"Liquidity {side} {float(liq):.2f} ({float(liq) - price:+.1f}) - {tag}")

    trig = d.get("choch_level") or d.get("bos_level")
    if trig:
        lines.append(f"Trigger: score >= {thr} or structure break {float(trig):.2f}")
    else:
        lines.append(f"Trigger: score >= {thr} with MTF agreement")

    if df is not None and len(df) > 25:
        ext = float(df["Low"].tail(20).min()) if bull else float(df["High"].tail(20).max())
        word = "below" if bull else "above"
        bar = "low" if bull else "high"
        lines.append(f"Invalidate {word} {ext:.2f} (20-bar {bar})")
    return lines


def _watch_level_line(details: dict, signal: str, price: float) -> str:
    """Build a one-line watch-level summary for WATCH_BUY / WATCH_SELL signals."""
    if not signal.startswith("WATCH") or not price:
        return ""
    bull = "BUY" in signal
    plan = details.get("watch_plan", [])
    if not plan:
        return ""
    first = plan[0]
    if "FVG" in first:
        tag = "FVG"
    elif "OB" in first:
        tag = "OB"
    elif "zone" in first.lower():
        tag = "zone"
    else:
        tag = ""
    paren = first.split("(")[1].rstrip(")") if "(" in first else ""
    lo_hi = first.split(tag)[0].strip().rstrip("-– ").strip() if tag else ""
    dist = ""
    if paren and "pts" in paren:
        dist = f" ({paren})"
    label = "watching BUY at" if bull else "watching SELL at"
    zone_text = f" | {label} {first.split('(')[0].strip()}" if tag else f" | {first}"
    return zone_text


def _format_watch_telegram(pair: str, result: dict, details: dict, signal: str) -> str:
    """Build a short Telegram message for a WATCH signal with specific levels."""
    bull = "BUY" in signal
    price = float(result.get("price") or 0)
    plan = details.get("watch_plan", [])
    buy_s = details.get("buy_score", 0)
    sell_s = details.get("sell_score", 0)
    zone_line = plan[0] if plan else "no zone identified"
    lo_hi_match = re.search(r"([\d.]+)-([\d.]+)", zone_line)
    dist_text = ""
    if lo_hi_match and price:
        lo, hi = float(lo_hi_match.group(1)), float(lo_hi_match.group(2))
        mid = (lo + hi) / 2
        dist = mid - price
        dist_text = f"  Mid {mid:.2f} ({dist:+.1f} pts from price)"

    action = "BUY" if bull else "SELL"
    emoji = "\U0001f441"  # eye emoji
    lines = [
        f"{emoji} WATCH_{action} {pair}",
        f"  Watching for {action} at {zone_line.split('(')[0].strip()}",
    ]
    if dist_text:
        lines.append(dist_text)
    lines.append(f"  Score: {buy_s}/{sell_s} | Price: {price:.2f}")
    if len(plan) > 1:
        lines.append(f"  {plan[1]}")
    return "\n".join(lines)


def _best_zone(details: dict, bull: bool, price: float):
    """Closest SMC zone (demand/supply, OB, FVG) matching the trade direction."""
    cands = []
    z = _parse_range(details.get("zone"))
    if z:
        ok = (z[0] == "demand") if bull else (z[0] == "supply")
        if ok:
            cands.append((f"{z[0].title()} zone", z[1], z[2]))
    ob = _parse_range(details.get("order_block"))
    if ob and ob[0] == "OB" and ("bullish" if bull else "bearish") in str(details.get("order_block", "")):
        cands.append((("Bullish OB" if bull else "Bearish OB"), ob[1], ob[2]))
    fv = _parse_range(details.get("fvg"))
    if fv and fv[0] == "FVG" and ("bullish" if bull else "bearish") in str(details.get("fvg", "")):
        cands.append((("Bullish FVG" if bull else "Bearish FVG"), fv[1], fv[2]))
    if not cands:
        return None
    return min(cands, key=lambda c: abs((c[1] + c[2]) / 2 - price))


def evaluate_entry_quality(result: dict, df, cfg: dict) -> dict | None:
    """Detect chasing: firm signal but price is stretched beyond its zone."""
    eqcfg = cfg.get("entry_quality", {})
    if not eqcfg.get("enabled", True):
        return None
    sig = result.get("signal", "")
    if sig not in FIRM_SIGNALS:
        return None
    price = float(result.get("price") or 0)
    if not price or df is None or len(df) < 30:
        return None
    atr_series = _atr(df, cfg.get("atr_period", 14))
    atr = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
    if not atr or atr != atr:
        return None
    bull = "BUY" in sig
    zone = _best_zone(result.get("details", {}), bull, price)
    if not zone:
        return None
    _, lo, hi = zone
    ref = hi if bull else lo
    dist = ((price - ref) if bull else (ref - price)) / atr
    max_chase = float(eqcfg.get("max_chase_atr", 1.2))
    d = result.get("details", {})
    return {
        "bull": bull, "zone": zone, "atr": atr,
        "dist_atr": round(dist, 2), "chasing": dist > max_chase,
        "sweep": d.get("nearest_ssl" if bull else "nearest_bsl"),
        "invalid": (float(df["Low"].tail(20).min()) if bull
                    else float(df["High"].tail(20).max())),
    }


def build_pending_plan(pair: str, result: dict, q: dict, cfg: dict, now_utc: datetime) -> dict:
    price = float(result.get("price"))
    atr = q["atr"]
    name, lo, hi = q["zone"]
    bull = q["bull"]
    rr = float(cfg.get("risk_reward", 2.0))
    buf = float(cfg.get("sl_atr_buffer", 1.0))

    entry = hi if bull else lo
    risk = buf * atr
    sl = entry - risk if bull else entry + risk
    tp = entry + risk * rr if bull else entry - risk * rr

    sweep = None
    raw_sweep = q.get("sweep")
    try:
        sweep = float(raw_sweep) if raw_sweep else None
    except (TypeError, ValueError):
        sweep = None

    return {
        "pair": pair, "side": "long" if bull else "short",
        "zone_name": name, "zone_lo": lo, "zone_hi": hi,
        "now_price": price, "dist_atr": q["dist_atr"],
        "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
        "rr": rr, "sweep": sweep, "invalid": round(q["invalid"], 2),
        "status": "waiting",
        "created": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "created_ts": now_utc.timestamp(),
        "mtf": result.get("details", {}).get("mtf_summary"),
        "news": result.get("details", {}).get("news_hold"),
    }


def format_pending_plan(pair: str, plan: dict, cfg: dict) -> str:
    long = plan["side"] == "long"
    emoji = "[PENDING]"
    side_word = "LONG" if long else "SHORT"
    dir_word = "BUY" if long else "SELL"
    lines = [
        f"{emoji} {pair} | {side_word} SETUP READY - WAIT, DON'T CHASE",
        f"Price now {plan['now_price']:.2f} is {plan['dist_atr']} ATR past the {plan['zone_name']} "
        f"{plan['zone_lo']:.2f}-{plan['zone_hi']:.2f}. Let it come back.",
        "",
        f"Plan A - Pullback {dir_word}:",
        f"   {dir_word} LIMIT {plan['entry']:.2f}",
        f"   SL {plan['sl']:.2f} | TP {plan['tp']:.2f} | R:R 1:{plan['rr']:.1f}",
    ]
    if plan.get("sweep"):
        verb = "dip under" if long else "spike above"
        act = "reclaim" if long else "reject"
        lines += [
            f"Plan B - Sweep {dir_word}:",
            f"   wait for {verb} {plan['sweep']:.2f}, then {act} -> enter",
        ]
    cmp_ = "<" if long else ">"
    lines += [
        "",
        f"Cancel if 5m closes {cmp_} {plan['invalid']:.2f}",
        f"Valid up to 24h | Created {plan['created']}",
    ]
    if plan.get("mtf"):
        lines.append(f"MTF: {plan['mtf']}")
    try:
        nl = next_event_line(cfg)
    except Exception:
        nl = None
    if nl:
        lines.append(f"News: {nl}")
    return "\n".join(lines)


def check_pending(pair: str, df, plans: dict[str, dict]) -> tuple[str | None, str]:
    """Update a waiting plan against the latest candle; returns (message, new_status)."""
    pl = plans.get(pair)
    if not pl or pl["status"] != "waiting":
        return None, ""
    last = df.iloc[-1]
    close, low, high = float(last["Close"]), float(last["Low"]), float(last["High"])
    long = pl["side"] == "long"

    invalid_hit = close < pl["invalid"] if long else close > pl["invalid"]
    touched = low <= pl["entry"] if long else high >= pl["entry"]
    swept = bool(pl.get("sweep")) and (low <= pl["sweep"] if long else high >= pl["sweep"])

    if time.time() - pl.get("created_ts", time.time()) > 86400:
        pl["status"] = "expired"
        return f"[EXPIRED] {pair} {pl['side'].upper()} plan expired after 24h without a touch", "expired"
    if invalid_hit:
        pl["status"] = "cancelled"
        word = "below" if long else "above"
        return (f"[CANCEL] {pair} {pl['side'].upper()} plan cancelled - closed {word} "
                f"{pl['invalid']:.2f}"), "cancelled"
    if swept:
        pl["status"] = "triggered"
        return (f"[HIT] {pair} {pl['side'].upper()} SWEEP HIT {pl['sweep']:.2f} - "
                f"A+ zone reached, watch for entry per plan"), "triggered"
    if touched:
        pl["status"] = "triggered"
        return (f"[HIT] {pair} {pl['side'].upper()} PLAN TRIGGERED @ ~{pl['entry']:.2f} - "
                f"entry zone reached. SL {pl['sl']:.2f} | TP {pl['tp']:.2f}"), "triggered"
    return None, "waiting"


_bias_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_BIAS_TTL_SECS = 600


def _fetch_bias(pair: str, cfg: dict) -> int:
    """Higher-timeframe bias from a cached frame; refetches at most once per TTL."""
    key = f"{pair}|{cfg['timeframe']}"
    now = time.time()
    cached = _bias_cache.get(key)
    if cached and now - cached[0] < _BIAS_TTL_SECS:
        return compute_bias(cached[1], cfg)
    bdf = fetch_forex_data(pair, cfg["bias_timeframe"], cfg.get("lookback_days", 30))
    _bias_cache[key] = (now, bdf)
    return compute_bias(bdf, cfg)


def _fallback_df(pair: str, cfg: dict) -> pd.DataFrame:
    """Last resort when yfinance is unreachable: locally stored candles."""
    df = load_candles(pair, cfg["timeframe"])
    if df is None or len(df) < int(cfg.get("ma_slow", 30)) + 20:
        raise RuntimeError(f"no usable price data for {pair}")
    fresh, msg = validate_data_freshness(df, cfg.get("max_data_age_minutes", 20))
    if not fresh:
        raise RuntimeError(f"stored candles stale for {pair}: {msg}")
    return df


def analyze_pair(pair: str, cfg: dict) -> tuple[dict, pd.DataFrame]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        df_future = executor.submit(fetch_forex_data, pair, cfg["timeframe"], cfg.get("lookback_days", 30))
        live_future = executor.submit(fetch_gold_spot_price)

        try:
            df = df_future.result(timeout=45)
        except Exception:
            df = None
        try:
            live_price, ts, source = live_future.result(timeout=30)
        except Exception:
            live_price, ts, source = None, None, None

    if df is None:
        df = _fallback_df(pair, cfg)
        print(f"[{pair}] price feed unavailable - using local store fallback ({len(df)} candles)", file=sys.stderr)

    bias = 0
    if cfg.get("bias_enabled", True) and cfg.get("bias_timeframe"):
        try:
            bias = _fetch_bias(pair, cfg)
        except Exception:
            bias = 0

    fresh, age_msg = validate_data_freshness(df, cfg.get("max_data_age_minutes", 10))
    if not fresh:
        print(f"  [{pair}] WARNING: {age_msg}")

    result = analyze(df, cfg, bias=bias, live_price=live_price)
    result["bias"] = bias
    if live_price:
        result["live_price"] = live_price
        result["live_source"] = source
    
    realtime_price = _get_realtime_price()
    if realtime_price:
        result["realtime_price"] = realtime_price
    
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
    plan = result.get("details", {}).get("watch_plan")
    if plan and signal.startswith("WATCH"):
        action = "BUY" if "BUY" in signal else "SELL"
        first_zone = plan[0].split("(")[0].strip()
        lines.append(f"  Watch: {action} at {first_zone}")
        for lvl in plan[:2]:
            lines.append(f"    - {lvl}")
    elif plan:
        lines.append(f"  Levels: {'; '.join(plan[:2])}")
    try:
        nl = next_event_line(cfg)
    except Exception:
        nl = None
    if nl:
        lines.append(f"  News: {nl}")
    return "\n".join(lines)


def _htf_direction(mtf: dict | None) -> float:
    """Composite direction of the higher timeframes only (4H/1H/45M/15M).

    Excludes the 5m entry row so the entry timeframe cannot vote on its own
    trend. Range: [-1, 1]; |value| < 0.30 means no clear HTF direction.
    """
    if not mtf:
        return 0.0
    weights = {"4h": 0.30, "1h": 0.26, "45m": 0.24, "15m": 0.20}
    rows = [(d, weights.get(tf, 0.18)) for tf, d in mtf.get("rows", []) if tf in weights]
    if not rows:
        return 0.0
    wsum = sum(w for _, w in rows)
    return round(sum(d * w for d, w in rows) / wsum, 3) if wsum else 0.0


def format_feed_down(pair: str, now_utc: datetime, cfg: dict) -> str:
    ts = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"[{ts}] {pair} | PRICE FEED UNAVAILABLE\n"
        f"  TradingView data is temporarily unreachable - the market itself may still be open.\n"
        f"  Retrying every {cfg['interval_seconds']}s; signals resume automatically."
    )


def _polling_loop(cfg):
    print(f"Starting SMC Gold Signal Bot. Pairs: {', '.join(cfg['pairs'])}")
    print(f"Entry: {cfg['timeframe']} | Bias: {cfg.get('bias_timeframe', 'none')}")
    print(f"Check every {cfg['interval_seconds']}s | Hourly digest: {cfg.get('hourly_digest', False)}")
    print(f"Real-time price: {cfg.get('realtime_price_check', False)} | Price interval: {cfg.get('realtime_price_interval', 5)}s")
    print("Model: 4H/1H/45M/30M/15M decide direction -> 5M gives the entry trigger")
    print("Telegram alerts fire only on confirmed entries. Standby info stays in this console.\n")

    last_signals: dict[str, str] = {}
    last_results: dict[str, dict] = {}
    last_dfs: dict[str, pd.DataFrame] = {}
    last_candle_ts: dict[str, pd.Timestamp] = {}
    last_digest_hour_ref = [None]
    feed_down: dict[str, bool] = {}
    last_brief_date_ref = [None]
    pending_plans: dict[str, dict] = {}
    tune_state: dict = {}

    try:
        journal.restore_open(journal.fetch_open_bot_trades(cfg))
        if journal._open:
            print(f"[journal] tracking open bot trades: {list(journal._open.values())}")
    except Exception as je:
        print(f"journal restore error: {je}", file=sys.stderr)

    while True:
        try:
            _poll_once(cfg, last_signals, last_results, last_dfs,
                       last_candle_ts, last_digest_hour_ref, feed_down,
                       last_brief_date_ref, pending_plans, tune_state)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[FATAL] Polling loop error (will retry in 30s): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            time.sleep(30)
            continue

        time.sleep(cfg["interval_seconds"])


def _poll_once(cfg, last_signals, last_results, last_dfs,
               last_candle_ts, last_digest_hour_ref, feed_down,
               last_brief_date_ref, pending_plans, tune_state):
    now_utc = datetime.now(timezone.utc)
    active = trading_allowed(now_utc, cfg)

    if cfg.get("news", {}).get("daily_brief", True) and last_brief_date_ref[0] != now_utc.date():
        try:
            brief = daily_brief(cfg)
        except Exception as be:
            print(f"news brief error: {be}", file=sys.stderr)
            brief = None
        if brief:
            print(brief)
            send_telegram(cfg, brief)
        last_brief_date_ref[0] = now_utc.date()

    if now_utc.weekday() == 6 and 21 <= now_utc.hour < 23:
        try:
            tune_msg = autotune.maybe_run(cfg, tune_state)
        except Exception as te:
            print(f"autotune error: {te}", file=sys.stderr)
            tune_msg = None
        if tune_msg:
            print(tune_msg)
            send_telegram(cfg, tune_msg)

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
                if not feed_down.get(pair):
                    message = format_feed_down(pair, now_utc, cfg)
                    print(message)
                    send_telegram(cfg, message)
                    feed_down[pair] = True
                continue
            if feed_down.pop(pair, None):
                message = f"{pair} | Price feed recovered - signal scanning resumed"
                print(message)
                send_telegram(cfg, message)

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

            try:
                ctx_lines = macro.build_context(df, now_utc)
                if ctx_lines:
                    details["context_lines"] = ctx_lines
            except Exception as ce:
                print(f"[{pair}] macro context error: {ce}", file=sys.stderr)

            mtf = None
            try:
                mtf = mtf_assess(pair, cfg, entry_result=result)
            except Exception as me:
                print(f"[{pair}] mtf error: {me}", file=sys.stderr)
            if mtf:
                details["mtf_summary"] = mtf["summary"]
                if signal in FIRM_SIGNALS and mtf["veto"]:
                    details["gated_from"] = details.get("gated_from") or signal
                    details["mtf_veto"] = True
                    signal = "WATCH_BUY" if "BUY" in signal else "WATCH_SELL"
                    result["signal"] = signal
                # Scalper filter: require 15m alignment for firm signals
                if (signal in FIRM_SIGNALS and cfg.get("scalper", {}).get("enabled", True)
                        and not mtf.get("scalp_15m_aligned", True)):
                    details["gated_from"] = details.get("gated_from") or signal
                    details["scalp_15m_blocked"] = True
                    signal = "WATCH_BUY" if "BUY" in signal else "WATCH_SELL"
                    result["signal"] = signal
                if mtf.get("high_confluence") and not mtf["veto"]:
                    details["mtf_high_confluence"] = True

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
                        if details.get("mtf_high_confluence"):
                            min_p = max(min_p - 0.05, 0.35)
                        if p_win < min_p:
                            details["gated_from"] = signal
                            signal = "WATCH_BUY" if "BUY" in signal else "WATCH_SELL"
                            result["signal"] = signal
                    elif p_win is None and assess.get("error"):
                        print(f"[{pair}] quant: {assess['error']}", file=sys.stderr)

            try:
                held, nev = blackout_check(cfg, now_utc)
            except Exception as ne:
                print(f"[{pair}] news error: {ne}", file=sys.stderr)
                held, nev = False, None
            if held and signal in FIRM_SIGNALS:
                details["news_hold"] = f"{nev['title']} @ {nev['dt']:%H:%M} UTC"
                details["gated_from"] = details.get("gated_from") or signal
                signal = "WATCH_BUY" if "BUY" in signal else "WATCH_SELL"
                result["signal"] = signal

            if signal.startswith("WATCH") and mtf:
                htf_comp = _htf_direction(mtf)
                htf_bullish = htf_comp > 0.20
                htf_bearish = htf_comp < -0.20
                pd_zone = details.get("pd_zone", "")
                if signal == "WATCH_SELL" and htf_bullish and pd_zone in ("deep_discount", "discount"):
                    details["zone_htf_block"] = "WATCH_SELL blocked: bullish HTF in discount zone"
                    signal = "NONE"
                    result["signal"] = signal
                elif signal == "WATCH_BUY" and htf_bearish and pd_zone in ("deep_premium", "premium"):
                    details["zone_htf_block"] = "WATCH_BUY blocked: bearish HTF in premium zone"
                    signal = "NONE"
                    result["signal"] = signal

            if signal.startswith("WATCH") and mtf:
                htf_comp = _htf_direction(mtf)
                pd_zone = details.get("pd_zone", "")
                sig_dir = 1 if "BUY" in signal else -1
                htf_aligned = (htf_comp > 0.20 and sig_dir > 0) or (htf_comp < -0.20 and sig_dir < 0)
                zone_match = (sig_dir > 0 and pd_zone in ("deep_discount", "discount")) or \
                             (sig_dir < 0 and pd_zone in ("deep_premium", "premium"))
                if htf_aligned and zone_match:
                    if sig_dir > 0:
                        details["buy_score"] = details.get("buy_score", 0) + 1
                    else:
                        details["sell_score"] = details.get("sell_score", 0) + 1
                    details["zone_htf_boost"] = True

            buy_score = details.get("buy_score", 0)
            sell_score = details.get("sell_score", 0)

            try:
                plan = build_watch_plan(result, df, cfg)
            except Exception as we:
                print(f"[{pair}] watch plan error: {we}", file=sys.stderr)
                plan = []
            if plan:
                details["watch_plan"] = plan

            # pending-plan lifecycle: trigger / cancel notifications
            if pending_plans.get(pair, {}).get("status") == "waiting":
                try:
                    pmsg, _ = check_pending(pair, df, pending_plans)
                    if pmsg:
                        print(pmsg)
                        send_telegram(cfg, pmsg)
                        last_signals[pair] = f"PENDING_{pending_plans[pair]['status']}"
                        if pending_plans[pair].get("status") == "triggered":
                            try:
                                journal.log_pending_trigger(cfg, pair, pending_plans[pair])
                            except Exception as je:
                                print(f"[{pair}] journal error: {je}", file=sys.stderr)
                except Exception as pe:
                    print(f"[{pair}] pending check error: {pe}", file=sys.stderr)

            try:
                for jmsg in journal.check_exits(cfg, {pair: float(df["Close"].iloc[-1])}):
                    print(jmsg)
                    send_telegram(cfg, jmsg)
            except Exception as je:
                print(f"[{pair}] journal exit error: {je}", file=sys.stderr)

            quality = None
            try:
                quality = evaluate_entry_quality(result, df, cfg)
            except Exception as ee:
                print(f"[{pair}] entry quality error: {ee}", file=sys.stderr)

            if signal in ("STRONG_BUY", "BUY", "STRONG_SELL", "SELL"):
                sig_up = "BUY" in signal
                htf_comp = _htf_direction(mtf)
                htf_clear = abs(htf_comp) >= 0.30
                htf_ok = (mtf is None) or (htf_clear and ((htf_comp > 0) == sig_up))
                last_bar = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
                candle_ok = bool(last_bar["close"] > last_bar["open"]) if sig_up \
                    else bool(last_bar["close"] < last_bar["open"])

                details["htf_direction"] = (
                    "LONG" if htf_comp > 0.30 else "SHORT" if htf_comp < -0.30 else "RANGE"
                )

                # Scalper: require recent momentum burst or squeeze release
                scalp_burst = True
                if cfg.get("scalper", {}).get("enabled", True) and df is not None and len(df) > 5:
                    try:
                        from strategy import _detect_ema_squeeze
                        sq = _detect_ema_squeeze(df["Close"], cfg)
                        details["squeeze"] = "releasing" if sq.get("releasing") else ("contracted" if sq.get("squeeze") else "normal")
                        # If squeeze is contracted and not releasing, still allow but note it
                        # Momentum burst: last candle body > 0.5 ATR
                        atr_val = float(_atr(df, cfg.get("atr_period", 14)).iloc[-1])
                        last_bar = df.iloc[-1]
                        burst_size = abs(float(last_bar["Close"]) - float(last_bar["Open"]))
                        if burst_size < 0.4 * atr_val:
                            details["momentum_burst"] = False
                        else:
                            details["momentum_burst"] = True
                    except Exception:
                        pass

                if not htf_ok:
                    why = "RANGE" if not htf_clear else ("bearish HTF" if sig_up else "bullish HTF")
                    print(f"[{pair}] {signal} blocked - higher timeframes {why} "
                          f"(HTF {htf_comp:+.2f}) | 5m B{buy_score}/S{sell_score}")
                    last_signals[pair] = "STANDBY"
                    pending_plans.pop(pair, None)
                elif not candle_ok:
                    side = "bullish" if sig_up else "bearish"
                    print(f"[{pair}] {signal} blocked - waiting for a confirming "
                          f"{side} 5m close | B{buy_score}/S{sell_score}")
                    last_signals[pair] = "STANDBY"
                elif quality and quality.get("chasing"):
                    new_plan = build_pending_plan(pair, result, quality, cfg, now_utc)
                    prev = pending_plans.get(pair)
                    stale = (not prev or prev.get("status") != "waiting"
                             or prev.get("side") != new_plan["side"]
                             or abs(prev.get("entry", 0) - new_plan["entry"]) > 0.25 * quality["atr"])
                    if stale:
                        pending_plans[pair] = new_plan
                        message = format_pending_plan(pair, new_plan, cfg)
                        print(message)
                        send_telegram(cfg, message)
                        last_signals[pair] = f"PENDING_{new_plan['side']}"
                else:
                    pending_plans.pop(pair, None)
                    if last_signals.get(pair) != signal:
                        message = format_alert(pair, result)
                        print(message)
                        send_telegram(cfg, message)
                        send_webhook(cfg, pair, result)
                        chart = generate_signal_chart(df, cfg, result, pair)
                        if chart:
                            send_telegram_photo(cfg, chart, message)
                        last_signals[pair] = signal
                        try:
                            journal.log_entry(cfg, pair, result)
                        except Exception as je:
                            print(f"[{pair}] journal error: {je}", file=sys.stderr)
            else:
                last_signals[pair] = signal if str(signal).startswith("WATCH") else "NONE"
                dom = "BUY" if buy_score > sell_score else "SELL" if sell_score > buy_score else None
                htf_txt = details.get("mtf_summary", "HTF warming up")
                lean = f" | leaning {dom} ({max(buy_score, sell_score)}/{threshold})" if dom else ""
                sess = next((c for c in details.get("context_lines", []) if c.startswith("Session")), "")
                watch_lvl = _watch_level_line(details, signal, float(result.get("price") or 0))
                print(f"[{pair}] {signal} | {htf_txt}{watch_lvl}{lean}" + (f" | {sess}" if sess else ""))

                if cfg.get("send_watch_alerts", False) and signal.startswith("WATCH") and watch_lvl:
                    try:
                        wmsg = _format_watch_telegram(pair, result, details, signal)
                        send_telegram(cfg, wmsg)
                    except Exception as we:
                        print(f"[{pair}] watch alert error: {we}", file=sys.stderr)

        except Exception as e:
            print(f"[{pair}] Error: {e}", file=sys.stderr)

    if cfg.get("hourly_digest", False) and last_digest_hour_ref[0] != now_utc.hour:
        last_digest_hour_ref[0] = now_utc.hour
        for pair in cfg["pairs"]:
            result = last_results.get(pair) if active else None
            if result is None:
                result = {"signal": "NONE", "bias": 0}
            message = format_digest(pair, result, now_utc, cfg)
            print(message)
            send_telegram(cfg, message)
            df = last_dfs.get(pair)
            if df is not None and result.get("signal", "NONE") != "NONE":
                try:
                    chart = generate_signal_chart(df, cfg, result, pair)
                    if chart:
                        send_telegram_photo(cfg, chart, message)
                except Exception as ce:
                    print(f"[{pair}] chart error: {ce}", file=sys.stderr)


def _realtime_price_stream(cfg):
    """Background thread for real-time price updates."""
    interval = cfg.get("realtime_price_interval", 5)
    print(f"[realtime] Price stream started (every {interval}s)")
    
    while True:
        try:
            live_price, ts, source = fetch_gold_spot_price()
            if live_price and live_price > 1000:
                with _realtime_price_cache["lock"]:
                    _realtime_price_cache["price"] = live_price
                    _realtime_price_cache["timestamp"] = datetime.now(timezone.utc)
        except Exception as e:
            print(f"[realtime] Price stream error: {e}", file=sys.stderr)
        
        time.sleep(interval)


def main() -> None:
    cfg = load_config()

    port = int(os.environ.get("PORT", cfg.get("webhook_server", {}).get("port", 8080)))
    host = cfg.get("webhook_server", {}).get("host", "0.0.0.0")
    use_webhook_server = cfg.get("webhook_server", {}).get("enabled", True)

    if use_webhook_server:
        poll_thread = threading.Thread(target=_polling_loop, args=(cfg,), daemon=True)
        poll_thread.start()
        print(f"Polling loop started in background")
        
        if cfg.get("realtime_price_check", False):
            price_thread = threading.Thread(target=_realtime_price_stream, args=(cfg,), daemon=True)
            price_thread.start()
            print(f"Real-time price stream started")
        
        print(f"Webhook server starting on {host}:{port}")
        print(f"  TradingView alert URL: http://YOUR_WEBHOOK_URL/webhook")
        print(f"  Health check: http://YOUR_WEBHOOK_URL/health")
        print(f"  Real-time price: http://YOUR_WEBHOOK_URL/price")
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
    import signal as _signal
    _state = {"shutdown": False}
    def _handle_exit(signum, frame):
        _state["shutdown"] = True
    _signal.signal(_signal.SIGINT, _handle_exit)
    _signal.signal(_signal.SIGTERM, _handle_exit)
    while not _state["shutdown"]:
        try:
            main()
            break
        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except Exception as e:
            print(f"\n[FATAL] Bot crashed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            print("Restarting in 15 seconds...", file=sys.stderr)
            for _ in range(15):
                if _state["shutdown"]:
                    break
                time.sleep(1)
            if not _state["shutdown"]:
                print("Restarting bot...", file=sys.stderr)
