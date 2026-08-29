"""MJBot Brain -- Self-taught intelligence system for gold trading.

Tracks session performance, pattern memory, and adapts thresholds based on
rolling win rates. Provides a confidence score for each signal based on
historical performance data.
"""
import json
import math
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

_DATA_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
_DATA_DIR.mkdir(exist_ok=True)

_SESSION_FILE = _DATA_DIR / "session_memory.json"
_PATTERN_FILE = _DATA_DIR / "pattern_memory.json"
_PERF_FILE = _DATA_DIR / "performance.json"


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _hour_bucket(ts: datetime) -> int:
    return ts.hour


def _session_name(hour: int) -> str:
    if 0 <= hour < 7:
        return "asian"
    elif 7 <= hour < 12:
        return "london"
    elif 12 <= hour < 15:
        return "ny_am"
    elif 15 <= hour < 18:
        return "ny_pm"
    else:
        return "off_hours"


def _setup_fingerprint(result: dict, details: dict) -> str:
    parts = []
    for key in ("pd_zone", "htf_direction"):
        val = details.get(key, "")
        if val:
            parts.append(val)
    factors = []
    for f in ("order_block", "fvg", "breaker_block", "mitigation_block",
              "sweep", "structure_event", "elliott_oabc", "crt", "zone"):
        if details.get(f):
            factors.append(f)
    if not factors:
        score = max(details.get("buy_score", 0), details.get("sell_score", 0))
        if score >= 5:
            factors.append("strong_score")
    parts.extend(sorted(factors)[:4])
    return "|".join(parts) if parts else "generic"


class Brain:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.session_data = _load(_SESSION_FILE)
        self.pattern_data = _load(_PATTERN_FILE)
        self.perf_data = _load(_PERF_FILE)
        self._ensure_structure()

    def _ensure_structure(self):
        if "sessions" not in self.session_data:
            self.session_data["sessions"] = {}
        if "patterns" not in self.pattern_data:
            self.pattern_data["patterns"] = {}
        if "trades" not in self.perf_data:
            self.perf_data["trades"] = []
        if "daily_counts" not in self.perf_data:
            self.perf_data["daily_counts"] = {}

    def record_trade_outcome(self, pair: str, result: dict, details: dict,
                              outcome: str, pnl: float = 0, r_multiple: float = 0):
        now = datetime.now(timezone.utc)
        hour = _hour_bucket(now)
        session = _session_name(hour)
        fingerprint = _setup_fingerprint(result, details)

        sess_key = f"{pair}|{session}"
        if sess_key not in self.session_data["sessions"]:
            self.session_data["sessions"][sess_key] = {
                "wins": 0, "losses": 0, "total_r": 0.0, "trades": []
            }
        sd = self.session_data["sessions"][sess_key]
        is_win = outcome == "win" or pnl > 0
        if is_win:
            sd["wins"] += 1
        else:
            sd["losses"] += 1
        sd["total_r"] = round(sd.get("total_r", 0) + r_multiple, 2)
        sd["trades"].append({
            "ts": now.isoformat(), "outcome": outcome,
            "pnl": round(pnl, 2), "r": round(r_multiple, 2),
            "signal": result.get("signal"), "fingerprint": fingerprint,
            "hour": hour,
        })
        sd["trades"] = sd["trades"][-100:]

        if fingerprint not in self.pattern_data["patterns"]:
            self.pattern_data["patterns"][fingerprint] = {
                "wins": 0, "losses": 0, "total_r": 0.0, "examples": []
            }
        pd = self.pattern_data["patterns"][fingerprint]
        if is_win:
            pd["wins"] += 1
        else:
            pd["losses"] += 1
        pd["total_r"] = round(pd.get("total_r", 0) + r_multiple, 2)
        pd["examples"].append({
            "ts": now.isoformat(), "outcome": outcome,
            "pnl": round(pnl, 2), "signal": result.get("signal"),
        })
        pd["examples"] = pd["examples"][-20:]

        self.perf_data["trades"].append({
            "ts": now.isoformat(), "pair": pair, "outcome": outcome,
            "pnl": round(pnl, 2), "r": round(r_multiple, 2),
            "session": session, "fingerprint": fingerprint,
            "hour": hour, "signal": result.get("signal"),
        })
        self.perf_data["trades"] = self.perf_data["trades"][-500:]

        day_key = now.strftime("%Y-%m-%d")
        dc = self.perf_data.get("daily_counts", {})
        if day_key not in dc:
            dc[day_key] = {"wins": 0, "losses": 0, "trades": 0}
        dc[day_key]["trades"] += 1
        if is_win:
            dc[day_key]["wins"] += 1
        else:
            dc[day_key]["losses"] += 1
        self.perf_data["daily_counts"] = dc

        _save(_SESSION_FILE, self.session_data)
        _save(_PATTERN_FILE, self.pattern_data)
        _save(_PERF_FILE, self.perf_data)

    def session_win_rate(self, pair: str, hour: int) -> float | None:
        session = _session_name(hour)
        key = f"{pair}|{session}"
        sd = self.session_data["sessions"].get(key)
        if not sd:
            return None
        total = sd.get("wins", 0) + sd.get("losses", 0)
        if total < 5:
            return None
        return sd["wins"] / total

    def pattern_confidence(self, fingerprint: str) -> float | None:
        pd = self.pattern_data["patterns"].get(fingerprint)
        if not pd:
            return None
        total = pd.get("wins", 0) + pd.get("losses", 0)
        if total < 3:
            return None
        return pd["wins"] / total

    def rolling_win_rate(self, days: int = 30) -> float | None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        trades = [
            t for t in self.perf_data.get("trades", [])
            if datetime.fromisoformat(t["ts"]) > cutoff
        ]
        if len(trades) < 10:
            return None
        wins = sum(1 for t in trades if t["outcome"] == "win")
        return wins / len(trades)

    def rolling_r_multiple(self, days: int = 30) -> float | None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        trades = [
            t for t in self.perf_data.get("trades", [])
            if datetime.fromisoformat(t["ts"]) > cutoff
        ]
        if len(trades) < 5:
            return None
        return sum(t.get("r", 0) for t in trades) / len(trades)

    def best_sessions(self, pair: str, top_n: int = 2) -> list[str]:
        results = []
        for key, sd in self.session_data["sessions"].items():
            if not key.startswith(pair + "|"):
                continue
            session = key.split("|")[1]
            total = sd.get("wins", 0) + sd.get("losses", 0)
            if total < 5:
                continue
            wr = sd["wins"] / total
            results.append((session, wr, sd.get("total_r", 0)))
        results.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [r[0] for r in results[:top_n]]

    def worst_sessions(self, pair: str, bottom_n: int = 1) -> list[str]:
        results = []
        for key, sd in self.session_data["sessions"].items():
            if not key.startswith(pair + "|"):
                continue
            session = key.split("|")[1]
            total = sd.get("wins", 0) + sd.get("losses", 0)
            if total < 5:
                continue
            wr = sd["wins"] / total
            results.append((session, wr, sd.get("total_r", 0)))
        results.sort(key=lambda x: (x[1], x[2]))
        return [r[0] for r in results[:bottom_n]]

    def compute_confidence(self, pair: str, result: dict, details: dict, mtf: dict | None) -> float:
        weights = self.cfg.get("smart_engine", {}).get("confidence_weights", {})
        w_struct = weights.get("structure", 0.25)
        w_session = weights.get("session", 0.20)
        w_mtf = weights.get("mtf", 0.25)
        w_quant = weights.get("quant", 0.15)
        w_pattern = weights.get("pattern_memory", 0.15)

        struct_score = 0.5
        buy_s = details.get("buy_score", 0)
        sell_s = details.get("sell_score", 0)
        max_s = max(buy_s, sell_s)
        if max_s >= 7:
            struct_score = 0.9
        elif max_s >= 5:
            struct_score = 0.7
        elif max_s >= 3:
            struct_score = 0.55

        now = datetime.now(timezone.utc)
        session_wr = self.session_win_rate(pair, now.hour)
        session_score = 0.5
        if session_wr is not None:
            session_score = session_wr

        mtf_score = 0.5
        if mtf:
            composite = abs(mtf.get("composite", 0))
            if mtf.get("high_confluence"):
                mtf_score = 0.9
            elif composite >= 0.4:
                mtf_score = 0.7
            elif composite >= 0.2:
                mtf_score = 0.55

        quant_score = 0.5
        p_win = details.get("p_win")
        if p_win is not None:
            quant_score = min(max(p_win, 0.0), 1.0)

        fingerprint = _setup_fingerprint(result, details)
        pattern_wr = self.pattern_confidence(fingerprint)
        pattern_score = 0.5
        if pattern_wr is not None:
            pattern_score = pattern_wr

        confidence = (
            struct_score * w_struct +
            session_score * w_session +
            mtf_score * w_mtf +
            quant_score * w_quant +
            pattern_score * w_pattern
        )

        rolling_wr = self.rolling_win_rate(30)
        if rolling_wr is not None:
            if rolling_wr >= 0.55:
                confidence = min(confidence * 1.05, 1.0)
            elif rolling_wr <= 0.40:
                confidence = confidence * 0.92

        return round(min(max(confidence, 0.0), 1.0), 3)

    def suggested_threshold_adjustment(self) -> float:
        rolling_wr = self.rolling_win_rate(30)
        if rolling_wr is None:
            return 0.0
        target = self.cfg.get("adaptive", {}).get("win_rate_target", 0.50)
        step = self.cfg.get("adaptive", {}).get("threshold_adjust_step", 0.02)
        if rolling_wr >= target + 0.05:
            return -step
        elif rolling_wr <= target - 0.05:
            return step
        return 0.0

    def should_skip_session(self, pair: str, hour: int) -> bool:
        worst = self.worst_sessions(pair, 1)
        if worst and _session_name(hour) == worst[0]:
            session_wr = self.session_win_rate(pair, hour)
            if session_wr is not None and session_wr < 0.30:
                return True
        return False

    def get_session_briefing(self, pair: str) -> str:
        now = datetime.now(timezone.utc)
        parts = [f"Brain Session Briefing ({now:%H:%M} UTC)"]
        for sess in ["london", "ny_am", "ny_pm"]:
            for h in range(24):
                if _session_name(h) == sess:
                    wr = self.session_win_rate(pair, h)
                    if wr is not None:
                        emoji = "STRONG" if wr >= 0.55 else "WEAK" if wr < 0.40 else "NEUTRAL"
                        parts.append(f"  {sess}: {wr:.0%} win rate [{emoji}]")
                    break
        rolling = self.rolling_win_rate(30)
        avg_r = self.rolling_r_multiple(30)
        if rolling is not None:
            parts.append(f"  Rolling 30d: {rolling:.0%} WR, avg R: {avg_r:+.2f}" if avg_r else f"  Rolling 30d: {rolling:.0%} WR")
        best = self.best_sessions(pair)
        if best:
            parts.append(f"  Best sessions: {', '.join(best)}")
        return "\n".join(parts)

    def get_pattern_report(self, top_n: int = 5) -> str:
        patterns = []
        for fp, pd in self.pattern_data.get("patterns", {}).items():
            total = pd.get("wins", 0) + pd.get("losses", 0)
            if total < 3:
                continue
            wr = pd["wins"] / total
            patterns.append((fp, wr, pd.get("total_r", 0), total))
        patterns.sort(key=lambda x: (x[1], x[2]), reverse=True)
        if not patterns:
            return "No pattern data yet (need 3+ trades per pattern)"
        lines = ["Pattern Memory Report"]
        for fp, wr, total_r, count in patterns[:top_n]:
            tag = "PROFITABLE" if wr >= 0.55 else "LOSING" if wr < 0.40 else "NEUTRAL"
            lines.append(f"  {fp}: {wr:.0%} WR ({count} trades, {total_r:+.1f}R) [{tag}]")
        return "\n".join(lines)

    def get_learning_summary(self, pair: str) -> str:
        """Generate a full learning summary for Telegram."""
        now = datetime.now(timezone.utc)
        lines = [
            f"🧠 BRAIN LEARNING SUMMARY",
            f"📅 {now.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]

        # Pattern learnings
        patterns = []
        for fp, pd in self.pattern_data.get("patterns", {}).items():
            total = pd.get("wins", 0) + pd.get("losses", 0)
            if total < 2:
                continue
            wr = pd["wins"] / total
            patterns.append((fp, wr, pd.get("total_r", 0), total))
        patterns.sort(key=lambda x: (x[1], x[2]), reverse=True)

        if patterns:
            lines.append("📊 PATTERNS LEARNED:")
            blocked_patterns = []
            active_patterns = []
            for fp, wr, total_r, count in patterns:
                if wr >= 0.55:
                    tag = "✅ ACTIVE"
                    active_patterns.append(fp)
                elif wr < 0.40:
                    tag = "🚫 BLOCKED"
                    blocked_patterns.append(fp)
                else:
                    tag = "⚠️ NEUTRAL"
                lines.append(f"  {fp}: {wr:.0%} WR ({count}t, {total_r:+.1f}R) {tag}")
            lines.append("")

            if blocked_patterns:
                lines.append("🚫 AUTO-BLOCKED PATTERNS:")
                for p in blocked_patterns:
                    lines.append(f"  • {p}")
                lines.append("")

            if active_patterns:
                lines.append("✅ TRUSTED PATTERNS:")
                for p in active_patterns:
                    lines.append(f"  • {p}")
                lines.append("")

        # Session learnings
        lines.append("⏰ SESSIONS LEARNED:")
        for sess in ["london", "ny_am", "ny_pm"]:
            for h in range(24):
                from macro import _session_name
                if _session_name(h) == sess:
                    wr = self.session_win_rate(pair, h)
                    if wr is not None:
                        if wr >= 0.55:
                            tag = "✅ ACTIVE"
                        elif wr < 0.30:
                            tag = "🚫 BLOCKED"
                        else:
                            tag = "⚠️ NEUTRAL"
                        lines.append(f"  {sess}: {wr:.0%} WR {tag}")
                    break
        lines.append("")

        # Overall stats
        rolling = self.rolling_win_rate(30)
        avg_r = self.rolling_r_multiple(30)
        if rolling is not None:
            lines.append(f"📈 OVERALL (30d): {rolling:.0%} WR")
            if avg_r is not None:
                lines.append(f"📊 Avg R-Multiple: {avg_r:+.2f}")
        lines.append("")

        # What's being enforced
        lines.append("🔒 ENFORCEMENT ACTIVE:")
        lines.append("  • Bad sessions auto-blocked (WR < 30%)")
        lines.append("  • Bad patterns auto-blocked (WR < 40%)")
        lines.append("  • Low confidence blocked (< 0.40)")
        lines.append("  • 12:00-13:00 UTC blackout")

        return "\n".join(lines)

    def get_weekly_learning_summary(self, pair: str) -> str:
        """Generate weekly learning summary for Friday 6PM Kenya time."""
        now = datetime.now(timezone.utc)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        # Get this week's trades
        week_trades = [
            t for t in self.perf_data.get("trades", [])
            if datetime.fromisoformat(t["ts"]).replace(tzinfo=timezone.utc) >= week_start
        ]

        week_wins = sum(1 for t in week_trades if t.get("outcome") == "win")
        week_losses = sum(1 for t in week_trades if t.get("outcome") == "loss")
        week_total = week_wins + week_losses
        week_r = sum(t.get("r", 0) for t in week_trades)
        week_pnl = sum(t.get("pnl", 0) for t in week_trades)
        week_wr = week_wins / week_total if week_total > 0 else 0

        lines = [
            f"🧠 WEEKLY LEARNING REPORT",
            f"📅 Week of {week_start.strftime('%b %d')} - {now.strftime('%b %d, %Y')}",
            f"🕐 Sent: {now.strftime('%A %I:%M %p')} EAT",
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            "📊 THIS WEEK'S PERFORMANCE:",
            f"  Trades: {week_total} ({week_wins}W / {week_losses}L)",
            f"  Win Rate: {week_wr:.0%}",
            f"  R-Multiple: {week_r:+.1f}R",
            f"  PnL: ${week_pnl:+.2f}",
            "",
        ]

        # Session breakdown
        lines.append("⏰ SESSION BREAKDOWN:")
        session_stats = {}
        for t in week_trades:
            sess = t.get("session", "unknown")
            if sess not in session_stats:
                session_stats[sess] = {"wins": 0, "losses": 0, "r": 0}
            if t.get("outcome") == "win":
                session_stats[sess]["wins"] += 1
            else:
                session_stats[sess]["losses"] += 1
            session_stats[sess]["r"] += t.get("r", 0)

        for sess, stats in sorted(session_stats.items()):
            total = stats["wins"] + stats["losses"]
            wr = stats["wins"] / total if total > 0 else 0
            tag = "✅" if wr >= 0.55 else "🚫" if wr < 0.40 else "⚠️"
            lines.append(f"  {tag} {sess}: {wr:.0%} WR ({stats['wins']}W/{stats['losses']}L, {stats['r']:+.1f}R)")
        lines.append("")

        # Pattern breakdown
        lines.append("🔍 PATTERN BREAKDOWN:")
        pattern_stats = {}
        for t in week_trades:
            fp = t.get("fingerprint", "unknown")
            if fp not in pattern_stats:
                pattern_stats[fp] = {"wins": 0, "losses": 0, "r": 0}
            if t.get("outcome") == "win":
                pattern_stats[fp]["wins"] += 1
            else:
                pattern_stats[fp]["losses"] += 1
            pattern_stats[fp]["r"] += t.get("r", 0)

        for fp, stats in sorted(pattern_stats.items()):
            total = stats["wins"] + stats["losses"]
            wr = stats["wins"] / total if total > 0 else 0
            tag = "✅" if wr >= 0.55 else "🚫" if wr < 0.40 else "⚠️"
            lines.append(f"  {tag} {fp}: {wr:.0%} WR ({stats['wins']}W/{stats['losses']}L, {stats['r']:+.1f}R)")
        lines.append("")

        # All-time patterns
        lines.append("📚 ALL-TIME PATTERN MEMORY:")
        for fp, pd in self.pattern_data.get("patterns", {}).items():
            total = pd.get("wins", 0) + pd.get("losses", 0)
            if total < 2:
                continue
            wr = pd["wins"] / total
            tag = "✅ TRUSTED" if wr >= 0.55 else "🚫 BLOCKED" if wr < 0.40 else "⚠️ NEUTRAL"
            lines.append(f"  {tag} {fp}: {wr:.0%} WR ({total}t, {pd.get('total_r', 0):+.1f}R)")
        lines.append("")

        # All-time sessions
        lines.append("📚 ALL-TIME SESSION MEMORY:")
        for sess in ["london", "ny_am", "ny_pm"]:
            for h in range(24):
                from macro import _session_name
                if _session_name(h) == sess:
                    wr = self.session_win_rate(pair, h)
                    if wr is not None:
                        tag = "✅ TRUSTED" if wr >= 0.55 else "🚫 BLOCKED" if wr < 0.30 else "⚠️ NEUTRAL"
                        lines.append(f"  {tag} {sess}: {wr:.0%} WR")
                    break
        lines.append("")

        # Enforcement status
        lines.append("🔒 ENFORCEMENT STATUS:")
        blocked_count = sum(1 for fp, pd in self.pattern_data.get("patterns", {}).items()
                           if (pd.get("wins", 0) + pd.get("losses", 0)) >= 3
                           and pd["wins"] / (pd.get("wins", 0) + pd.get("losses", 0)) < 0.40)
        lines.append(f"  • {blocked_count} patterns auto-blocked")
        lines.append(f"  • Bad sessions auto-blocked (WR < 30%)")
        lines.append(f"  • Low confidence blocked (< 0.40)")
        lines.append(f"  • 12:00-13:00 UTC blackout active")
        lines.append("")

        # Overall stats
        rolling = self.rolling_win_rate(30)
        avg_r = self.rolling_r_multiple(30)
        if rolling is not None:
            lines.append(f"📈 OVERALL (30d): {rolling:.0%} WR, avg R: {avg_r:+.2f}" if avg_r else f"📈 OVERALL (30d): {rolling:.0%} WR")

        return "\n".join(lines)

    def daily_reset(self):
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        old_dates = [
            k for k in self.perf_data.get("daily_counts", {})
            if datetime.strptime(k, "%Y-%m-%d").replace(tzinfo=timezone.utc) < cutoff
        ]
        for k in old_dates:
            del self.perf_data["daily_counts"][k]

        for key, sd in self.session_data.get("sessions", {}).items():
            if len(sd.get("trades", [])) > 100:
                sd["trades"] = sd["trades"][-100:]

        for key, pd in self.pattern_data.get("patterns", {}).items():
            if len(pd.get("examples", [])) > 20:
                pd["examples"] = pd["examples"][-20:]

        _save(_SESSION_FILE, self.session_data)
        _save(_PATTERN_FILE, self.pattern_data)
        _save(_PERF_FILE, self.perf_data)
