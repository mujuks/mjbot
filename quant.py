"""Meta-labeling quant layer (Lopez de Prado, Advances in Financial ML).

Primary model: the SMC signal engine (decides side).
Meta model   : logistic regression predicting P(entry hits TP before SL),
               trained walk-forward with purge/embargo (no lookahead).

Design notes
- Training events are ENTRY EVENTS (signal-state transitions), mirroring how
  bot.py alerts once per state change - not every bar carrying a signal.
- Labels are triple-barrier outcomes {0,1} using each signal's own ATR-based
  SL/TP, with an ATR-floor timeout rule.
- Features are strictly causal and identical at train and inference time
  (computed once over the full frame, indexed by bar).
- Gating: p_win >= min_probability keeps a firm BUY/SELL; below that the bot
  demotes to WATCH_*. Confidence above threshold scales suggested risk.
"""
import sys
import time
import traceback
from datetime import timezone

import numpy as np
import pandas as pd

from data_fetcher import fetch_forex_data
from data_store import load_candles
from strategy import ema, rsi, macd, build_signal_series, _atr

FEATURE_NAMES = [
    "side", "score_edge", "rsi_norm", "macd_hist_atr", "atr_pct",
    "vwap_dist_atr", "vol_z", "ema_spread_atr", "ret10_side",
    "hour_sin", "hour_cos",
]

# Computed into the base frame for research but EXCLUDED from the model:
# walk-forward showed they dilute OOS precision (40% vs 52% @0.55).
EXPERIMENTAL_FEATURES = ["vol_regime", "active_session", "mtf_align"]

FIRM = ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL")

_cache = {}


def _side_of(state: str) -> float:
    return 1.0 if "BUY" in state else -1.0


def compute_base_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Causal per-bar indicator columns (identical slicing at any index)."""
    close = df["Close"]
    vol = df["Volume"].astype(float)
    atr = _atr(df, cfg.get("atr_period", 14))
    safe_atr = atr.replace(0, np.nan)

    fast = ema(close, cfg["ma_fast"])
    slow = ema(close, cfg["ma_slow"])
    rsi_val = rsi(close, cfg["rsi_period"])
    _, _, hist = macd(close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])

    tv = (df["High"] + df["Low"] + close) / 3.0
    lb = int(cfg.get("vwap_lookback", 78))
    vwap = (tv * vol).rolling(lb, min_periods=1).sum() / vol.rolling(lb, min_periods=1).sum()

    v_mean = vol.rolling(20, min_periods=5).mean()
    v_std = vol.rolling(20, min_periods=5).std().replace(0, np.nan)

    hours = df.index.hour.to_numpy()
    f = pd.DataFrame(index=df.index)
    f["rsi_norm"] = ((rsi_val - 50.0) / 50.0).to_numpy()
    f["macd_hist_atr"] = (hist / safe_atr).clip(-10, 10).to_numpy()
    atr_pct = atr / close.replace(0, np.nan) * 100.0
    f["atr_pct"] = atr_pct.to_numpy()
    f["vwap_dist_atr"] = ((close - vwap) / safe_atr).clip(-10, 10).to_numpy()
    f["vol_z"] = ((vol - v_mean) / v_std).clip(-5, 5).fillna(0.0).to_numpy()
    f["ema_spread_atr"] = ((fast - slow) / safe_atr).clip(-15, 15).to_numpy()
    f["ret10"] = close.pct_change(10).fillna(0.0).to_numpy() * 100.0
    f["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    f["vol_regime"] = atr_pct.rolling(2000, min_periods=200).rank(pct=True).fillna(0.5).to_numpy()
    f["active_session"] = ((hours >= 12) & (hours < 17)).astype(float)
    return f


def _mtf_trend(pair: str, index: pd.DatetimeIndex, cfg: dict) -> pd.Series:
    """Higher-timeframe trend in {-1, 0, +1}, shifted one HTF bar (no lookahead)."""
    btf = cfg.get("bias_timeframe")
    if not cfg.get("bias_enabled", True) or not btf:
        return pd.Series(0.0, index=index)
    try:
        days = max(int(cfg.get("quant", {}).get("train_days", 45)), 30) + 2
        bdf = fetch_forex_data(pair, btf, days)
        if bdf is None or len(bdf) < int(cfg.get("ma_slow", 50)) + 5:
            raise ValueError("insufficient htf data")
        c = bdf["Close"]
        spread = ema(c, cfg.get("ma_fast", 20)) - ema(c, cfg.get("ma_slow", 50))
        atr_h = _atr(bdf, cfg.get("atr_period", 14)).replace(0, np.nan)
        strength = (spread / atr_h).abs()
        trend = pd.Series(np.where(spread > 0, 1.0, -1.0), index=bdf.index)
        trend[strength < 0.25] = 0.0
        return trend.shift(1).reindex(index, method="ffill").fillna(0.0)
    except Exception:
        return pd.Series(0.0, index=index)


def get_entry_positions(sig: pd.DataFrame) -> np.ndarray:
    """Positions where a DIRECTION starts (family transition), like real entries.

    BUY->STRONG_BUY escalations are not new entries; only NONE->long,
    NONE->short and long<->short flips count.
    """
    def fam(st):
        if "BUY" in st:
            return "L"
        if "SELL" in st:
            return "S"
        return "N"

    states = sig["signal"].to_numpy()
    out = []
    prev = "N"
    for i, st in enumerate(states):
        f = fam(st)
        if f != "N" and f != prev:
            out.append(i)
        prev = f
    return np.asarray(out, int)


def make_event_row(base: pd.DataFrame, i: int, side: float,
                   buy_score: float, sell_score: float) -> np.ndarray:
    total = max(buy_score + sell_score, 1.0)
    vals = {
        "side": side,
        "score_edge": side * (buy_score - sell_score) / total,
        "rsi_norm": base["rsi_norm"].iat[i] * side,
        "macd_hist_atr": np.tanh(base["macd_hist_atr"].iat[i]) * side,
        "atr_pct": base["atr_pct"].iat[i],
        "vwap_dist_atr": np.tanh(base["vwap_dist_atr"].iat[i]) * side,
        "vol_z": base["vol_z"].iat[i],
        "ema_spread_atr": np.tanh(base["ema_spread_atr"].iat[i]) * side,
        "ret10_side": np.tanh(base["ret10"].iat[i] / 2.0) * side,
        "hour_sin": base["hour_sin"].iat[i],
        "hour_cos": base["hour_cos"].iat[i],
    }
    return np.array([vals[k] for k in FEATURE_NAMES], float)


def label_events(df: pd.DataFrame, sig: pd.DataFrame, positions: np.ndarray,
                 cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Triple-barrier labels {1: TP first, 0: SL first or timeout-loss}."""
    qcfg = cfg.get("quant", {})
    horizon = int(qcfg.get("horizon_bars", 60))
    floor_mult = float(qcfg.get("timeout_floor_atr", 0.25))
    atr = _atr(df, cfg.get("atr_period", 14)).to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    n = len(df)

    entry = sig["entry"].to_numpy(float)
    sl = sig["stop_loss"].to_numpy(float)
    tp = sig["take_profit"].to_numpy(float)
    states = sig["signal"].to_numpy()

    y = []
    kept = []
    for i in positions:
        i = int(i)
        side = _side_of(states[i])
        e, s, t = entry[i], sl[i], tp[i]
        if not all(np.isfinite(v) for v in (e, s, t)) or abs(e - s) <= 0:
            continue
        outcome = None
        last_j = min(i + horizon, n - 1)
        for j in range(i + 1, last_j + 1):
            if side > 0:
                if high[j] >= t:
                    outcome = 1
                    break
                if low[j] <= s:
                    outcome = 0
                    break
            else:
                if low[j] <= t:
                    outcome = 1
                    break
                if high[j] >= s:
                    outcome = 0
                    break
        if outcome is None:
            move = (close[last_j] - e) * side
            outcome = 1 if move > floor_mult * max(atr[i], 1e-9) else 0
        y.append(outcome)
        kept.append(i)
    return np.asarray(kept, int), np.asarray(y, int)


class LogisticModel:
    """L2 logistic regression, standardized inputs, full-batch GD."""

    def __init__(self, l2: float = 0.5, lr: float = 0.3, iters: int = 800):
        self.l2, self.lr, self.iters = l2, lr, iters
        self.w = self.b = self.mu = self.sd = None

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = (X - self.mu) / self.sd
        n = len(Z)
        self.w = np.zeros(Z.shape[1])
        pm = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.b = float(np.log(pm / (1 - pm)))
        prev_loss = np.inf
        for _ in range(self.iters):
            p = self._sig(Z @ self.w + self.b)
            self.w -= self.lr * (Z.T @ (p - y) / n + self.l2 * self.w / n)
            self.b -= self.lr * float(np.mean(p - y))
            loss = -float(np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
            if abs(prev_loss - loss) < 1e-7:
                break
            prev_loss = loss
        return self

    def predict_proba(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return self._sig(Z @ self.w + self.b)


def _fit_on(df, sig, base, positions, labels, cfg):
    qcfg = cfg.get("quant", {})
    X = np.stack([
        make_event_row(base, int(p), _side_of(sig["signal"].iat[int(p)]),
                       sig["buy_score"].iat[int(p)], sig["sell_score"].iat[int(p)])
        for p in positions
    ])
    model = LogisticModel(l2=float(qcfg.get("l2", 0.5)))
    model.fit(X, np.asarray(labels))
    return model


def _training_frame(pair: str | None, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Merge live candles with the local store so training data keeps growing."""
    train_days = int(cfg.get("quant", {}).get("train_days", 45))
    if pair:
        store = load_candles(pair, cfg["timeframe"])
        if store is not None and len(store):
            full = pd.concat([store[~store.index.isin(df.index)], df])
            full.index = pd.to_datetime(pd.Index(full.index), utc=True).tz_convert(df.index.tz)
            full = full.sort_index()
            full = full[~full.index.duplicated(keep="last")]
            cutoff = full.index[-1] - pd.Timedelta(days=train_days)
            full = full[full.index >= cutoff]
            if len(full) >= len(df):
                return full
    return df


def prepare_events(df: pd.DataFrame, cfg: dict, pair: str | None = None):
    """Build signal series, base features, entry events and labels."""
    full = _training_frame(pair, df, cfg)
    sig = build_signal_series(full, cfg)
    base = compute_base_features(full, cfg)
    raw_positions = get_entry_positions(sig)
    positions, labels = label_events(full, sig, raw_positions, cfg)
    return sig, base, positions, labels


def walk_forward(df: pd.DataFrame, cfg: dict, pair: str | None = None) -> dict:
    """Expanding-window OOS probabilities per entry event (purged + embargoed)."""
    qcfg = cfg.get("quant", {})
    horizon = int(qcfg.get("horizon_bars", 60))
    embargo = int(qcfg.get("embargo_bars", 30))
    folds = int(qcfg.get("walk_forward_folds", 4))
    min_train = int(qcfg.get("min_train_events", 60))

    sig, base, positions, labels = prepare_events(df, cfg, pair)
    m = len(positions)
    probs = np.full(m, np.nan)
    base_rate = float(labels.mean()) if m else 0.5
    folds_used = 0

    if m >= min_train * 2:
        start = max(min_train, m // 4)
        edges = np.linspace(start, m, folds + 1).astype(int)
        for k in range(len(edges) - 1):
            lo, hi = int(edges[k]), int(edges[k + 1])
            cutoff_bar = int(positions[lo]) - horizon - embargo
            train_sel = positions[:lo] <= cutoff_bar
            if train_sel.sum() < min_train:
                continue
            model = _fit_on(df, sig, base, positions[:lo][train_sel], labels[:lo][train_sel], cfg)
            X_test = np.stack([
                make_event_row(base, int(p), _side_of(sig["signal"].iat[int(p)]),
                               sig["buy_score"].iat[int(p)], sig["sell_score"].iat[int(p)])
                for p in positions[lo:hi]
            ])
            probs[lo:hi] = model.predict_proba(X_test)
            folds_used += 1

    probs[np.isnan(probs)] = base_rate
    return {
        "sig": sig, "base": base, "positions": positions,
        "labels": labels, "probs": probs, "base_rate": base_rate,
        "folds_used": folds_used, "n_events": m,
    }


def threshold_table(labels: np.ndarray, probs: np.ndarray) -> list[dict]:
    out = []
    n = len(labels)
    for thr in (0.50, 0.52, 0.55, 0.58, 0.60, 0.65):
        sel = probs >= thr
        passed = int(sel.sum())
        wins = int(labels[sel].sum()) if passed else 0
        out.append({
            "threshold": thr, "trades": passed,
            "coverage_pct": round(passed / n * 100, 1) if n else 0.0,
            "precision_pct": round(wins / passed * 100, 1) if passed else 0.0,
        })
    return out


def gate_indices(wf: dict, threshold: float) -> np.ndarray:
    return wf["positions"][wf["probs"] >= threshold]


def suggested_risk_pct(p_win: float, cfg: dict) -> float | None:
    qcfg = cfg.get("quant", {})
    thr = float(qcfg.get("min_probability", 0.55))
    base = float(qcfg.get("base_risk_pct", 1.0))
    if p_win is None or p_win < thr:
        return None
    scale = 1.0 + (p_win - thr) / max(1.0 - thr, 1e-9)
    return round(base * min(scale, 2.0), 2)


def assess_signal(pair: str, df: pd.DataFrame, result: dict, cfg: dict) -> dict | None:
    """Score the CURRENT setup with a cached, fully-resolved-history model."""
    qcfg = cfg.get("quant", {})
    if not qcfg.get("enabled", True):
        return None

    now = time.time()
    retrain_secs = float(qcfg.get("retrain_hours", 6)) * 3600
    last_ts = df.index[-1]
    firm = result.get("signal") in FIRM

    c = _cache.get(pair)
    stale = (
        c is None
        or now - c["epoch"] > retrain_secs
        or (firm and c.get("last_ts") != last_ts)
    )
    if stale:
        try:
            sig, base, positions, labels = prepare_events(df, cfg, pair)
            resolvable = positions <= len(base) - 1 - int(qcfg.get("horizon_bars", 60))
            pos_r, lab_r = positions[resolvable], labels[resolvable]
            if len(pos_r) >= int(qcfg.get("min_train_events", 60)):
                model = _fit_on(base, sig, base, pos_r, lab_r, cfg)
                br = float(lab_r.mean())
            else:
                model, br = None, float(labels.mean()) if len(labels) else 0.5
            c = {
                "model": model, "base": base, "epoch": now,
                "last_ts": last_ts, "events": int(len(pos_r)),
                "base_rate": br,
                "trained_at": pd.Timestamp.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
            }
            _cache[pair] = c
        except Exception:
            traceback.print_exc(file=sys.stderr)
            if c is None:
                return {"p_win": None, "error": "model unavailable"}
            c["epoch"] = now
            c["last_ts"] = last_ts

    if c is None or c["model"] is None or not firm:
        return None

    try:
        base = c["base"]
        try:
            i = base.index.get_loc(last_ts)
            if not isinstance(i, (int, np.integer)):
                i = len(base) - 1
        except KeyError:
            i = len(base) - 1
        x = make_event_row(
            base, int(i),
            1.0 if "BUY" in result["signal"] else -1.0,
            result.get("details", {}).get("buy_score", 0),
            result.get("details", {}).get("sell_score", 0),
        ).reshape(1, -1)
        p = float(c["model"].predict_proba(x)[0])
        return {"p_win": p, "base_rate": c["base_rate"],
                "events": c["events"], "trained_at": c["trained_at"]}
    except Exception:
        return {"p_win": None, "error": "scoring failed"}
