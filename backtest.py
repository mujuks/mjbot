import sys

import numpy as np
import pandas as pd

from alerts import load_config
from data_fetcher import fetch_forex_data
from strategy import bias_series, build_signal_series, trading_mask
import quant as quant_mod


def backtest(pair: str, cfg: dict, lookback_days: int = 365, initial_balance: float = 10000.0,
             df: pd.DataFrame = None, gate_positions: np.ndarray = None,
             breakeven_at_r: float | None = None) -> dict:
    tf = cfg["timeframe"]
    if tf.endswith("m"):
        lookback_days = min(lookback_days, 60)
    if df is None:
        df = fetch_forex_data(pair, tf, lookback_days)
    if df is None or df.empty or len(df) < 60:
        return {"pair": pair, "error": "Not enough data"}

    cost_points = float(cfg.get("quant", {}).get("cost_points", 0.35))

    bias = None
    if cfg.get("bias_enabled", True) and cfg.get("bias_timeframe"):
        bias_df = fetch_forex_data(pair, cfg["bias_timeframe"], lookback_days)
        if bias_df is not None and not bias_df.empty:
            b = pd.Series(bias_series(bias_df, cfg), index=bias_df.index)
            b = b.shift(1)
            bias = b.reindex(df.index, method="ffill").fillna(0).to_numpy(int)

    sig = build_signal_series(df, cfg, bias=bias)
    mask = trading_mask(df.index, cfg)
    sig.loc[~mask, "signal"] = "NONE"

    allow = np.ones(len(df), bool)
    if gate_positions is not None:
        allow[:] = False
        allow[np.asarray(gate_positions, int)] = True

    closes = df["Close"].to_numpy(float)

    balance = initial_balance
    position = None
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    risk_dist = 0.0
    be_armed = False
    trades = []
    equity_curve = []

    for i in range(len(df)):
        signal = sig["signal"].iloc[i]
        price = float(closes[i])

        if position is not None and breakeven_at_r and risk_dist > 0:
            if position == "long" and price >= entry_price + risk_dist * breakeven_at_r and stop_loss < entry_price:
                stop_loss = entry_price
                be_armed = True
            elif position == "short" and price <= entry_price - risk_dist * breakeven_at_r and stop_loss > entry_price:
                stop_loss = entry_price
                be_armed = True

        if position == "long":
            if price <= stop_loss:
                pnl = (stop_loss - entry_price - 2 * cost_points) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "long", "pnl_pct": pnl * 100, "close": "BE" if be_armed else "SL"})
                position = None
                be_armed = False
            elif price >= take_profit:
                pnl = (take_profit - entry_price - 2 * cost_points) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "long", "pnl_pct": pnl * 100, "close": "TP"})
                position = None
            elif signal in ("SELL", "STRONG_SELL"):
                pnl = (price - entry_price - 2 * cost_points) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "long", "pnl_pct": pnl * 100, "close": "signal"})
                position = None
        elif position == "short":
            if price >= stop_loss:
                pnl = (entry_price - stop_loss - 2 * cost_points) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "short", "pnl_pct": pnl * 100, "close": "BE" if be_armed else "SL"})
                position = None
                be_armed = False
            elif price <= take_profit:
                pnl = (entry_price - take_profit - 2 * cost_points) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "short", "pnl_pct": pnl * 100, "close": "TP"})
                position = None
            elif signal in ("BUY", "STRONG_BUY"):
                pnl = (entry_price - price - 2 * cost_points) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "short", "pnl_pct": pnl * 100, "close": "signal"})
                position = None

        if position is None and allow[i] and signal in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL"):
            position = "long" if signal in ("BUY", "STRONG_BUY") else "short"
            entry_price = float(sig["entry"].iloc[i])
            stop_loss = float(sig["stop_loss"].iloc[i])
            take_profit = float(sig["take_profit"].iloc[i])
            risk_dist = abs(entry_price - stop_loss)
            be_armed = False

        equity_curve.append(balance)

    if position is not None:
        pnl = (
            (price - entry_price - 2 * cost_points) / entry_price
            if position == "long"
            else (entry_price - price - 2 * cost_points) / entry_price
        )
        balance *= 1 + pnl
        trades.append({"type": position, "pnl_pct": pnl * 100, "close": "open"})

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    be_count = sum(1 for t in trades if t["close"] == "BE")
    decided = [t for t in trades if t["close"] != "BE"]
    wins_d = [t for t in decided if t["pnl_pct"] > 0]
    losses_d = [t for t in decided if t["pnl_pct"] <= 0]
    avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0.0

    peak = -1e18
    max_drawdown = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100
        max_drawdown = max(max_drawdown, dd)

    sl_count = sum(1 for t in trades if t["close"] == "SL")
    tp_count = sum(1 for t in trades if t["close"] == "TP")

    return {
        "pair": pair,
        "timeframe": cfg["timeframe"],
        "lookback_days": lookback_days,
        "initial_balance": initial_balance,
        "final_balance": round(balance, 2),
        "return_pct": round((balance / initial_balance - 1) * 100, 2),
        "trades": len(trades),
        "wins": len(wins_d),
        "losses": len(losses_d),
        "be_exits": be_count,
        "win_rate": round(len(wins_d) / len(decided) * 100, 1) if decided else 0.0,
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "profit_factor": round(sum(t["pnl_pct"] for t in wins) / abs(sum(t["pnl_pct"] for t in losses)), 2)
        if losses else float("inf"),
        "sl_hits": sl_count,
        "tp_hits": tp_count,
    }


def _print_result(tag: str, r: dict):
    print(
        f"{tag}:\n"
        f"  Balance: {r['initial_balance']} -> {r['final_balance']} "
        f"({r['return_pct']:+.2f}%)\n"
        f"  Trades: {r['trades']} | Wins: {r['wins']} | Losses: {r['losses']} "
        f"| BE exits: {r.get('be_exits', 0)} | Win rate: {r['win_rate']}%\n"
        f"  Avg win: {r['avg_win_pct']:+.2f}% | Avg loss: {r['avg_loss_pct']:+.2f}% "
        f"| PF: {r['profit_factor']}\n"
        f"  SL hits: {r['sl_hits']} | TP hits: {r['tp_hits']} "
        f"| Max drawdown: {r['max_drawdown_pct']}%\n"
    )


def main() -> None:
    cfg = load_config()
    lookback = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    pairs = sys.argv[2:] if len(sys.argv) > 2 else cfg["pairs"]

    for pair in pairs:
        try:
            tf = cfg["timeframe"]
            days = min(lookback, 60) if tf.endswith("m") else lookback
            df = fetch_forex_data(pair, tf, days)

            base = backtest(pair, cfg, days, df=df)
            if "error" in base:
                print(f"[{pair}] {base['error']}")
                continue
            print(f"{pair} ({tf}, {days}d, costs={cfg.get('quant', {}).get('cost_points', 0.35)} pts/side)")
            _print_result("BASELINE (all signals)", base)
            base_be = backtest(pair, cfg, days, df=df, breakeven_at_r=1.0)
            _print_result("BASELINE + breakeven @1R", base_be)

            qcfg = cfg.get("quant", {})
            if not qcfg.get("enabled", True):
                continue
            wf = quant_mod.walk_forward(df, cfg, pair)
            print(
                f"META-MODEL: {wf['n_events']} entry events | base win rate "
                f"{wf['base_rate'] * 100:.1f}% | folds used: {wf['folds_used']}"
            )
            table = quant_mod.threshold_table(wf["labels"], wf["probs"])
            print("  Threshold | Trades | Coverage | Precision (OOS)")
            for row in table:
                print(f"     {row['threshold']:.2f}    |  {row['trades']:4d}  |  {row['coverage_pct']:5.1f}%  |  {row['precision_pct']:.1f}%")

            thr = float(qcfg.get("min_probability", 0.55))
            gated_pos = quant_mod.gate_indices(wf, thr)
            gated = backtest(pair, cfg, days, df=df, gate_positions=gated_pos)
            _print_result(f"GATED (p_win >= {thr})", gated)
            gated_be = backtest(pair, cfg, days, df=df, gate_positions=gated_pos, breakeven_at_r=1.0)
            _print_result(f"GATED + breakeven @1R", gated_be)
        except Exception as e:
            print(f"[{pair}] Backtest failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
