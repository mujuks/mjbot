import sys

import pandas as pd

from alerts import load_config
from data_fetcher import fetch_forex_data
from strategy import bias_series, build_signal_series, trading_mask


def backtest(pair: str, cfg: dict, lookback_days: int = 365, initial_balance: float = 10000.0) -> dict:
    tf = cfg["timeframe"]
    if tf.endswith("m"):
        lookback_days = min(lookback_days, 60)
    df = fetch_forex_data(pair, tf, lookback_days)
    if df is None or df.empty or len(df) < 60:
        return {"pair": pair, "error": "Not enough data"}

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

    closes = df["Close"].to_numpy(float)

    balance = initial_balance
    position = None
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trades = []
    equity_curve = []

    for i in range(len(df)):
        signal = sig["signal"].iloc[i]
        price = float(closes[i])

        if position == "long":
            if price <= stop_loss:
                pnl = (stop_loss - entry_price) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "long", "pnl_pct": pnl * 100, "close": "SL"})
                position = None
            elif price >= take_profit:
                pnl = (take_profit - entry_price) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "long", "pnl_pct": pnl * 100, "close": "TP"})
                position = None
            elif signal in ("SELL", "STRONG_SELL"):
                pnl = (price - entry_price) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "long", "pnl_pct": pnl * 100, "close": "signal"})
                position = None
        elif position == "short":
            if price >= stop_loss:
                pnl = (entry_price - stop_loss) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "short", "pnl_pct": pnl * 100, "close": "SL"})
                position = None
            elif price <= take_profit:
                pnl = (entry_price - take_profit) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "short", "pnl_pct": pnl * 100, "close": "TP"})
                position = None
            elif signal in ("BUY", "STRONG_BUY"):
                pnl = (entry_price - price) / entry_price
                balance *= 1 + pnl
                trades.append({"type": "short", "pnl_pct": pnl * 100, "close": "signal"})
                position = None

        if position is None and signal in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL"):
            position = "long" if signal in ("BUY", "STRONG_BUY") else "short"
            entry_price = float(sig["entry"].iloc[i])
            stop_loss = float(sig["stop_loss"].iloc[i])
            take_profit = float(sig["take_profit"].iloc[i])

        equity_curve.append(balance)

    if position is not None:
        pnl = (
            (price - entry_price) / entry_price
            if position == "long"
            else (entry_price - price) / entry_price
        )
        balance *= 1 + pnl
        trades.append({"type": position, "pnl_pct": pnl * 100, "close": "open"})

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
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
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "profit_factor": round(sum(t["pnl_pct"] for t in wins) / abs(sum(t["pnl_pct"] for t in losses)), 2)
        if losses else float("inf"),
        "sl_hits": sl_count,
        "tp_hits": tp_count,
    }


def main() -> None:
    cfg = load_config()
    lookback = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    pairs = sys.argv[2:] if len(sys.argv) > 2 else cfg["pairs"]

    for pair in pairs:
        try:
            result = backtest(pair, cfg, lookback)
            if "error" in result:
                print(f"[{pair}] {result['error']}")
                continue
            print(
                f"{result['pair']} ({result['timeframe']}, {result['lookback_days']}d):\n"
                f"  Balance: {result['initial_balance']} -> {result['final_balance']} "
                f"({result['return_pct']:+.2f}%)\n"
                f"  Trades: {result['trades']} | Wins: {result['wins']} | Losses: {result['losses']} "
                f"| Win rate: {result['win_rate']}%\n"
                f"  Avg win: {result['avg_win_pct']:+.2f}% | Avg loss: {result['avg_loss_pct']:+.2f}% "
                f"| PF: {result['profit_factor']}\n"
                f"  SL hits: {result['sl_hits']} | TP hits: {result['tp_hits']} "
                f"| Max drawdown: {result['max_drawdown_pct']}%\n"
            )
        except Exception as e:
            print(f"[{pair}] Backtest failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
