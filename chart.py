import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import numpy as np
import pandas as pd

from zones import _atr


def generate_signal_chart(df: pd.DataFrame, cfg: dict, result: dict, pair: str) -> bytes | None:
    if df is None or df.empty or len(df) < 60:
        return None

    n_bars = min(120, len(df))
    chart_df = df.iloc[-n_bars:].copy()

    if not isinstance(chart_df.index, pd.DatetimeIndex):
        return None

    close = chart_df["Close"]
    atr_val = _atr(chart_df, cfg.get("atr_period", 14))

    fast = close.ewm(span=cfg.get("ma_fast", 10), adjust=False).mean()
    slow = close.ewm(span=cfg.get("ma_slow", 30), adjust=False).mean()

    signal = result.get("signal", "NONE")
    entry = result.get("entry", 0.0)
    sl = result.get("stop_loss", 0.0)
    tp = result.get("take_profit", 0.0)
    price = result.get("price", 0.0)
    bias_label = result.get("details", {}).get("bias", "flat")
    score = result.get("score", 0)

    add_plots = []
    add_plots.append(mpf.make_addplot(fast, color="#2196F3", width=1.2, label="EMA Fast"))
    add_plots.append(mpf.make_addplot(slow, color="#FF9800", width=1.2, label="EMA Slow"))

    buy_markers = pd.Series(np.nan, index=chart_df.index)
    sell_markers = pd.Series(np.nan, index=chart_df.index)

    if signal in ("BUY", "STRONG_BUY"):
        buy_markers.iloc[-1] = chart_df["Low"].iloc[-1] - 0.3 * atr_val.iloc[-1]
    elif signal in ("SELL", "STRONG_SELL"):
        sell_markers.iloc[-1] = chart_df["High"].iloc[-1] + 0.3 * atr_val.iloc[-1]

    if not buy_markers.isna().all():
        add_plots.append(mpf.make_addplot(buy_markers, type="scatter", markersize=120,
                          marker="^", color="#00C853"))
    if not sell_markers.isna().all():
        add_plots.append(mpf.make_addplot(sell_markers, type="scatter", markersize=120,
                          marker="v", color="#FF1744"))

    fig, axes = mpf.plot(
        chart_df,
        type="candle",
        style="charles",
        addplot=add_plots,
        volume=True,
        returnfig=True,
        figsize=(10, 6),
        tight_layout=True,
    )

    ax_main = axes[0]
    ax_vol = axes[2]

    ax_main.axhline(y=entry, color="#2196F3", linestyle="--", linewidth=1, alpha=0.8, label=f"Entry {entry:.2f}")
    ax_main.axhline(y=sl, color="#FF1744", linestyle="--", linewidth=1, alpha=0.8, label=f"SL {sl:.2f}")
    ax_main.axhline(y=tp, color="#00C853", linestyle="--", linewidth=1, alpha=0.8, label=f"TP {tp:.2f}")

    signal_colors = {
        "BUY": "#00C853", "STRONG_BUY": "#00E676",
        "SELL": "#FF1744", "STRONG_SELL": "#FF5252",
    }
    sig_color = signal_colors.get(signal, "#9E9E9E")
    sig_text = signal.replace("_", " ") if signal != "NONE" else "NO SIGNAL"

    bias_colors = {"bullish": "#00C853", "bearish": "#FF1744", "flat": "#9E9E9E"}
    bias_color = bias_colors.get(bias_label, "#9E9E9E")

    ax_main.legend(loc="upper left", fontsize=8, framealpha=0.7)

    title = f"{pair} | {sig_text} | Bias: {bias_label.upper()} | Score: {score}"
    ax_main.set_title(title, fontsize=12, fontweight="bold", color=sig_color, pad=10)

    subtitle = f"Entry: {entry:.2f}  SL: {sl:.2f}  TP: {tp:.2f}  Price: {price:.2f}"
    ax_main.text(0.5, 1.01, subtitle, transform=ax_main.transAxes,
                 ha="center", fontsize=9, color="#BDBDBD", family="monospace")

    fig.patch.set_facecolor("#1A1A2E")
    for ax in [ax_main, ax_vol]:
        ax.set_facecolor("#16213E")
        ax.tick_params(colors="#BDBDBD")
        ax.xaxis.label.set_color("#BDBDBD")
        ax.yaxis.label.set_color("#BDBDBD")

    ax_main.grid(True, alpha=0.15, color="#BDBDBD")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
