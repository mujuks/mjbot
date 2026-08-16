import pandas as pd


def detect_liquidity_sweep(df: pd.DataFrame, cfg: dict) -> dict:
    left = cfg.get("sweep_pivot", 3)
    window = cfg.get("sweep_window", 3)

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)

    piv_high = []
    piv_low = []
    for i in range(left, n - left):
        if highs[i] == max(highs[i - left:i + left + 1]):
            piv_high.append((i, highs[i]))
        if lows[i] == min(lows[i - left:i + left + 1]):
            piv_low.append((i, lows[i]))

    result = {"bullish": False, "bearish": False, "level": None, "bars_ago": None}

    for i in range(max(1, n - window), n):
        for pi, pl in piv_low:
            if pi < i and lows[i] < pl and closes[i] > pl:
                result.update({"bullish": True, "level": float(pl), "bars_ago": n - 1 - i})
                break
        for pi, ph in piv_high:
            if pi < i and highs[i] > ph and closes[i] < ph:
                result.update({"bearish": True, "level": float(ph), "bars_ago": n - 1 - i})
                break

    return result
