# MJBot - Gold Trading Signal Bot

A Python bot that generates automated BUY/SELL signals for gold (GC=F) with entry, stop loss, and take profit levels. It scans **15-minute scalping entries** filtered by a **1-hour trend bias** and only trades during the **London/NY sessions** (the highest-activity windows for gold), then delivers signals via Telegram and webhook.

## Features

- **Multi-timeframe confluence** - 15m entries with a 1h EMA trend bias filter (never trades against the higher-timeframe trend)
- **Supply/Demand zones** - detects unfilled supply and demand areas with freshness scoring
- **Liquidity sweeps** - catches stop hunts of recent pivot highs/lows
- **Market mover** - trend + strong candle momentum detection
- **Volume confirmation** - volume spike filter
- **MACD + RSI** - momentum confluence and overbought/oversold filters
- **Session filter** - only trades Monday-Friday during London/NY hours (default 07:00-21:00 UTC); avoids Asian session, rollover, and market closings
- **News blackout windows** - optional configurable UTC windows (e.g. NFP/CPI/FOMC) where no signals fire
- **Entry / SL / TP** - every signal includes a stop loss (1 ATR past the structure) and 2R take profit
- **Telegram alerts** - instant notifications on new signals + an hourly report digest
- **Webhook publisher** - POST signal JSON to any endpoint
- **Backtester** - SL/TP-aware historical performance testing, bias- and session-accurate

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`

## Quick Start

1. Copy the config template and fill in your settings:

   ```bash
   cp config.example.json config.json
   ```

2. (Optional) Set up Telegram:
   - Create a bot with [@BotFather](https://t.me/BotFather) to get a token
   - Message your bot once, then get your chat ID from `getUpdates`
   - Add both to `config.json` under `telegram`

3. Run the bot:

   ```bash
   python bot.py
   ```

The bot checks gold every `interval_seconds` (default 15 min = one fresh 15m candle), alerts on new signals only (no repeat spam), and sends a full **hourly report** to Telegram when `hourly_digest` is enabled.

## Backtest

```bash
python backtest.py 60           # 60 days of 15m data (max for intraday), all configured pairs
python backtest.py 30 GC=F      # custom lookback and symbol
```

Example output (60 days, 15m entries + 1h bias + session filter):

```
GC=F (15m, 60d):
  Balance: 10000.0 -> 10857.28 (+8.57%)
  Trades: 104 | Wins: 42 | Losses: 62 | Win rate: 40.4%
  Avg win: +0.80% | Avg loss: -0.40% | PF: 1.34
  SL hits: 57 | TP hits: 40 | Max drawdown: 5.05%
```

Note: yfinance caps intraday data at 60 days, so `lookback_days` above 60 is clamped to 60 for minute timeframes.

## TradingView

`gold_signals.pine` is a Pine Script v5 replica of the strategy. Open a **GC=F 15m** chart in TradingView, open the Pine Editor, paste the script, and it will plot the same signals (including the 1h bias filter and session filter) with native alerts (entry/SL/TP included in the alert message).

## Configuration

| Key | Default | Description |
|---|---|---|
| `pairs` | `["GC=F"]` | Symbols to watch (yfinance format) |
| `timeframe` | `15m` | Entry candlestick interval (scalping) |
| `bias_timeframe` | `1h` | Higher-timeframe trend filter |
| `bias_enabled` | `true` | Block signals against the bias trend |
| `interval_seconds` | `900` | How often to check for signals |
| `lookback_days` | `30` | Data window (minute TFs capped at 60 by yfinance) |
| `ma_fast` / `ma_slow` | `10` / `30` | Momentum + bias trend EMAs |
| `rsi_period` | `14` | RSI lookback |
| `macd_*` | `12/26/9` | MACD parameters |
| `atr_period` | `14` | ATR for SL/TP and zone sizing |
| `zone_*` | - | Supply/demand zone detection |
| `sweep_*` | - | Liquidity sweep detection |
| `volume_*` | - | Volume spike filter |
| `signal_threshold` | `3` | Min confluence score for a signal |
| `strong_threshold` | `5` | Score for STRONG_BUY/STRONG_SELL |
| `risk_reward` | `2.0` | Take profit / risk ratio |
| `sl_atr_buffer` | `1.0` | Stop loss distance in ATRs |
| `sessions` | `07:00-21:00 UTC, Mon-Fri` | Active trading windows |
| `blackout_windows` | `[]` | UTC news blackouts, e.g. `[{"start": "2026-09-04T12:00", "end": "2026-09-04T13:00"}]` |
| `hourly_digest` | `true` | Send an hourly Telegram report |

## Project Structure

```
alerts.py          Telegram + webhook alerting
backtest.py        SL/TP-aware historical backtester
bot.py             Main loop (signal generation + hourly digest)
config.example.json  Config template (copy to config.json)
data_fetcher.py    yfinance data downloader
liquidity.py       Liquidity sweep detector
strategy.py        Confluence engine + vectorized signal series
zones.py           Supply/demand zone detector
gold_signals.pine  TradingView Pine Script replica
```

## Security

- `config.json` contains your Telegram token and is **gitignored** - never commit or share it
- Keep your bot token private; anyone with it can control your bot

## Disclaimer

Trading involves risk. This bot is for educational purposes and is not financial advice. Always test on a demo account and never risk money you cannot afford to lose.
