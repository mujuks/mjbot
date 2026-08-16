# MJBot - Gold Trading Signal Bot

A Python bot that generates automated BUY/SELL signals for gold (GC=F) with entry, stop loss, and take profit levels. Combines supply/demand zones, liquidity sweeps, momentum, volume, and MACD/RSI into a single confluence engine, then delivers signals via Telegram and webhook.

## Features

- **Supply/Demand zones** - detects unfilled supply and demand areas with freshness scoring
- **Liquidity sweeps** - catches stop hunts of recent pivot highs/lows
- **Market mover** - trend + strong candle momentum detection
- **Volume confirmation** - volume spike filter
- **MACD + RSI** - momentum confluence and overbought/oversold filters
- **Entry / SL / TP** - every signal includes a stop loss (1 ATR past the structure) and 2R take profit
- **Telegram alerts** - instant notifications on new signals
- **Webhook publisher** - POST signal JSON to any endpoint
- **Backtester** - SL/TP-aware historical performance testing

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

The bot checks gold prices every `interval_seconds` (default 5 min) and alerts on new signals only (no repeat spam).

## Backtest

```bash
python backtest.py 365          # 365 days of 1h data, all configured pairs
python backtest.py 90 GC=F      # custom lookback and symbol
```

Example output:

```
GC=F (1h, 365d):
  Balance: 10000.0 -> 14897.47 (+48.97%)
  Trades: 397 | Wins: 182 | Losses: 215 | Win rate: 45.8%
  Avg win: +0.98% | Avg loss: -0.63% | PF: 1.31
  SL hits: 177 | TP hits: 121 | Max drawdown: 9.53%
```

## TradingView

`gold_signals.pine` is a Pine Script v5 replica of the strategy. Open a **GC=F** 1h chart in TradingView, open the Pine Editor, paste the script, and it will plot the same signals with native alerts (entry/SL/TP included in the alert message).

## Configuration

| Key | Default | Description |
|---|---|---|
| `pairs` | `["GC=F"]` | Symbols to watch (yfinance format) |
| `timeframe` | `1h` | Candlestick interval |
| `interval_seconds` | `300` | How often to check for signals |
| `ma_fast` / `ma_slow` | `10` / `30` | Momentum trend EMAs |
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

## Project Structure

```
alerts.py          Telegram + webhook alerting
backtest.py        SL/TP-aware historical backtester
bot.py             Main loop (signal generation + alerts)
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
