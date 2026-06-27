# QUICKSTART — Trade Tonight

> 5-minute setup to go from clone to live paper trading.

---

## Step 1: Install

```bash
cd pumpfun-agent
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Configure secrets

```bash
cp config/config.yaml.example config/config.yaml
cp .env.example .env
```

Edit `.env` — fill in **at minimum**:

```bash
# REQUIRED for any trading:
SOLANA_WALLET_SEED="your 12 or 24 word seed phrase"
HELIUS_API_KEY="your helius key from https://helius.dev"

# REQUIRED for notifications:
TELEGRAM_BOT_TOKEN="from @BotFather"
TELEGRAM_CHAT_ID="your chat id"

# OPTIONAL but recommended:
TWITTER_BEARER_TOKEN=""          # enables sentiment analysis
BIRDEYE_API_KEY=""               # enables backtesting with real OHLCV
```

Edit `config/config.yaml` — set **at minimum**:

```yaml
trading:
  mode: "paper"           # START HERE. Change to "live" only after testing.

chains:
  solana:
    rpc_endpoints:
      - "https://mainnet.helius-rpc.com/?api-key=YOUR_HELIUS_KEY"
    ws_endpoint: "wss://mainnet.helius-rpc.com/?api-key=YOUR_HELIUS_KEY"
    allocated_capital_sol: 1.0     # how much SOL the bot thinks it has
```

## Step 3: Run

```bash
python orchestrator.py
```

You should see within 10 seconds:
- `orchestrator.running` — the bot started
- `sniping.new_launch_detected` — it's detecting real pump.fun launches
- `regime.assessed` — market regime classified
- `rugpull_blocked` — anti-rug gate working

## Step 4: Monitor

- **Logs**: stdout (structured JSON) + `data/agent.log`
- **Dashboard** (if enabled in config): http://localhost:8080
- **Telegram**: send `/status` to your bot

## Step 5: Go live (AFTER paper testing)

1. Fund your dedicated wallet with **0.1-0.5 SOL** (start small!)
2. Change `trading.mode: "live"` in config
3. Re-run `python orchestrator.py`
4. Watch the first real trade closely

---

## What each module does (cheat sheet)

| You see in logs | What it means |
|---|---|
| `sniping.new_launch_detected` | New pump.fun token found via log parsing |
| `sniping.rugpull_blocked` | Token failed anti-rug checks (good — protecting you) |
| `executor.paper_buy` | Simulated buy in paper mode |
| `executor.buy_failed` | Real buy failed — check the error |
| `orchestrator.soft_rug_exit` | ML model predicts a gradual dump — exiting |
| `orchestrator.peak_hype_exit` | Crowd is euphoric — selling into the top |
| `orchestrator.crowd_fade_signal` | Crowd is overcrowded — fading the other way |
| `omega.toxic_flow_detected` | Informed sellers distributing into retail buys |
| `omega.smart_money_divergence` | Price up but flow negative = distribution |
| `regime.assessed label=risk_off` | Market is dead/choppy — new buys paused |
| `latency.trace op=snipe total_ms=XX` | Your sniping latency (target: <500ms) |

## Kill switch (emergency stop)

Any of these stops the bot within 2 seconds:
- **File**: `touch data/KILL_SWITCH`
- **Telegram**: send `/kill` to your bot
- **Dashboard**: click the KILL button at http://localhost:8080
- **Ctrl+C**: graceful shutdown

## Troubleshooting

| Problem | Fix |
|---|---|
| `ConfigError: config file not found` | `cp config/config.yaml.example config/config.yaml` |
| `solana.wallet.seed_missing` | Set `SOLANA_WALLET_SEED` in `.env` |
| `notifier.disabled_missing_creds` | Set `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `.env` |
| `No reachable Solana RPC` | Check `HELIUS_API_KEY` and the RPC URL in config |
| Bot detects launches but never buys | Paper mode = simulated only. Or: anti-rug is blocking everything (lower thresholds in config). Or: regime is risk_off. |
