# Pump.fun Autonomous Trading Agent

A production-grade, multi-strategy, multi-chain autonomous trading agent for **pump.fun** (Solana) and pump.fun-like launches on EVM chains (Base / Ethereum).

[![CI](https://github.com/YOUR_USERNAME/pumpfun-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/pumpfun-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> ⚠️ **DISCLAIMER**: Trading memecoins is extremely risky. This software can lose your entire capital. Use a dedicated wallet with funds you can afford to lose. Test thoroughly in paper mode before going live. The author is not responsible for any financial loss.

---

## ✨ Features

### Core Trading
| Module | Description |
|--------|-------------|
| **Multi-chain** | Solana (pump.fun native) + EVM (Base / Ethereum) |
| **Sniping** | Real pump.fun create-event parser → buy within ms |
| **Copy-trading** | Replicates trades from profitable wallets |
| **Momentum** | RSI + volume breakout detection on DexScreener |
| **Grid / Scalping** | Grid orders on established tokens |
| **Anti-rugpull** | Real data: Helius holders, DexScreener liquidity, mint/freeze authority, TokenSniffer honeypot, micro-buy test |

### Risk Management
| Module | Description |
|--------|-------------|
| **Fixed + Kelly sizing** | Configurable fixed SOL/ETH or Half-Kelly dynamic |
| **Daily loss cap** | Circuit breaker on drawdown threshold |
| **Smart exits** | Break-even, trailing stop, TP ladder (25% @ +50/100/200%), time decay |
| **Dev wallet tracker** | Instant exit if dev sells (WebSocket-monitored) |
| **MEV protection** | Jito multi-tx bundles (tip + swap atomic) |
| **Anti-sandwich** | Dynamic slippage based on trade size vs liquidity |

### Advanced Analytics (`analysis/`)
| Module | Description |
|--------|-------------|
| **OrderFlowAnalyzer** | Real-time buy/sell pressure, whale detection, velocity |
| **SocialGraphAnalyzer** | Wallet clustering, smart money detection, syndicate identification |
| **MevDetector** | Detects sandwich attacks on our executed txs |
| **LifecycleAnalyzer** | BIRTH/INFANT/ADOLESCENT/MATURE/MIGRATED stages + size multiplier |
| **LiquidityDepthAnalyzer** | Slippage curve for various trade sizes, max profitable size |
| **SentimentAnalyzer** | Twitter/X + Telegram mentions, hype score, influencer detection |
| **AlphaSignalGenerator** | Weighted ensemble of all 8 analyzers → single 0-100 score |
| **BondingCurveAnalyzer** | Real pump.fun reserves math, migration ETA estimation |
| **MigrationDetector** | Detects Raydium migration events (huge catalyst) |

### Operational
| Module | Description |
|--------|-------------|
| **Non-custodial** | Wallet keys never leave the local process |
| **Persistence** | SQLite: positions, trades, blacklist, daily PnL, config audit |
| **Telegram alerts** | Trade notifications + `/kill` + `/status` commands |
| **Dashboard** | FastAPI dashboard with **live parameter tuning** (48 tunables) |
| **24/7 live** | Asyncio-based, SIGINT/SIGTERM-aware graceful shutdown |
| **Hot-reload** | Change any parameter via dashboard without restart |

---

## 📁 Project Structure

```
pumpfun-agent/
├── analysis/                        # Advanced analytics (NEW)
│   ├── alpha_signal.py              # Composite ensemble of all analyzers
│   ├── lifecycle.py                 # BIRTH/INFANT/ADOLESCENT/MATURE/MIGRATED
│   ├── liquidity_depth.py           # Slippage curve + max position size
│   ├── mev_detector.py              # Sandwich attack detection
│   ├── order_flow.py                # Buy/sell pressure, whale tracking
│   ├── sentiment.py                 # Twitter/X + Telegram hype score
│   └── social_graph.py              # Wallet clustering, smart money
├── chains/
│   ├── base_chain.py
│   ├── solana_adapter.py
│   ├── evm_adapter.py
│   ├── jito_client.py
│   └── chain_factory.py
├── strategies/
│   ├── base_strategy.py
│   ├── sniping.py
│   ├── copy_trade.py
│   ├── momentum.py
│   ├── grid_scalping.py
│   └── anti_rugpull.py
├── risk/
│   ├── types.py
│   ├── risk_manager.py
│   ├── smart_exits.py
│   ├── pyramiding.py
│   └── backtester.py
├── execution/
│   └── executor.py
├── utils/
│   ├── config_loader.py             # + TUNABLE_SCHEMA + hot-update
│   ├── config_hot_reload.py         # Push live updates to running modules
│   ├── logger.py
│   ├── wallet_manager.py
│   ├── kill_switch.py
│   ├── notifier.py
│   ├── persistence.py               # SQLite + config audit log
│   ├── pumpfun_parser.py
│   ├── data_providers.py            # DexScreener + Helius + Birdeye + TokenSniffer
│   ├── dashboard.py                 # FastAPI + parameter editor
│   ├── bonding_curve.py
│   ├── token_scorer.py
│   ├── gas_optimizer.py
│   ├── dev_tracker.py
│   ├── migration_detector.py
│   └── anti_sandwich.py
├── config/
│   ├── config.yaml.example
│   └── config.yaml                  # (gitignored)
├── scripts/
│   ├── healthcheck.sh
│   ├── install_systemd.sh
│   └── push_to_github.sh            # One-command GitHub push
├── tests/
│   └── test_risk_manager.py
├── .github/
│   └── workflows/
│       └── ci.yml                   # Python 3.11 + 3.12 tests
├── data/                            # SQLite DB + logs (gitignored)
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
├── orchestrator.py                  # Main entrypoint
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- A Solana wallet with SOL (test with 0.1-0.5 SOL first)
- An EVM wallet with ETH/Base ETH (optional, if using EVM chains)
- A Telegram bot token (talk to [@BotFather](https://t.me/BotFather))
- Helius API key (https://helius.dev — free tier works)
- Birdeye API key (optional, only needed for backtesting — https://birdeye.so)

### 2. Install
```bash
cd pumpfun-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp config/config.yaml.example config/config.yaml
cp .env.example .env
```

Edit `.env` and fill in:
- `SOLANA_WALLET_SEED` — your dedicated trading wallet's seed phrase
- `HELIUS_API_KEY` — get from https://helius.dev
- `BIRDEYE_API_KEY` — only if you enable backtesting
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- Optional: `BASE_WALLET_PRIVKEY` / `ETH_WALLET_PRIVKEY`

Edit `config/config.yaml`:
- Set `trading.mode: "paper"` for first runs
- Set `monitoring.dashboard.enabled: true` to enable the dashboard
- Set `jito.enabled: true` for MEV-protected sniping
- Add target wallets to `strategies.copy_trade.target_wallets` if enabling copy-trade
- Disable strategies you don't want (`enabled: false`)

### 4. Run healthcheck
```bash
bash scripts/healthcheck.sh
```

### 5. Test in paper mode
```bash
python orchestrator.py
```

Monitor via:
- **Dashboard**: http://localhost:8080
- **Telegram**: send `/status` to your bot

### 6. Go live
- Fund the dedicated wallet with a small amount (e.g. 0.5 SOL)
- Set `trading.mode: "live"` in config
- Re-run `python orchestrator.py`

---

## 🛡️ Safety Mechanisms

| Mechanism | Trigger | Effect |
|-----------|---------|--------|
| File kill switch | Create file `data/KILL_SWITCH` | Agent halts within 2s |
| Telegram `/kill` | Send `/kill` to your bot | Instant halt |
| Dashboard KILL button | Click button at http://localhost:8080 | Instant halt |
| Daily loss cap | Daily PnL < -`daily_loss_cap_pct` | Auto-halt + Telegram alert |
| Per-token stop-loss | Position reaches `-stop_loss_pct` | Auto-close |
| Per-token take-profit | Position reaches `+take_profit_pct` | Auto-close |
| Max open positions | Exceeds `max_open_positions` | New buys blocked |
| Trade frequency cap | > `max_trade_frequency_per_minute` | New buys blocked |
| Token blacklist | Token failed rugpull check or hit SL | Blocked for `blacklist_duration_minutes` |
| Anti-rugpull gate | Holder concentration / mint authority / honeypot | Buy blocked + alert |

---

## 🧪 Backtesting with Real Data

The backtester fetches real OHLCV from Birdeye and replays signals on historical data. To enforce a quality gate:

```yaml
risk:
  backtesting:
    enabled: true
    min_required_sharpe: 0.8
    min_required_winrate: 0.40
    historical_data_days: 30
```

When enabled, the orchestrator will refuse to start a strategy whose 30-day backtest falls below these thresholds. You need `BIRDEYE_API_KEY` in your `.env`.

---

## 💾 Persistence (SQLite)

The agent persists the following to `data/agent.db`:

| Table | Content |
|-------|---------|
| `positions` | Currently open positions (restored on restart) |
| `trades` | Full trade history (used by Kelly estimator) |
| `blacklist` | Tokens blocked with expiry timestamps |
| `daily_pnl` | Per-day PnL snapshots (for daily loss cap recovery) |
| `agent_state` | Key/value store for misc state |

If the agent restarts mid-trade, open positions are restored and exits continue to be monitored. Daily PnL also carries over within the same UTC day.

---

## ⚡ MEV Protection (Jito)

Sniping on pump.fun is highly competitive — your pending buys can be frontrun by MEV bots watching the public mempool.

Jito bundles solve this:
- Atomic: the whole bundle lands or none of it does
- Priority inclusion via validator tips
- Endpoint rotation across 6 Jito block engines (NY/Frankfurt/Amsterdam/Tokyo/SLC)

```yaml
jito:
  enabled: true
  min_size_sol: 0.1          # only use Jito for buys >= 0.1 SOL
  tip_lamports: 0.001        # ~0.001 SOL tip
```

For buys below `min_size_sol`, the agent uses normal RPC submission (tip not worth the cost).

---

## 📊 Dashboard & Live Parameter Tuning

When `monitoring.dashboard.enabled: true`, the agent starts a FastAPI server at `http://localhost:8080`:

### HTML UI
- **📊 Monitoring** — live positions, trades, blacklist, daily PnL
- **⚙️ Parameters** — 48 tunable params grouped by category, with toggles/inputs/dirty tracking/audit log
- **📜 Audit Log** — every config change with old→new values + timestamp

### JSON API
**Monitoring:**
- `GET /api/status` — agent state, open positions, daily PnL, kill switch
- `GET /api/positions` — open positions (live from SQLite)
- `GET /api/trades` — last 100 trades
- `GET /api/blacklist` — current blacklist with expirations
- `GET /api/wallets` — redacted wallet summary

**Live tuning:**
- `GET /api/config` — all 48 tunable params with current values + schema
- `POST /api/config` — batch update (hot-reloads live modules)
- `GET /api/config/audit` — audit log of recent config changes

**Advanced analytics:**
- `GET /api/analyze/{mint}` — composite AlphaSignal (0-100 score from all 8 analyzers)
- `GET /api/order_flow/{mint}` — buy/sell pressure across 1m/5m/15m/1h windows
- `GET /api/lifecycle/{mint}` — lifecycle stage + size multiplier
- `GET /api/liquidity/{mint}` — slippage curve + recommended max position
- `GET /api/sentiment/{mint}` — Twitter/Telegram hype score
- `GET /api/mev/stats` — MEV/sandwich attack statistics
- `GET /api/wallet/{address}` — on-chain wallet reputation profile

**Controls:**
- `POST /api/kill` — trigger kill switch

Set `DASHBOARD_TOKEN` env var to require `?token=...` for sensitive endpoints.

---

## 🚢 Deployment (VPS)

### Option A: systemd
```bash
sudo bash scripts/install_systemd.sh
sudo systemctl start pumpfun-agent
sudo journalctl -u pumpfun-agent -f
```

### Option B: tmux (quick & dirty)
```bash
tmux new -s agent
cd pumpfun-agent && source .venv/bin/activate
python orchestrator.py
# Ctrl+B then D to detach
# tmux attach -t agent to re-attach
```

### Dashboard tunnel (optional)
If you want to view the dashboard from your laptop without exposing port 8080 publicly:
```bash
ssh -L 8080:localhost:8080 user@your-vps
# Then open http://localhost:8080 in your local browser
```

---

## 🧠 Architecture Notes

### Async-first
Everything is `asyncio`. Each strategy is a long-running task that emits `Signal` objects to a shared `asyncio.Queue`. A single consumer task reads the queue, routes through `RiskManager.approve()` → `Executor.execute()` → `chain_adapter.buy/sell()`.

### Non-custodial
- Wallet keys are loaded from env vars at startup.
- `WalletManager` exposes only the public address to other modules.
- All signing happens locally (Solana keypair / eth_account `sign_transaction`).
- **Never** log private keys. The logger explicitly redacts them.

### Anti-rugpull (real data)
The `AntiRugpullStrategy.check()` runs synchronously before every BUY. Real checks:
- **Holders** via Helius `getTokenLargestAccounts` (count + top-10 concentration)
- **Liquidity** via DexScreener API
- **Mint authority** renounced via Helius `getAccountInfo`
- **Freeze authority** renounced (Solana) — protects against seller freezing
- **Honeypot** via TokenSniffer (EVM) or freeze-authority check (Solana)

Results are cached for 5 minutes per token to avoid hammering the APIs.

### Pump.fun create event parser
`utils/pumpfun_parser.py` decodes the Anchor IDL create instruction:
- Discriminator = first 8 bytes of `sha256("global:create")` = `0x181ec828051c0777`
- Layout (Borsh): name (string), symbol (string), uri (string), mint (Pubkey), bonding_curve (Pubkey), user (Pubkey)

This is invoked by `SnipingStrategy.evaluate()` to extract the real mint address from each new launch signature.

### Risk manager (persisted)
The `RiskManager` is the single source of truth for position state. It enforces:
- Sizing (fixed or Kelly, using the persisted trade history)
- Daily loss cap (circuit breaker, persisted across restarts)
- Stop-loss / take-profit / max-hold
- Trade frequency cap
- Token blacklist (persisted with expiry)

### Extensibility
To add a new strategy:
1. Create `strategies/my_strategy.py` extending `BaseStrategy`.
2. Add a config block in `config.yaml` under `strategies:`.
3. Register it in `orchestrator.start()`.

To add a new chain:
1. Create `chains/my_chain_adapter.py` extending `BaseChainAdapter`.
2. Add config block in `config.yaml` under `chains:`.
3. Register in `chains/chain_factory.py`.

To add a new data provider:
1. Add a client class in `utils/data_providers.py`.
2. Register it in `get_providers()`.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Smoke tests cover the RiskManager approval gate (BUY/HOLD/SELL + kill switch).

---

## ⚠️ Known Limitations & TODO

- **EVM sell path**: requires ERC-20 `approve` before swap; not fully implemented.
- **Honeypot micro-buy on Solana**: not implemented (relies on freeze-authority check instead).
- **Jito tip instruction**: currently submits the swap tx alone in a Jito bundle; multi-tx bundle (with tip as separate leading ix) would require building a legacy `Message` from the v0 transaction.
- **Dashboard auth**: optional token-based only. For production, put behind a reverse proxy with auth (nginx + basic auth).

---

## 📜 License

MIT — see `LICENSE` (not included by default; add if you intend to distribute).

---

## 🤝 Contributing

This is a starting template. PRs welcome for:
- Real EVM pump.fun clone support (pumpa.fun, etc.)
- Multi-tx Jito bundles (tip + swap in same atomic bundle)
- WebSocket-based dashboard (live updates without polling)
- PostgreSQL persistence for high-frequency deployments
- Telegram inline buttons for approve/reject per trade
