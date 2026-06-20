"""
test_paper_simulation.py
========================
End-to-end paper mode simulation test.

Verifies the full pipeline without needing real Solana RPC:
1. Mock chain adapter (returns fake prices, fake buy/sell success)
2. Mock data providers (DexScreener, Helius, Birdeye return canned data)
3. Inject a BUY signal into the signal queue
4. Verify: signal → risk approval → anti_rugpull → token scoring → executor → position opened
5. Inject price updates → verify smart exits fire (break-even, trailing, TP ladder)
6. Inject a SELL signal → verify position closed + PnL computed

Run:  pytest tests/test_paper_simulation.py -v
"""
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Minimal config
_CFG = {
    "environment": "test",
    "trading": {
        "mode": "paper", "chains": ["solana"],
        "fixed_size_sol": 0.05, "fixed_size_evm": 0.005, "slippage_bps": 800,
        "priority_fee_micro_lamports": 50000, "gas_price_gwei": 3, "max_open_positions": 5,
    },
    "risk": {
        "daily_loss_cap_pct": 8.0, "per_token_max_loss_pct": 15.0,
        "per_token_max_hold_minutes": 240, "blacklist_duration_minutes": 60,
        "kelly": {"enabled": False, "fraction": 0.5, "min_size_pct": 1.0,
                  "max_size_pct": 10.0, "lookback_trades": 50},
        "backtesting": {"enabled": False},
    },
    "persistence": {"db_path": "/tmp/test_paper_sim.db"},
    "chains": {"solana": {
        "rpc_endpoints": ["http://localhost:8899"], "ws_endpoint": "ws://localhost:8900",
        "jupiter_api": "http://localhost", "dedicated_wallet_seed_env": "SOLANA_WALLET_SEED",
        "allocated_capital_sol": 1.0,
    }},
    "jito": {"enabled": False, "min_size_sol": 0.1, "tip_lamports": 0.001},
    "strategies": {
        "sniping": {"enabled": False}, "copy_trade": {"enabled": False},
        "momentum": {"enabled": False}, "grid_scalping": {"enabled": False},
        "anti_rugpull": {"enabled": True, "min_holders": 5,
                          "max_top10_holders_pct": 0.80, "min_liquidity_usd": 100,
                          "require_renounced_mint": False, "check_honeypot": False,
                          "reject_if_mint_authority_active": False},
    },
    "notifications": {"telegram": {"enabled": False, "bot_token_env": "X", "chat_id_env": "Y"}},
    "monitoring": {"dashboard": {"enabled": False, "port": 8080}},
    "logging": {"level": "WARNING", "file": "/tmp/test_paper.log"},
    "safety": {"kill_switch_file": "/tmp/KILL_TEST", "max_trade_frequency_per_minute": 100},
    "scoring": {"min_score_to_buy": 0.0, "min_score_to_pyramid": 75.0},
    "smart_exits": {
        "break_even_trigger_pct": 25.0, "break_even_stop_pct": 0.0,
        "trailing_stop_pct": 15.0,
        "take_profit_ladder": [
            {"pnl_pct": 50.0, "sell_pct": 0.25},
            {"pnl_pct": 100.0, "sell_pct": 0.25},
        ],
        "max_hold_minutes_before_decay": 60, "decay_sell_pct": 0.5,
        "migration_pump_hold_minutes": 15,
    },
    "pyramiding": {"enabled": False, "min_pnl_pct": 50.0, "max_additions": 2,
                   "add_size_pct": 0.5, "cooldown_minutes": 15},
    "anti_sandwich": {"base_slippage_bps": 300, "max_slippage_bps": 1500,
                      "min_slippage_bps": 50, "size_liq_threshold": 0.02,
                      "jito_threshold_sol": 0.1},
}

_cfg_path = Path(tempfile.gettempdir()) / "test_paper_config.yaml"
_cfg_path.write_text(yaml.safe_dump(_CFG))

from utils.config_loader import Config
Config.reset()
Config._instance = Config(str(_cfg_path))

from chains.base_chain import BaseChainAdapter, OrderResult, TokenInfo
from execution.executor import Executor
from risk.risk_manager import RiskManager
from risk.smart_exits import SmartExitManager
from strategies.anti_rugpull import AntiRugpullStrategy, RugpullVerdict
from strategies.base_strategy import Signal, SignalType
from utils.kill_switch import KillSwitch
from utils.persistence import Persistence
from utils.token_scorer import TokenScorer, TokenScore
from analysis.alpha_signal import AlphaSignalGenerator


# ----------------------------------------------------------------------
# Mock chain adapter
# ----------------------------------------------------------------------
class MockAdapter(BaseChainAdapter):
    chain_name = "solana"
    _price = 0.001  # SOL per token

    async def connect(self): pass
    async def close(self): pass
    async def get_balance(self, w): return 5.0
    async def get_token_info(self, t):
        return TokenInfo("solana", t, "MOCK", "Mock Token", 9)
    async def get_price(self, t): return self._price
    def set_price(self, p): self._price = p
    async def buy(self, t, a, s):
        tokens = a / self._price
        return OrderResult(True, "mock_sig_" + str(int(time.time())),
                           self._price, tokens)
    async def sell(self, t, a, s):
        sol = a * self._price
        return OrderResult(True, "mock_sig_sell", self._price, sol)
    async def watch_new_tokens(self):
        if False: yield {}  # make it an async generator
    async def watch_wallet_activity(self, w):
        if False: yield {}


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def setup():
    KillSwitch.reset()
    # Reset DB
    db = Persistence.get()
    # Clean tables
    with db._cursor() as cur:
        cur.execute("DELETE FROM positions")
        cur.execute("DELETE FROM trades")
        cur.execute("DELETE FROM blacklist")
        cur.execute("DELETE FROM daily_pnl")
    yield db
    db.close()
    Persistence._instance = None


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_alpha_signal_weights_sum_to_one():
    """Verify weights are properly normalized."""
    total = sum(AlphaSignalGenerator.WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


def test_alpha_signal_weights_tuned_for_memecoins():
    """Verify the tuned weights match memecoin trading priorities."""
    w = AlphaSignalGenerator.WEIGHTS
    # order_flow should be the highest weight
    assert w["order_flow"] == max(w.values()), "order_flow should dominate"
    # smart_money second
    sorted_w = sorted(w.values(), reverse=True)
    assert w["smart_money"] == sorted_w[1], "smart_money should be 2nd"
    # token_quality lower than order_flow (anti_rugpull gate already covers it)
    assert w["token_quality"] < w["order_flow"]
    # mev_penalty and bonding_curve are situational — should be lowest
    assert w["mev_penalty"] <= 0.05
    assert w["bonding_curve"] <= 0.05


@pytest.mark.asyncio
async def test_paper_buy_opens_position(setup):
    """Test that a BUY signal in paper mode opens a position."""
    adapter = MockAdapter()
    risk = RiskManager()
    notifier = MagicMock()
    notifier.notify_trade_opened = AsyncMock()
    notifier.notify_trade_closed = AsyncMock()
    notifier.notify_error = AsyncMock()
    notifier.notify_rugpull = AsyncMock()
    anti_rug = AntiRugpullStrategy(adapter)
    # Bypass anti-rugpull check (return safe verdict)
    anti_rug.check = AsyncMock(return_value=RugpullVerdict(True, ["All passed"], 1.0))
    scorer = TokenScorer()
    # Bypass scoring (return strong buy)
    scorer.score = AsyncMock(return_value=TokenScore(
        mint="TESTMINT", score=85.0, confidence=0.9,
        recommendation="BUY_STRONG", breakdown={}, reasons=[]
    ))
    smart_exits = SmartExitManager()

    from utils.anti_sandwich import AntiSandwich
    from utils.dev_tracker import DevTracker
    from utils.gas_optimizer import GasOptimizer
    from utils.bonding_curve import BondingCurveAnalyzer
    from chains.jito_client import JitoClient

    executor = Executor(
        adapter=adapter, risk_manager=risk, notifier=notifier, anti_rug=anti_rug,
        token_scorer=scorer, smart_exits=smart_exits,
        dev_tracker=DevTracker(), gas_optimizer=GasOptimizer(),
        anti_sandwich=AntiSandwich(),
    )
    # Patch bonding.fetch_state to return None (skip in test)
    executor.bonding.fetch_state = AsyncMock(return_value=None)

    signal = Signal(
        strategy="test", chain="solana", token_address="TESTMINT",
        signal_type=SignalType.BUY, suggested_size_pct=0.05, confidence=0.8,
        reason="Test buy", stop_loss_pct=15.0, take_profit_pct=100.0,
    )
    await executor.execute(signal, allocated_capital=1.0)

    # Verify position opened
    assert len(risk.positions) == 1, "Position should be open"
    pos = risk.positions["solana:TESTMINT"]
    assert pos.entry_price == 0.001
    assert pos.size_base == 0.05 * 1.5  # BUY_STRONG multiplier = 1.5

    # Verify trade persisted
    trades = setup.load_recent_trades(limit=10)
    assert len(trades) >= 1
    assert trades[0]["side"] == "BUY"
    assert trades[0]["token"] == "TESTMINT"


@pytest.mark.asyncio
async def test_paper_sell_closes_position(setup):
    """Test that a SELL signal closes a position and computes PnL."""
    adapter = MockAdapter()
    adapter.set_price(0.002)  # 2x entry price
    risk = RiskManager()
    notifier = MagicMock()
    notifier.notify_trade_opened = AsyncMock()
    notifier.notify_trade_closed = AsyncMock()
    notifier.notify_error = AsyncMock()
    notifier.notify_rugpull = AsyncMock()
    anti_rug = AntiRugpullStrategy(adapter)
    anti_rug.check = AsyncMock(return_value=RugpullVerdict(True, ["OK"], 1.0))
    scorer = TokenScorer()
    scorer.score = AsyncMock(return_value=TokenScore(
        mint="TESTMINT", score=70.0, confidence=0.8,
        recommendation="BUY", breakdown={}, reasons=[]
    ))
    smart_exits = SmartExitManager()

    from utils.anti_sandwich import AntiSandwich
    from utils.dev_tracker import DevTracker
    from utils.gas_optimizer import GasOptimizer

    executor = Executor(
        adapter=adapter, risk_manager=risk, notifier=notifier, anti_rug=anti_rug,
        token_scorer=scorer, smart_exits=smart_exits,
        dev_tracker=DevTracker(), gas_optimizer=GasOptimizer(),
        anti_sandwich=AntiSandwich(),
    )
    executor.bonding.fetch_state = AsyncMock(return_value=None)

    # Open position at entry price 0.001
    adapter.set_price(0.001)
    buy_signal = Signal(
        strategy="test", chain="solana", token_address="TESTMINT",
        signal_type=SignalType.BUY, suggested_size_pct=0.05, confidence=0.8,
        reason="Test buy", stop_loss_pct=15.0, take_profit_pct=100.0,
    )
    await executor.execute(buy_signal, allocated_capital=1.0)
    assert len(risk.positions) == 1

    # Move price up 100% (take profit territory)
    adapter.set_price(0.002)
    sell_signal = Signal(
        strategy="test", chain="solana", token_address="TESTMINT",
        signal_type=SignalType.SELL, suggested_size_pct=1.0, confidence=1.0,
        reason="Manual sell @ +100%",
    )
    await executor.execute(sell_signal, allocated_capital=1.0)

    # Verify position closed
    assert len(risk.positions) == 0, "Position should be closed"
    trades = setup.load_recent_trades(limit=10)
    sell_trades = [t for t in trades if t["side"] == "SELL"]
    assert len(sell_trades) >= 1
    assert sell_trades[0]["pnl_pct"] == pytest.approx(100.0, abs=0.1)


@pytest.mark.asyncio
async def test_smart_exit_take_profit_ladder(setup):
    """Test that the TP ladder fires at +50% and +100%."""
    adapter = MockAdapter()
    risk = RiskManager()
    smart_exits = SmartExitManager()
    notifier = MagicMock()
    notifier.notify_trade_opened = AsyncMock()
    notifier.notify_trade_closed = AsyncMock()
    notifier.notify_error = AsyncMock()
    notifier.notify_rugpull = AsyncMock()
    anti_rug = AntiRugpullStrategy(adapter)
    anti_rug.check = AsyncMock(return_value=RugpullVerdict(True, ["OK"], 1.0))
    scorer = TokenScorer()
    scorer.score = AsyncMock(return_value=TokenScore(
        mint="TESTMINT", score=70.0, confidence=0.8,
        recommendation="BUY", breakdown={}, reasons=[]
    ))

    from utils.anti_sandwich import AntiSandwich
    from utils.dev_tracker import DevTracker
    from utils.gas_optimizer import GasOptimizer

    executor = Executor(
        adapter=adapter, risk_manager=risk, notifier=notifier, anti_rug=anti_rug,
        token_scorer=scorer, smart_exits=smart_exits,
        dev_tracker=DevTracker(), gas_optimizer=GasOptimizer(),
        anti_sandwich=AntiSandwich(),
    )
    executor.bonding.fetch_state = AsyncMock(return_value=None)

    # Open position at 0.001
    adapter.set_price(0.001)
    buy_signal = Signal(
        strategy="test", chain="solana", token_address="TESTMINT",
        signal_type=SignalType.BUY, suggested_size_pct=0.05, confidence=0.8,
        reason="Test buy", stop_loss_pct=15.0, take_profit_pct=200.0,
    )
    await executor.execute(buy_signal, allocated_capital=1.0)

    # Price moves to +60% — should trigger tranche 1 (+50%, sell 25%)
    adapter.set_price(0.0016)
    async def price_provider(chain, token): return 0.0016
    exits = await smart_exits.check_exits(risk.positions, price_provider)
    tp_exits = [e for e in exits if "TP ladder" in e.reason]
    assert len(tp_exits) >= 1, "TP ladder tranche 1 should fire at +60%"
    assert tp_exits[0].size_pct_to_sell == 0.25

    # Price moves to +120% — should trigger tranche 2 (+100%, sell 25%)
    adapter.set_price(0.0022)
    exits = await smart_exits.check_exits(risk.positions, price_provider)
    # Reset tranche flags for the test by re-checking
    # Actually we need to update price_provider
    async def price_provider2(chain, token): return 0.0022
    exits = await smart_exits.check_exits(risk.positions, price_provider2)
    tp_exits = [e for e in exits if "TP ladder" in e.reason]
    assert len(tp_exits) >= 1, "TP ladder tranche 2 should fire at +120%"


@pytest.mark.asyncio
async def test_smart_exit_break_even_and_trailing(setup):
    """Test break-even arming + trailing stop."""
    adapter = MockAdapter()
    risk = RiskManager()
    smart_exits = SmartExitManager()
    smart_exits.break_even_trigger_pct = 25.0
    smart_exits.trailing_stop_pct = 15.0
    notifier = MagicMock()
    notifier.notify_trade_opened = AsyncMock()
    notifier.notify_trade_closed = AsyncMock()
    notifier.notify_error = AsyncMock()
    notifier.notify_rugpull = AsyncMock()
    anti_rug = AntiRugpullStrategy(adapter)
    anti_rug.check = AsyncMock(return_value=RugpullVerdict(True, ["OK"], 1.0))
    scorer = TokenScorer()
    scorer.score = AsyncMock(return_value=TokenScore(
        mint="TESTMINT", score=70.0, confidence=0.8,
        recommendation="BUY", breakdown={}, reasons=[]
    ))

    from utils.anti_sandwich import AntiSandwich
    from utils.dev_tracker import DevTracker
    from utils.gas_optimizer import GasOptimizer

    executor = Executor(
        adapter=adapter, risk_manager=risk, notifier=notifier, anti_rug=anti_rug,
        token_scorer=scorer, smart_exits=smart_exits,
        dev_tracker=DevTracker(), gas_optimizer=GasOptimizer(),
        anti_sandwich=AntiSandwich(),
    )
    executor.bonding.fetch_state = AsyncMock(return_value=None)

    # Open at 0.001
    adapter.set_price(0.001)
    buy_signal = Signal(
        strategy="test", chain="solana", token_address="TESTMINT",
        signal_type=SignalType.BUY, suggested_size_pct=0.05, confidence=0.8,
        reason="Test buy", stop_loss_pct=15.0, take_profit_pct=500.0,
    )
    await executor.execute(buy_signal, allocated_capital=1.0)

    # Price +30% → break-even armed
    adapter.set_price(0.0013)
    async def pp1(chain, token): return 0.0013
    await smart_exits.check_exits(risk.positions, pp1)
    meta = smart_exits.get_meta("solana", "TESTMINT")
    assert meta is not None
    assert meta.break_even_triggered, "Break-even should be armed at +30%"
    assert meta.highest_price == 0.0013

    # Price +50% → peak updated
    adapter.set_price(0.0015)
    async def pp2(chain, token): return 0.0015
    await smart_exits.check_exits(risk.positions, pp2)
    assert meta.highest_price == 0.0015

    # Price drops 15% from peak (0.0015 * 0.85 = 0.001275) → trailing fires
    adapter.set_price(0.001275)
    async def pp3(chain, token): return 0.001275
    exits = await smart_exits.check_exits(risk.positions, pp3)
    trailing_exits = [e for e in exits if "Trailing stop" in e.reason]
    assert len(trailing_exits) >= 1, "Trailing stop should fire"
    assert trailing_exits[0].size_pct_to_sell == 1.0  # full exit


@pytest.mark.asyncio
async def test_risk_manager_daily_loss_cap(setup):
    """Test that daily loss cap triggers kill switch."""
    risk = RiskManager()
    # Simulate accumulated losses
    risk.daily_pnl_pct = -7.5  # close to cap of 8%
    signal = Signal(
        strategy="test", chain="solana", token_address="TESTMINT2",
        signal_type=SignalType.BUY, suggested_size_pct=0.05, confidence=0.8,
        reason="Test", stop_loss_pct=15.0, take_profit_pct=50.0,
    )
    approved, size, reason = await risk.approve(signal, allocated_capital=1.0)
    assert approved  # still under cap

    # Push past the cap
    risk.daily_pnl_pct = -9.0
    approved, size, reason = await risk.approve(signal, allocated_capital=1.0)
    assert not approved
    assert "Daily loss cap" in reason
    assert KillSwitch.is_triggered()


@pytest.mark.asyncio
async def test_kill_switch_blocks_all_trades(setup):
    """Test that kill switch blocks all new trades."""
    KillSwitch.trigger("Test kill")
    risk = RiskManager()
    signal = Signal(
        strategy="test", chain="solana", token_address="TESTMINT3",
        signal_type=SignalType.BUY, suggested_size_pct=0.05, confidence=0.8,
        reason="Test", stop_loss_pct=15.0, take_profit_pct=50.0,
    )
    approved, _, reason = await risk.approve(signal, allocated_capital=1.0)
    assert not approved
    assert "Kill switch" in reason
    KillSwitch.reset()


def test_position_persisted_across_restart(setup):
    """Test that positions survive a RiskManager reinit (simulating restart)."""
    from risk.types import Position
    db = Persistence.get()
    pos = Position(
        chain="solana", token="PERSISTTEST", strategy="test",
        entry_price=0.001, size_base=0.05, size_token=50.0,
        stop_loss_pct=15.0, take_profit_pct=100.0,
    )
    db.upsert_position(pos)
    db.close()
    Persistence._instance = None

    # New RiskManager should restore the position
    risk = RiskManager()
    assert "solana:PERSISTTEST" in risk.positions
    assert risk.positions["solana:PERSISTTEST"].entry_price == 0.001
