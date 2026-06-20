"""
test_risk_manager.py
====================
Smoke tests for the RiskManager approval gate.
Run:  pytest tests/test_risk_manager.py -v
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Write a minimal config to a temp file before any test imports Config
import tempfile, yaml
_TMP_CFG = {
    "environment": "test",
    "trading": {
        "mode": "paper",
        "chains": ["solana"],
        "fixed_size_sol": 0.05,
        "fixed_size_evm": 0.005,
        "slippage_bps": 800,
        "priority_fee_micro_lamports": 50000,
        "gas_price_gwei": 3,
        "max_open_positions": 5,
    },
    "risk": {
        "daily_loss_cap_pct": 8.0,
        "per_token_max_loss_pct": 15.0,
        "per_token_max_hold_minutes": 240,
        "blacklist_duration_minutes": 60,
        "kelly": {"enabled": False, "fraction": 0.5,
                  "min_size_pct": 1.0, "max_size_pct": 10.0, "lookback_trades": 50},
        "backtesting": {"enabled": False},
    },
    "chains": {"solana": {
        "rpc_endpoints": ["http://localhost:8899"],
        "ws_endpoint": "ws://localhost:8900",
        "jupiter_api": "http://localhost",
        "dedicated_wallet_seed_env": "SOLANA_WALLET_SEED",
        "allocated_capital_sol": 1.0,
    }},
    "strategies": {"sniping": {"enabled": False}, "copy_trade": {"enabled": False},
                   "momentum": {"enabled": False}, "grid_scalping": {"enabled": False},
                   "anti_rugpull": {"enabled": True}},
    "notifications": {"telegram": {"enabled": False, "bot_token_env": "X", "chat_id_env": "Y"}},
    "logging": {"level": "WARNING", "file": "./data/test.log"},
    "safety": {"kill_switch_file": "./data/KILL_SWITCH",
               "max_trade_frequency_per_minute": 10},
    "scoring": {"min_score_to_buy": 55.0, "min_score_to_pyramid": 75.0},
    "smart_exits": {
        "break_even_trigger_pct": 25.0, "break_even_stop_pct": 0.0,
        "trailing_stop_pct": 15.0,
        "take_profit_ladder": [
            {"pnl_pct": 50.0, "sell_pct": 0.25},
            {"pnl_pct": 100.0, "sell_pct": 0.25},
            {"pnl_pct": 200.0, "sell_pct": 0.25},
        ],
        "max_hold_minutes_before_decay": 60, "decay_sell_pct": 0.5,
        "migration_pump_hold_minutes": 15,
    },
    "pyramiding": {"enabled": True, "min_pnl_pct": 50.0, "max_additions": 2,
                   "add_size_pct": 0.5, "cooldown_minutes": 15},
    "anti_sandwich": {"base_slippage_bps": 300, "max_slippage_bps": 1500,
                      "min_slippage_bps": 50, "size_liq_threshold": 0.02,
                      "jito_threshold_sol": 0.1},
}

cfg_path = Path(tempfile.gettempdir()) / "test_config.yaml"
cfg_path.write_text(yaml.safe_dump(_TMP_CFG))
os.environ["PYTHONWARNINGS"] = "ignore"

from utils.config_loader import Config
Config.reset()
Config._instance = Config(str(cfg_path))

from strategies.base_strategy import Signal, SignalType
from risk.risk_manager import RiskManager
from utils.kill_switch import KillSwitch


@pytest.fixture
def rm():
    KillSwitch.reset()
    return RiskManager()


def test_buy_approved_fixed_size(rm):
    sig = Signal("sniping", "solana", "MINT123", SignalType.BUY,
                 0.02, 0.6, "test", stop_loss_pct=10, take_profit_pct=50)
    approved, size, reason = asyncio.run(rm.approve(sig, allocated_capital=1.0))
    assert approved, f"Should approve: {reason}"
    assert size == pytest.approx(0.05)


def test_hold_rejected(rm):
    sig = Signal("test", "solana", "X", SignalType.HOLD, 0, 0, "n/a")
    approved, _, _ = asyncio.run(rm.approve(sig, 1.0))
    assert not approved


def test_sell_without_position_rejected(rm):
    sig = Signal("test", "solana", "Y", SignalType.SELL, 1.0, 0.5, "exit")
    approved, _, _ = asyncio.run(rm.approve(sig, 1.0))
    assert not approved


def test_kill_switch_blocks(rm):
    KillSwitch.trigger("test")
    sig = Signal("test", "solana", "Z", SignalType.BUY, 0.01, 0.5, "x")
    approved, _, _ = asyncio.run(rm.approve(sig, 1.0))
    assert not approved
    KillSwitch.reset()
