"""
config_loader.py
================
Singleton-style configuration loader.
Loads YAML config + .env secrets, validates required keys.
Supports hot-update + persist to disk for the dashboard parameter editor.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


class ConfigError(Exception):
    pass


# Schema of all tunable parameters (used for validation + UI generation).
# Each entry: path -> (type, min, max, description)
# Paths use dot notation: "trading.fixed_size_sol"
TUNABLE_SCHEMA: dict[str, dict] = {
    # ---- Trading ----
    "trading.fixed_size_sol": {"type": "float", "min": 0.001, "max": 100.0,
                                "desc": "Default buy size in SOL (Strategy A: fixed sizing)"},
    "trading.fixed_size_evm": {"type": "float", "min": 0.0001, "max": 10.0,
                                "desc": "Default buy size in ETH for EVM chains"},
    "trading.slippage_bps": {"type": "int", "min": 50, "max": 5000,
                              "desc": "Default slippage tolerance (bps). 100=1%, 1000=10%"},
    "trading.priority_fee_micro_lamports": {"type": "int", "min": 0, "max": 10_000_000,
                                            "desc": "Solana priority fee (micro-lamports). Auto-overridden by GasOptimizer if active."},
    "trading.gas_price_gwei": {"type": "float", "min": 0.1, "max": 500.0,
                                "desc": "EVM gas price cap (gwei). Auto-overridden by GasOptimizer if active."},
    "trading.max_open_positions": {"type": "int", "min": 1, "max": 100,
                                    "desc": "Max simultaneous open positions"},

    # ---- Risk ----
    "risk.daily_loss_cap_pct": {"type": "float", "min": 1.0, "max": 50.0,
                                 "desc": "Halt agent if daily PnL drops below -X% of allocated capital"},
    "risk.per_token_max_loss_pct": {"type": "float", "min": 1.0, "max": 50.0,
                                     "desc": "Hard stop-loss per position (%)"},
    "risk.per_token_max_hold_minutes": {"type": "int", "min": 5, "max": 1440,
                                         "desc": "Max hold time per position (minutes). 1440 = 1 day."},
    "risk.blacklist_duration_minutes": {"type": "int", "min": 0, "max": 10080,
                                         "desc": "How long to blacklist a token after rugpull/SL (minutes)"},
    "risk.kelly.enabled": {"type": "bool", "desc": "Use Kelly criterion sizing instead of fixed size"},
    "risk.kelly.fraction": {"type": "float", "min": 0.1, "max": 1.0,
                             "desc": "Kelly fraction (0.5 = Half-Kelly, recommended)"},
    "risk.kelly.min_size_pct": {"type": "float", "min": 0.1, "max": 50.0,
                                 "desc": "Min position size as % of allocated capital"},
    "risk.kelly.max_size_pct": {"type": "float", "min": 1.0, "max": 100.0,
                                 "desc": "Max position size as % of allocated capital"},
    "risk.kelly.lookback_trades": {"type": "int", "min": 10, "max": 500,
                                    "desc": "Number of recent trades to estimate edge for Kelly"},

    # ---- Smart Exits ----
    "smart_exits.break_even_trigger_pct": {"type": "float", "min": 5.0, "max": 100.0,
                                            "desc": "Move stop to entry once position reaches +X%"},
    "smart_exits.break_even_stop_pct": {"type": "float", "min": -5.0, "max": 50.0,
                                         "desc": "Break-even stop offset (%). 0 = at entry, -2 = 2% below entry."},
    "smart_exits.trailing_stop_pct": {"type": "float", "min": 1.0, "max": 50.0,
                                       "desc": "Trail X% below peak price after break-even armed"},
    "smart_exits.max_hold_minutes_before_decay": {"type": "int", "min": 5, "max": 720,
                                                   "desc": "If PnL is flat after X minutes, sell 50%"},
    "smart_exits.decay_sell_pct": {"type": "float", "min": 0.1, "max": 1.0,
                                    "desc": "Fraction to sell on time decay (0.5 = 50%)"},
    "smart_exits.migration_pump_hold_minutes": {"type": "int", "min": 0, "max": 120,
                                                 "desc": "Hold through migration pump for X minutes"},

    # ---- Pyramiding ----
    "pyramiding.enabled": {"type": "bool", "desc": "Enable pyramiding on winning positions"},
    "pyramiding.min_pnl_pct": {"type": "float", "min": 10.0, "max": 500.0,
                                "desc": "Min PnL (%) before pyramiding allowed"},
    "pyramiding.max_additions": {"type": "int", "min": 0, "max": 10,
                                  "desc": "Max pyramid buys per position"},
    "pyramiding.add_size_pct": {"type": "float", "min": 0.1, "max": 2.0,
                                 "desc": "Each pyramid = X * original size (0.5 = 50%)"},
    "pyramiding.cooldown_minutes": {"type": "int", "min": 1, "max": 240,
                                     "desc": "Cooldown between pyramid buys"},

    # ---- Scoring ----
    "scoring.min_score_to_buy": {"type": "float", "min": 0.0, "max": 100.0,
                                  "desc": "Tokens scoring below this are skipped (TokenScorer)"},
    "scoring.min_score_to_pyramid": {"type": "float", "min": 0.0, "max": 100.0,
                                      "desc": "Tokens scoring above this can be pyramided"},

    # ---- Anti-Sandwich ----
    "anti_sandwich.base_slippage_bps": {"type": "int", "min": 50, "max": 3000,
                                         "desc": "Base slippage for medium trades (bps)"},
    "anti_sandwich.max_slippage_bps": {"type": "int", "min": 100, "max": 5000,
                                        "desc": "Hard cap on slippage (bps)"},
    "anti_sandwich.min_slippage_bps": {"type": "int", "min": 10, "max": 500,
                                        "desc": "Floor on slippage (bps). Tiny trades use this."},
    "anti_sandwich.size_liq_threshold": {"type": "float", "min": 0.001, "max": 0.5,
                                          "desc": "Trade size / liquidity ratio threshold for tight slippage"},
    "anti_sandwich.jito_threshold_sol": {"type": "float", "min": 0.0, "max": 10.0,
                                          "desc": "Use Jito bundle for buys >= X SOL"},

    # ---- Jito ----
    "jito.enabled": {"type": "bool", "desc": "Enable Jito bundle submission for MEV protection"},
    "jito.min_size_sol": {"type": "float", "min": 0.0, "max": 10.0,
                           "desc": "Min buy size (SOL) to use Jito bundles"},
    "jito.tip_lamports": {"type": "float", "min": 0.0, "max": 1.0,
                           "desc": "Jito tip in SOL (0.001 = 1M lamports)"},

    # ---- Strategy toggles ----
    "strategies.sniping.enabled": {"type": "bool", "desc": "Snipe new pump.fun launches"},
    "strategies.copy_trade.enabled": {"type": "bool", "desc": "Copy-trade target wallets"},
    "strategies.momentum.enabled": {"type": "bool", "desc": "RSI + volume breakout"},
    "strategies.grid_scalping.enabled": {"type": "bool", "desc": "Grid orders on established tokens"},
    "strategies.anti_rugpull.enabled": {"type": "bool", "desc": "Pre-trade rugpull checks (KEEP ON)"},

    # ---- Strategy params ----
    "strategies.sniping.max_buy_delay_ms": {"type": "int", "min": 100, "max": 10000,
                                             "desc": "Max latency from launch detection to buy (ms)"},
    "strategies.sniping.min_initial_liquidity_usd": {"type": "float", "min": 0, "max": 1_000_000,
                                                      "desc": "Min liquidity at launch for sniping (USD)"},
    "strategies.momentum.scan_interval_sec": {"type": "int", "min": 5, "max": 300,
                                               "desc": "How often to scan for momentum signals (sec)"},
    "strategies.momentum.min_volume_5m_usd": {"type": "float", "min": 0, "max": 10_000_000,
                                               "desc": "Min 5-min volume (USD) for momentum buy"},
    "strategies.momentum.min_price_change_5m_pct": {"type": "float", "min": 0, "max": 500,
                                                     "desc": "Min 5-min price change (%) for momentum buy"},
    "strategies.momentum.rsi_oversold": {"type": "float", "min": 5, "max": 50,
                                          "desc": "RSI oversold threshold (buy signal)"},
    "strategies.momentum.rsi_overbought": {"type": "float", "min": 50, "max": 95,
                                            "desc": "RSI overbought threshold (sell signal)"},
}


class Config:
    _instance: "Config | None" = None

    def __init__(self, config_path: str = "config/config.yaml"):
        self._config_path = Path(config_path)
        if not self._config_path.exists():
            raise ConfigError(
                f"Config file not found: {self._config_path}. "
                "Copy config/config.yaml.example to config/config.yaml and fill in secrets."
            )

        load_dotenv()

        with open(self._config_path, "r", encoding="utf-8") as f:
            self._data: dict[str, Any] = yaml.safe_load(f)

        self._validate()

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------
    @classmethod
    def get(cls, config_path: str | None = None) -> "Config":
        if cls._instance is None:
            cls._instance = cls(config_path or "config/config.yaml")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self._data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def set_nested(self, keys: list[str], value: Any) -> None:
        """Set a value at a nested path. Creates intermediate dicts."""
        cur: Any = self._data
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = value

    def env(self, key: str, required: bool = True) -> str:
        val = os.environ.get(key, "")
        if required and not val:
            raise ConfigError(f"Environment variable '{key}' is required but not set.")
        return val

    # ------------------------------------------------------------------
    # Hot-update + persist
    # ------------------------------------------------------------------
    def update_tunable(self, path: str, value: Any) -> tuple[bool, str]:
        """
        Update a single tunable parameter.
        path: dot notation, e.g. "trading.fixed_size_sol"
        value: must match the schema's expected type.
        Returns (success, message).
        """
        if path not in TUNABLE_SCHEMA:
            return False, f"Path '{path}' is not a tunable parameter"

        schema = TUNABLE_SCHEMA[path]
        keys = path.split(".")

        # Type coercion + validation
        try:
            if schema["type"] == "bool":
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                coerced = bool(value)
            elif schema["type"] == "int":
                coerced = int(float(value))
                if "min" in schema and coerced < schema["min"]:
                    return False, f"Value {coerced} below min {schema['min']}"
                if "max" in schema and coerced > schema["max"]:
                    return False, f"Value {coerced} above max {schema['max']}"
            elif schema["type"] == "float":
                coerced = float(value)
                if "min" in schema and coerced < schema["min"]:
                    return False, f"Value {coerced} below min {schema['min']}"
                if "max" in schema and coerced > schema["max"]:
                    return False, f"Value {coerced} above max {schema['max']}"
            else:
                return False, f"Unknown type {schema['type']}"
        except (ValueError, TypeError) as e:
            return False, f"Type coercion failed: {e}"

        # Apply
        self.set_nested(keys, coerced)
        return True, f"Updated {path} = {coerced}"

    def save(self) -> None:
        """Persist current config to disk. Backs up the previous file first."""
        backup_path = self._config_path.with_suffix(".yaml.bak")
        if self._config_path.exists():
            shutil.copy2(self._config_path, backup_path)
        with open(self._config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, default_flow_style=False, sort_keys=False)

    def get_tunable_view(self) -> dict[str, dict]:
        """
        Returns all tunable parameters with current values + schema info.
        Used by the dashboard to render the parameter editor.
        """
        out: dict[str, dict] = {}
        for path, schema in TUNABLE_SCHEMA.items():
            keys = path.split(".")
            current = self.get_nested(*keys, default=None)
            out[path] = {
                "current": current,
                "type": schema["type"],
                "min": schema.get("min"),
                "max": schema.get("max"),
                "desc": schema["desc"],
            }
        return out

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        required_top = ["trading", "risk", "chains", "strategies", "notifications", "safety"]
        for k in required_top:
            if k not in self._data:
                raise ConfigError(f"Missing top-level key '{k}' in config.")
        if self._data["trading"]["mode"] not in ("paper", "live"):
            raise ConfigError("trading.mode must be 'paper' or 'live'")
        if not self._data["trading"]["chains"]:
            raise ConfigError("trading.chains cannot be empty")
