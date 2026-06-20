"""
config_hot_reload.py
====================
Service that pushes config changes from the Config singleton into live
running modules (RiskManager, SmartExitManager, Pyramider, GasOptimizer,
AntiSandwich, JitoClient, etc.).

When the dashboard POSTs a config update:
1. Config singleton is mutated in-memory + persisted to YAML.
2. An entry is written to the audit log (SQLite).
3. apply_hot_reload() is called to refresh live module attributes.
4. A Telegram notification is sent (optional).

The Orchestrator holds references to all live module instances and calls
apply_hot_reload() after any successful update.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

from utils.config_loader import Config
from utils.logger import setup_logger
from utils.persistence import Persistence

log = setup_logger("hot_reload")


@dataclass
class ConfigChange:
    path: str
    old_value: Any
    new_value: Any
    ts: float
    source: str = "dashboard"   # "dashboard" | "api" | "auto"


class ConfigHotReloader:
    """Applies config changes to live modules + logs them."""

    def __init__(self, orchestrator=None) -> None:
        self.orch = orchestrator
        self.db = Persistence.get()

    def attach(self, orchestrator) -> None:
        self.orch = orchestrator

    # ------------------------------------------------------------------
    def apply_update(self, path: str, new_value: Any, source: str = "dashboard") -> tuple[bool, str]:
        """
        Update a single tunable parameter and push to live modules.
        Returns (success, message).
        """
        cfg = Config.get()
        keys = path.split(".")
        old_value = cfg.get_nested(*keys, default=None)

        success, msg = cfg.update_tunable(path, new_value)
        if not success:
            return False, msg

        # Persist to disk
        try:
            cfg.save()
        except Exception as e:
            log.error("hot_reload.persist_failed", error=str(e))
            return False, f"Update applied in-memory but persist failed: {e}"

        # Audit log
        change = ConfigChange(
            path=path, old_value=old_value, new_value=new_value,
            ts=time.time(), source=source,
        )
        self._log_change(change)

        # Push to live modules
        try:
            self._apply_to_live(path)
        except Exception as e:
            log.error("hot_reload.live_apply_failed", path=path, error=str(e))
            return True, f"Updated {path} (live apply failed: {e})"

        log.info("hot_reload.applied", path=path,
                 old=old_value, new=new_value, source=source)
        return True, f"Updated {path}: {old_value} → {new_value}"

    def apply_updates(self, updates: dict[str, Any], source: str = "dashboard") -> list[tuple[str, bool, str]]:
        """Batch update. Returns list of (path, success, message)."""
        results: list[tuple[str, bool, str]] = []
        for path, value in updates.items():
            ok, msg = self.apply_update(path, value, source=source)
            results.append((path, ok, msg))
        return results

    # ------------------------------------------------------------------
    def _apply_to_live(self, path: str) -> None:
        """Push a single parameter change to the live module(s) that use it."""
        if not self.orch:
            log.warning("hot_reload.no_orchestrator", msg="Changes will apply on next restart")
            return

        cfg = Config.get()

        # Trading params
        if path == "trading.fixed_size_sol":
            # Used inside RiskManager._compute_size via cfg, no attribute to update
            pass
        elif path == "trading.max_open_positions":
            self.orch.risk.tcfg = cfg["trading"]
        elif path == "trading.slippage_bps":
            # Used inside Executor via cfg, no attribute to update
            pass

        # Risk params
        elif path.startswith("risk."):
            self.orch.risk.rcfg = cfg["risk"]
            if path.startswith("risk.kelly."):
                self.orch.risk.rcfg["kelly"] = cfg["risk"]["kelly"]

        # Smart exits
        elif path.startswith("smart_exits."):
            secfg = cfg.get_nested("smart_exits", default={})
            if path == "smart_exits.break_even_trigger_pct":
                self.orch.smart_exits.break_even_trigger_pct = secfg.get("break_even_trigger_pct", 25.0)
            elif path == "smart_exits.break_even_stop_pct":
                self.orch.smart_exits.break_even_stop_pct = secfg.get("break_even_stop_pct", 0.0)
            elif path == "smart_exits.trailing_stop_pct":
                self.orch.smart_exits.trailing_stop_pct = secfg.get("trailing_stop_pct", 15.0)
            elif path == "smart_exits.max_hold_minutes_before_decay":
                self.orch.smart_exits.max_hold_minutes_before_decay = secfg.get("max_hold_minutes_before_decay", 60)
            elif path == "smart_exits.decay_sell_pct":
                self.orch.smart_exits.decay_sell_pct = secfg.get("decay_sell_pct", 0.5)
            elif path == "smart_exits.migration_pump_hold_minutes":
                self.orch.smart_exits.migration_pump_hold_minutes = secfg.get("migration_pump_hold_minutes", 15)
            elif path == "smart_exits.take_profit_ladder":
                self.orch.smart_exits.tps = secfg.get("take_profit_ladder", [])

        # Pyramiding
        elif path.startswith("pyramiding."):
            pcfg = cfg.get_nested("pyramiding", default={})
            self.orch.pyramider.enabled = pcfg.get("enabled", True)
            self.orch.pyramider.min_pnl_pct_to_pyramid = pcfg.get("min_pnl_pct", 50.0)
            self.orch.pyramider.max_additions = pcfg.get("max_additions", 2)
            self.orch.pyramider.add_size_pct = pcfg.get("add_size_pct", 0.5)
            self.orch.pyramider.cooldown_minutes = pcfg.get("cooldown_minutes", 15)

        # Scoring
        elif path.startswith("scoring."):
            scfg = cfg.get_nested("scoring", default={})
            self.orch.token_scorer.min_score_to_buy = scfg.get("min_score_to_buy", 55.0)
            self.orch.token_scorer.min_score_to_pyramid = scfg.get("min_score_to_pyramid", 75.0)

        # Anti-sandwich
        elif path.startswith("anti_sandwich."):
            ascfg = cfg.get_nested("anti_sandwich", default={})
            self.orch.anti_sandwich.base_slippage_bps = ascfg.get("base_slippage_bps", 300)
            self.orch.anti_sandwich.max_slippage_bps = ascfg.get("max_slippage_bps", 1500)
            self.orch.anti_sandwich.min_slippage_bps = ascfg.get("min_slippage_bps", 50)
            self.orch.anti_sandwich.size_to_liquidity_threshold = ascfg.get("size_liq_threshold", 0.02)
            self.orch.anti_sandwich.jito_threshold_sol = ascfg.get("jito_threshold_sol", 0.1)

        # Jito
        elif path.startswith("jito."):
            jcfg = cfg.get_nested("jito", default={})
            # JitoClient instances are created per-call in SolanaAdapter.buy(),
            # so no live attribute to update; new calls will read fresh config.

        # Strategy toggles — these need orchestrator-level action
        elif path in ("strategies.sniping.enabled", "strategies.copy_trade.enabled",
                      "strategies.momentum.enabled", "strategies.grid_scalping.enabled",
                      "strategies.anti_rugpull.enabled"):
            log.info("hot_reload.strategy_toggle_changed", path=path,
                     msg="Strategy toggles require restart to take effect (live strategies cannot be hot-toggled)")

        # Strategy params (read live from cfg by the strategy itself)
        elif path.startswith("strategies."):
            # Strategies re-read their config block on each loop iteration via Config.get_nested()
            # No attribute update needed
            pass

        else:
            log.warning("hot_reload.unhandled_path", path=path)

    # ------------------------------------------------------------------
    def _log_change(self, change: ConfigChange) -> None:
        """Write the change to the SQLite audit log."""
        try:
            with self.db._cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS config_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        path TEXT NOT NULL,
                        old_value TEXT,
                        new_value TEXT,
                        ts REAL NOT NULL,
                        source TEXT
                    )
                """)
                cur.execute("""
                    INSERT INTO config_audit (path, old_value, new_value, ts, source)
                    VALUES (?, ?, ?, ?, ?)
                """, (change.path, str(change.old_value), str(change.new_value),
                      change.ts, change.source))
        except Exception as e:
            log.error("hot_reload.audit_log_failed", error=str(e))

    def load_audit_log(self, limit: int = 50) -> list[dict]:
        """Returns the last N config changes."""
        try:
            with self.db._cursor() as cur:
                cur.execute("""
                    SELECT path, old_value, new_value, ts, source
                    FROM config_audit
                    ORDER BY ts DESC LIMIT ?
                """, (limit,))
                return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []
