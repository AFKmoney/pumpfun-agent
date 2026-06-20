"""
migration_detector.py
=====================
Detects pump.fun → Raydium migration events.

When pump.fun migrates a token, it:
1. Burns the bonding curve account.
2. Creates a Raydium LP pool with the real reserves (~85 SOL + remaining tokens).
3. The token suddenly has real DEX liquidity → price typically spikes 50-500%
   within 5-30 minutes as wider market can now trade it.

This module watches for migration signals:
- Bonding curve completion % jumps from <95% to "complete=true"
- A new Raydium pool is created with the token
- Liquidity USD on DexScreener suddenly jumps >5x

When detected, we:
- Hold the position longer (extend max hold time)
- Tighten trailing stop (lock in the pump)
- Optionally pyramid (add to position) at the start of the pump

The actual exit logic lives in SmartExitManager; this module just signals
"migration detected" so other modules can react.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Awaitable

from utils.bonding_curve import BondingCurveAnalyzer, BondingCurveState
from utils.config_loader import Config
from utils.data_providers import get_providers
from utils.logger import setup_logger

log = setup_logger("migration_detector")


@dataclass
class MigrationEvent:
    chain: str
    token: str
    block_time: float
    liq_before: float
    liq_after: float
    pump_pct: float


class MigrationDetector:
    """Polls open positions for migration events."""

    def __init__(self) -> None:
        self.cfg = Config.get()
        self.bonding = BondingCurveAnalyzer()
        self._last_states: dict[str, BondingCurveState] = {}  # token -> last BC state
        self._callbacks: list[Callable[[MigrationEvent], Awaitable[None]]] = []
        self._migrated_tokens: set[str] = set()

    def register_callback(self, cb: Callable[[MigrationEvent], Awaitable[None]]) -> None:
        self._callbacks.append(cb)

    async def check_position(self, chain: str, token: str) -> None:
        """Check a single open position for migration."""
        if token in self._migrated_tokens:
            return  # already migrated, no need to re-check

        providers = get_providers()
        try:
            current_liq = await providers["dexscreener"].get_liquidity_usd(token)
            bc_state = await self.bonding.fetch_state(token)
        except Exception as e:
            log.warning("migration.check_failed", token=token, error=str(e))
            return

        if not bc_state:
            return

        last = self._last_states.get(token)
        self._last_states[token] = bc_state

        if not last:
            return  # first check, just store baseline

        # Migration signals:
        # 1. Bonding curve complete flag flipped to True
        # 2. Liquidity jumped 5x+
        # 3. Real SOL reserves hit 85+
        migrated = False
        if last.completion_pct < 95 and bc_state.completion_pct >= 100:
            migrated = True
            log.info("migration.bc_complete", token=token)
        elif current_liq > 0 and last.current_price_usd > 0:
            # Compare liquidity ratios
            last_liq_estimate = last.real_sol_reserves * 200  # rough USD
            if current_liq > last_liq_estimate * 5 and current_liq > 50_000:
                migrated = True
                log.info("migration.liq_jump", token=token,
                         before=last_liq_estimate, after=current_liq)

        if migrated:
            self._migrated_tokens.add(token)
            event = MigrationEvent(
                chain=chain, token=token,
                block_time=time.time(),
                liq_before=last.real_sol_reserves * 200,
                liq_after=current_liq,
                pump_pct=((current_liq - last.real_sol_reserves * 200) /
                          max(1, last.real_sol_reserves * 200) * 100),
            )
            log.warning("migration.DETECTED",
                        token=token, pump_pct=event.pump_pct,
                        liq_before=event.liq_before, liq_after=event.liq_after)
            for cb in self._callbacks:
                try:
                    await cb(event)
                except Exception as e:
                    log.error("migration.callback_failed", error=str(e))
