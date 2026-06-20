"""
lifecycle.py
============
Token lifecycle analyzer.

Pump.fun tokens follow predictable lifecycle stages:
1. BIRTH (0-5 min): bonding curve launches. Extreme volatility. Most scams.
2. INFANT (5-30 min): early holders establish. Survives = real project, dies = rug.
3. ADOLESCENT (30 min - 4h): price discovery. Survivors gain traction.
4. MATURE (4-24h): either dying or preparing migration.
5. MIGRATED (24h+): on Raydium. Now trades like a normal token.

Each stage has different optimal strategy:
- BIRTH: high risk, high reward. Only snipe with very tight SL.
- INFANT: best risk/reward. Buy if holder count growing + dev hasn't sold.
- ADOLESCENT: momentum trading. Use RSI + volume signals.
- MATURE: avoid unless nearing migration.
- MIGRATED: classic momentum strategy, lower size.

This module classifies a token's current stage and recommends the right
strategy + sizing multiplier.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from utils.bonding_curve import BondingCurveAnalyzer, BondingCurveState
from utils.config_loader import Config
from utils.data_providers import get_providers
from utils.logger import setup_logger

log = setup_logger("lifecycle")


class LifecycleStage(str, Enum):
    BIRTH = "BIRTH"                 # 0-5 min
    INFANT = "INFANT"               # 5-30 min
    ADOLESCENT = "ADOLESCENT"       # 30 min - 4h
    MATURE = "MATURE"               # 4-24h
    MIGRATED = "MIGRATED"           # >24h or bonding curve complete
    DEAD = "DEAD"                   # no activity for 1h+


@dataclass
class LifecycleAssessment:
    mint: str
    stage: LifecycleStage
    age_minutes: float
    bonding_curve_pct: float
    holder_velocity: float          # holders gained per minute (last 5 min)
    price_velocity_pct: float       # % price change per minute (last 5 min)
    trade_velocity: float           # trades per minute (last 5 min)
    is_dying: bool                  # activity dropping off
    is_breaking_out: bool           # sudden surge in activity
    size_multiplier: float          # recommended position size multiplier (0.0 - 2.0)
    recommended_strategy: str       # "snipe" | "momentum" | "avoid" | "hold"
    risk_level: str                 # "low" | "medium" | "high" | "extreme"
    notes: list[str]


class LifecycleAnalyzer:
    """Classifies token lifecycle stage and recommends strategy."""

    # Stage boundaries (minutes)
    BIRTH_MAX_MIN = 5
    INFANT_MAX_MIN = 30
    ADOLESCENT_MAX_MIN = 240
    MATURE_MAX_MIN = 1440

    def __init__(self) -> None:
        self.cfg = Config.get()
        self.bonding = BondingCurveAnalyzer()
        # Per-token history of (ts, holders, price, trades_count)
        self._history: dict[str, list[tuple[float, int, float, int]]] = {}

    def record_observation(self, mint: str, holders: int, price: float, trade_count: int) -> None:
        """Called periodically (e.g., every 30s) for each tracked token."""
        if mint not in self._history:
            self._history[mint] = []
        self._history[mint].append((time.time(), holders, price, trade_count))
        # Keep last 24h
        cutoff = time.time() - 86400
        self._history[mint] = [(t, h, p, c) for t, h, p, c in self._history[mint] if t > cutoff]

    async def assess(self, mint: str, created_at_ts: Optional[float] = None) -> LifecycleAssessment:
        """Compute lifecycle assessment for a token."""
        now = time.time()
        # Age
        if created_at_ts:
            age_minutes = (now - created_at_ts) / 60
        else:
            # Estimate from history
            history = self._history.get(mint, [])
            if history:
                age_minutes = (now - history[0][0]) / 60
            else:
                age_minutes = 0

        # Bonding curve state
        bc_state = await self.bonding.fetch_state(mint)
        bc_pct = bc_state.completion_pct if bc_state else 0

        # Velocities from history
        history = self._history.get(mint, [])
        holder_vel = 0.0
        price_vel = 0.0
        trade_vel = 0.0
        if len(history) >= 2:
            # Last 5 minutes
            recent = [(t, h, p, c) for t, h, p, c in history if now - t <= 300]
            if len(recent) >= 2:
                dt = recent[-1][0] - recent[0][0]
                if dt > 0:
                    holder_vel = (recent[-1][1] - recent[0][1]) / (dt / 60)
                    price_vel = ((recent[-1][2] - recent[0][2]) / max(recent[0][2], 1e-9)) / (dt / 60) * 100
                    trade_vel = (recent[-1][3] - recent[0][3]) / (dt / 60)

        # Stage classification
        if bc_state and bc_state.completion_pct >= 100:
            stage = LifecycleStage.MIGRATED
        elif age_minutes <= self.BIRTH_MAX_MIN:
            stage = LifecycleStage.BIRTH
        elif age_minutes <= self.INFANT_MAX_MIN:
            stage = LifecycleStage.INFANT
        elif age_minutes <= self.ADOLESCENT_MAX_MIN:
            stage = LifecycleStage.ADOLESCENT
        elif age_minutes <= self.MATURE_MAX_MIN:
            stage = LifecycleStage.MATURE
        else:
            stage = LifecycleStage.DEAD

        # Dying detection: activity dropping
        is_dying = holder_vel < 0 or (trade_vel < 0.5 and age_minutes > 30)
        # Breakout detection: sudden surge
        is_breaking_out = holder_vel > 5 or trade_vel > 10 or price_vel > 20

        # Stage-specific recommendations
        size_mult = 1.0
        recommended = "momentum"
        risk = "medium"
        notes: list[str] = []

        if stage == LifecycleStage.BIRTH:
            size_mult = 0.5
            recommended = "snipe"
            risk = "extreme"
            notes.append("BIRTH: extreme risk, use tight SL")
            if is_breaking_out:
                size_mult = 0.7
                notes.append("Breakout detected — size up slightly")
        elif stage == LifecycleStage.INFANT:
            size_mult = 1.5
            recommended = "momentum"
            risk = "high"
            notes.append("INFANT: best risk/reward window")
            if holder_vel > 2:
                size_mult = 2.0
                notes.append("Strong holder growth — full size")
        elif stage == LifecycleStage.ADOLESCENT:
            size_mult = 1.0
            recommended = "momentum"
            risk = "medium"
            notes.append("ADOLESCENT: standard momentum strategy")
        elif stage == LifecycleStage.MATURE:
            size_mult = 0.3
            recommended = "avoid"
            risk = "high"
            notes.append("MATURE: stale, avoid unless migrating soon")
            if bc_pct > 90:
                size_mult = 0.8
                recommended = "hold"
                notes.append("Near migration — consider position")
        elif stage == LifecycleStage.MIGRATED:
            size_mult = 1.0
            recommended = "momentum"
            risk = "medium"
            notes.append("MIGRATED: normal trading")
        else:  # DEAD
            size_mult = 0.0
            recommended = "avoid"
            risk = "high"
            notes.append("DEAD: no activity")

        if is_dying and stage != LifecycleStage.DEAD:
            size_mult *= 0.5
            notes.append("Activity dying — reduce size")

        return LifecycleAssessment(
            mint=mint,
            stage=stage,
            age_minutes=age_minutes,
            bonding_curve_pct=bc_pct,
            holder_velocity=holder_vel,
            price_velocity_pct=price_vel,
            trade_velocity=trade_vel,
            is_dying=is_dying,
            is_breaking_out=is_breaking_out,
            size_multiplier=size_mult,
            recommended_strategy=recommended,
            risk_level=risk,
            notes=notes,
        )
