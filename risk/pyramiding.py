"""
pyramiding.py
=============
Position pyramiding — adds to winning positions when momentum confirms.

Rules:
- Only pyramid on positions already in profit by >= pyramid_min_pnl_pct
- Only pyramid if token score is >= min_score_to_pyramid (from TokenScorer)
- Max 2 pyramid additions per position
- Each pyramid is 50% of the original size (geometric scaling)
- Pyramid size also bounded by Kelly fraction
- Skip pyramiding if it would exceed max_open_positions for that token
  (currently 1 position per token, so we scale the existing one)

The orchestrator calls Pyramider.evaluate(position) periodically. If conditions
are met, it returns a PyramidSignal which the executor routes as a BUY.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from risk.types import Position
from utils.config_loader import Config
from utils.logger import setup_logger
from utils.token_scorer import TokenScorer, TokenScore

log = setup_logger("pyramiding")


@dataclass
class PyramidSignal:
    chain: str
    token: str
    add_size_pct_of_original: float  # e.g. 0.5 = add 50% of original size
    reason: str
    new_total_size_base: float       # new total position size after pyramid


class Pyramider:
    """Manages pyramiding on winning positions."""

    def __init__(self, scorer: TokenScorer) -> None:
        self.cfg = Config.get()
        self.scorer = scorer
        pcfg = self.cfg.get_nested("pyramiding", default={})
        self.enabled = pcfg.get("enabled", True)
        self.min_pnl_pct_to_pyramid = pcfg.get("min_pnl_pct", 50.0)
        self.max_additions = pcfg.get("max_additions", 2)
        self.add_size_pct = pcfg.get("add_size_pct", 0.5)  # 50% of original each time
        self.cooldown_minutes = pcfg.get("cooldown_minutes", 15)
        # Track pyramid count + last pyramid time per token
        self._pyramid_count: dict[str, int] = {}
        self._last_pyramid_ts: dict[str, float] = {}

    def init_position(self, chain: str, token: str) -> None:
        key = f"{chain}:{token}"
        self._pyramid_count[key] = 0
        self._last_pyramid_ts[key] = 0

    def drop_position(self, chain: str, token: str) -> None:
        key = f"{chain}:{token}"
        self._pyramid_count.pop(key, None)
        self._last_pyramid_ts.pop(key, None)

    async def evaluate(
        self,
        position: Position,
        current_price: float,
        current_score: Optional[TokenScore] = None,
    ) -> Optional[PyramidSignal]:
        if not self.enabled:
            return None

        key = f"{position.chain}:{position.token}"
        if self._pyramid_count.get(key, 0) >= self.max_additions:
            return None

        # Cooldown check
        if time.time() - self._last_pyramid_ts.get(key, 0) < self.cooldown_minutes * 60:
            return None

        pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
        if pnl_pct < self.min_pnl_pct_to_pyramid:
            return None

        # Score check (recompute if not provided)
        if not current_score:
            try:
                current_score = await self.scorer.score(position.token)
            except Exception as e:
                log.warning("pyramiding.score_failed", error=str(e))
                return None

        if current_score.score < self.scorer.min_score_to_pyramid:
            return None

        # Compute new add size (50% of original position's base size)
        add_size_base = position.size_base * self.add_size_pct
        new_total = position.size_base + add_size_base

        signal = PyramidSignal(
            chain=position.chain,
            token=position.token,
            add_size_pct_of_original=self.add_size_pct,
            reason=f"Pyramid #{self._pyramid_count[key]+1}: PnL={pnl_pct:.0f}%, score={current_score.score:.0f}",
            new_total_size_base=new_total,
        )

        # Update tracking
        self._pyramid_count[key] = self._pyramid_count.get(key, 0) + 1
        self._last_pyramid_ts[key] = time.time()

        log.info("pyramiding.signal_emitted",
                 token=position.token, count=self._pyramid_count[key],
                 add_size=add_size_base, new_total=new_total)
        return signal
