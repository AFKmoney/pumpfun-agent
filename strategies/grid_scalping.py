"""
grid_scalping.py
================
Grid / scalping strategy on a fixed list of established tokens.
Maintains N buy orders and N sell orders spaced grid_spacing_pct apart.
Each fill triggers the opposite side order.

This is intentionally a simplified version suitable for bonding-curve tokens
that have migrated to Raydium/Uniswap. For pure pump.fun (bonding curve),
grid logic is approximated by periodic re-balancing.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

from chains.base_chain import BaseChainAdapter
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("grid_scalping")


@dataclass
class GridLevel:
    token: str
    side: str          # "buy" | "sell"
    price: float
    size_sol: float
    filled: bool = False


class GridScalpingStrategy(BaseStrategy):
    name = "grid_scalping"

    def __init__(self, adapter: BaseChainAdapter, signal_queue: asyncio.Queue) -> None:
        super().__init__(adapter)
        self.queue = signal_queue
        self.cfg = Config.get().get_nested("strategies", "grid_scalping", default={})
        self.target_tokens: list[str] = self.cfg.get("target_tokens", [])
        self.levels: int = self.cfg.get("grid_levels", 8)
        self.spacing_pct: float = self.cfg.get("grid_spacing_pct", 1.5)
        self.order_size_sol: float = self.cfg.get("order_size_sol", 0.02)
        self._grids: dict[str, list[GridLevel]] = {}
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        log.info("grid_scalping.started", tokens=self.target_tokens, levels=self.levels)
        try:
            for token in self.target_tokens:
                mid = await self.adapter.get_price(token)
                if mid <= 0:
                    continue
                self._grids[token] = self._build_grid(token, mid)
            while not self._stop_event.is_set():
                for token, grid in self._grids.items():
                    await self._tick(token, grid)
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            log.info("grid_scalping.cancelled")
            raise

    def _build_grid(self, token: str, mid_price: float) -> list[GridLevel]:
        levels: list[GridLevel] = []
        for i in range(1, self.levels + 1):
            buy_price = mid_price * (1 - (self.spacing_pct / 100) * i)
            sell_price = mid_price * (1 + (self.spacing_pct / 100) * i)
            levels.append(GridLevel(token, "buy", buy_price, self.order_size_sol))
            levels.append(GridLevel(token, "sell", sell_price, self.order_size_sol))
        return levels

    async def _tick(self, token: str, grid: list[GridLevel]) -> None:
        price = await self.adapter.get_price(token)
        if price <= 0:
            return
        for lvl in grid:
            if lvl.filled:
                continue
            if lvl.side == "buy" and price <= lvl.price:
                await self.queue.put(Signal(
                    strategy=self.name, chain=self.adapter.chain_name, token_address=token,
                    signal_type=SignalType.BUY, suggested_size_pct=0.005, confidence=0.5,
                    reason=f"Grid buy @ {lvl.price}", target_price=lvl.price,
                ))
                lvl.filled = True
            elif lvl.side == "sell" and price >= lvl.price:
                await self.queue.put(Signal(
                    strategy=self.name, chain=self.adapter.chain_name, token_address=token,
                    signal_type=SignalType.SELL, suggested_size_pct=0.005, confidence=0.5,
                    reason=f"Grid sell @ {lvl.price}", target_price=lvl.price,
                ))
                lvl.filled = True

    async def evaluate(self, *args, **kwargs) -> Signal:
        # Grid strategy emits signals directly via _tick; not used by orchestrator.
        return Signal(self.name, self.adapter.chain_name, "", SignalType.HOLD, 0, 0, "n/a")
