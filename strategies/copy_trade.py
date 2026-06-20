"""
copy_trade.py
=============
Copy-trading strategy:
1. Maintain a watchlist of profitable wallets (configurable in config.yaml).
2. On each new signature from a target wallet, decode the swap.
3. If it's a buy of a pump.fun token we don't hold, replicate it.
4. If it's a sell, optionally replicate the exit.

Optional pre-trade filter: only copy if the target wallet's 30d PnL > threshold.
"""
from __future__ import annotations

import asyncio
from typing import Iterable

from chains.base_chain import BaseChainAdapter
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("copy_trade")


class CopyTradeStrategy(BaseStrategy):
    name = "copy_trade"

    def __init__(self, adapter: BaseChainAdapter, signal_queue: asyncio.Queue) -> None:
        super().__init__(adapter)
        self.queue = signal_queue
        self.cfg = Config.get().get_nested("strategies", "copy_trade", default={})
        self.target_wallets: list[str] = self.cfg.get("target_wallets", [])
        self.follow_delay_ms_max = self.cfg.get("follow_delay_ms_max", 3000)
        self.max_follow_size = self.cfg.get("max_follow_size_sol", 0.2)

    async def run(self) -> None:
        log.info("copy_trade.started", chain=self.adapter.chain_name, targets=self.target_wallets)
        tasks = [asyncio.create_task(self._tail_wallet(w)) for w in self.target_wallets]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            log.info("copy_trade.cancelled")
            raise

    async def _tail_wallet(self, wallet: str) -> None:
        async for event in self.adapter.watch_wallet_activity(wallet):
            try:
                sig = await self.evaluate(event)
                if sig and sig.signal_type != SignalType.HOLD:
                    await self.queue.put(sig)
            except Exception as e:
                log.error("copy_trade.evaluate_failed", wallet=wallet, error=str(e))

    async def evaluate(self, event: dict) -> Signal | None:
        tx = event.get("tx")
        if not tx:
            return None
        # In production: decode instruction / log to extract swap details.
        # We emit a BUY signal placeholder.
        token = "PARSED_TOKEN_MINT"
        return Signal(
            strategy=self.name,
            chain=event["chain"],
            token_address=token,
            signal_type=SignalType.BUY,
            suggested_size_pct=0.01,
            confidence=0.55,
            reason=f"Wallet {event['wallet'][:8]}... bought",
            stop_loss_pct=20.0,
            take_profit_pct=50.0,
        )
