"""
sniping.py
==========
Sniper strategy: detect new pump.fun token launches and buy within milliseconds.

Flow:
1. Subscribe to pump.fun program logs via the adapter's `watch_new_tokens()`.
2. The adapter now emits events that may already carry the parsed mint /
   bonding_curve / user (low-latency path, no getTransaction round-trip).
3. If the event lacks the parsed mint, fall back to fetching the tx.
4. Run anti-rugpull pre-flight checks.
5. Fire BUY signal carrying the dev wallet (in metadata) so the executor can
   arm the DevTracker on the resulting position.

Latency target: <500ms from launch detection to buy tx submission.
"""
from __future__ import annotations

import asyncio
import time

from chains.base_chain import BaseChainAdapter
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from utils.config_loader import Config
from utils.logger import setup_logger
from utils.pumpfun_parser import PUMP_FUN_PROGRAM, fetch_create_event_from_signature

log = setup_logger("sniping")


class SnipingStrategy(BaseStrategy):
    name = "sniping"

    def __init__(self, adapter: BaseChainAdapter, signal_queue: asyncio.Queue, anti_rug=None) -> None:
        super().__init__(adapter)
        self.queue = signal_queue
        self.anti_rug = anti_rug  # injected by orchestrator (so we share the same instance)
        self.cfg = Config.get().get_nested("strategies", "sniping", default={})
        self.scan_interval = self.cfg.get("new_pool_scan_interval_sec", 2)
        self.max_delay_ms = self.cfg.get("max_buy_delay_ms", 1500)
        self.min_initial_liq = self.cfg.get("min_initial_liquidity_usd", 5000)
        self.max_buy_tax_bps = self.cfg.get("max_buy_tax_bps", 500)

    async def run(self) -> None:
        log.info("sniping.started", chain=self.adapter.chain_name, program=PUMP_FUN_PROGRAM)
        try:
            async for event in self.adapter.watch_new_tokens():
                t0 = time.monotonic()
                sig = await self.evaluate(event)
                if sig and sig.signal_type == SignalType.BUY:
                    await self.queue.put(sig)
                    log.info("sniping.signal_emitted",
                             chain=sig.chain, token=sig.token_address,
                             elapsed_ms=int((time.monotonic() - t0) * 1000))
        except asyncio.CancelledError:
            log.info("sniping.cancelled")
            raise

    async def evaluate(self, event: dict) -> Signal | None:
        chain = event.get("chain", self.adapter.chain_name)
        signature = event.get("signature")
        if not signature:
            return None

        mint = event.get("mint")
        bonding_curve = event.get("bonding_curve")
        dev_wallet = event.get("user")
        symbol = ""
        name = ""

        # 1. If the adapter could not parse the mint from logs, fetch the tx
        if not mint:
            try:
                create_event = await fetch_create_event_from_signature(self.adapter, signature)
            except Exception as e:
                log.warning("sniping.fetch_create_failed", sig=signature, error=str(e))
                return None
            if not create_event:
                return None
            mint = create_event.mint
            bonding_curve = bonding_curve or create_event.bonding_curve
            dev_wallet = dev_wallet or create_event.user
            symbol = create_event.symbol
            name = create_event.name

        log.info("sniping.new_launch_detected",
                 mint=mint, symbol=symbol or "?", name=(name or "?")[:40],
                 source=event.get("source", "fetch"))

        # 2. Anti-rugpull gate
        if self.anti_rug is not None:
            try:
                verdict = await self.anti_rug.check(mint)
                if not verdict.is_safe:
                    log.info("sniping.rugpull_blocked", mint=mint, reasons=verdict.reasons)
                    return None
            except Exception as e:
                log.warning("sniping.anti_rug_check_failed", mint=mint, error=str(e))
                return None

        # 3. Emit BUY signal carrying the dev wallet for the executor's DevTracker
        return Signal(
            strategy=self.name,
            chain=chain,
            token_address=mint,
            signal_type=SignalType.BUY,
            suggested_size_pct=0.02,   # 2% of allocated capital
            confidence=0.65,
            reason=f"New pump.fun launch: {symbol or '?'} ({(name or '?')[:30]})",
            stop_loss_pct=15.0,
            take_profit_pct=80.0,
            metadata={
                "dev_wallet": dev_wallet,
                "bonding_curve": bonding_curve,
                "token_symbol": symbol,
                "token_name": name,
            },
        )
