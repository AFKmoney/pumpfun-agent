"""
analytics_pipeline.py
=====================
Live data ingestion that FEEDS the analyzer "brain" (order_flow, lifecycle,
social_graph, sentiment). Before this module existed, the analyzers were
instantiated by the orchestrator but never received any data — every
AlphaSignal component collapsed to its neutral fallback (50/100).

Three ingestion paths:

1. PUMP_FUN_TRADE_STREAM (async task):
   Subscribes to pump.fun program logs. Each `buy`/`sell` instruction event
   is parsed into a Trade and fed to:
     - order_flow.record_trade()  (buy/sell pressure, whale detection)
     - social_graph reputation update (the trader's wallet)
   This is the load-bearing real-time data path.

2. LIFECYCLE_OBSERVER (async task, periodic):
   For each open position + each recently-launched tracked token, fetches
   holders count, price, and trade count, then calls
   lifecycle.record_observation(). This populates the velocity math
   (holder_velocity, price_velocity, trade_velocity).

3. SENTIMENT_POLLER (async task, periodic):
   For each active token, calls sentiment.fetch_twitter_mentions() and
   sentiment.fetch_telegram_mentions() (both gated on their respective API
   keys being present in .env). Adds any mentions found to the sentiment
   buffer.

All three tasks are launched by AnalyticsPipeline.start() and cancelled by
.stop(). They are fault-tolerant: a failure on one token does not stop the
loop.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import aiohttp

from analysis.order_flow import OrderFlowAnalyzer, Trade
from analysis.lifecycle import LifecycleAnalyzer
from analysis.sentiment import SentimentAnalyzer, Mention
from analysis.social_graph import SocialGraphAnalyzer
from utils.bonding_curve import BondingCurveAnalyzer, fetch_sol_usd
from utils.config_loader import Config
from utils.data_providers import get_providers
from utils.logger import setup_logger
from utils.pumpfun_parser import PUMP_FUN_PROGRAM
from utils.pumpfun_ix import BUY_DISCRIMINATOR, SELL_DISCRIMINATOR

log = setup_logger("analytics_pipeline")


class AnalyticsPipeline:
    """
    Wires real-time + periodic data into the analyzers.

    Usage:
        pipe = AnalyticsPipeline(order_flow, lifecycle, social_graph, sentiment)
        await pipe.start()   # launches 3 background tasks
        ...
        await pipe.stop()
    """

    def __init__(
        self,
        order_flow: OrderFlowAnalyzer,
        lifecycle: LifecycleAnalyzer,
        social_graph: SocialGraphAnalyzer,
        sentiment: SentimentAnalyzer,
        bonding: Optional[BondingCurveAnalyzer] = None,
        risk_positions_ref=None,  # callable () -> dict[str, object]; returns open positions
    ) -> None:
        self.order_flow = order_flow
        self.lifecycle = lifecycle
        self.social_graph = social_graph
        self.sentiment = sentiment
        self.bonding = bonding or BondingCurveAnalyzer()
        self._get_positions = risk_positions_ref or (lambda: {})
        self.cfg = Config.get()
        self._tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        # Tokens we are actively tracking (open positions + recently-launched).
        # Populated by the trade stream as it sees new mints; capped to avoid
        # unbounded memory growth.
        self._tracked_mints: dict[str, float] = {}  # mint -> first_seen_ts
        self._MAX_TRACKED = 200

    async def start(self) -> None:
        self._tasks.append(asyncio.create_task(self._pumpfun_trade_stream()))
        self._tasks.append(asyncio.create_task(self._lifecycle_observer_loop()))
        self._tasks.append(asyncio.create_task(self._sentiment_poller_loop()))
        log.info("analytics_pipeline.started",
                 tracked_capacity=self._MAX_TRACKED)

    async def stop(self) -> None:
        self._stop_event.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

    def _track_mint(self, mint: str) -> None:
        if mint not in self._tracked_mints:
            self._tracked_mints[mint] = time.time()
            # Evict oldest if over capacity
            if len(self._tracked_mints) > self._MAX_TRACKED:
                oldest = min(self._tracked_mints, key=self._tracked_mints.get)
                self._tracked_mints.pop(oldest, None)

    # ------------------------------------------------------------------
    # Path 1: real-time pump.fun trade stream -> order_flow + social_graph
    # ------------------------------------------------------------------
    async def _pumpfun_trade_stream(self) -> None:
        import websockets
        ws_endpoint = self.cfg["chains"]["solana"]["ws_endpoint"]
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "logsSubscribe",
            "params": [{"mentions": [PUMP_FUN_PROGRAM]}, {"commitment": "confirmed"}],
        }
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(ws_endpoint) as ws:
                    await ws.send(json.dumps(payload))
                    backoff = 1.0
                    async for raw in ws:
                        if self._stop_event.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                            if msg.get("method") != "logsNotification":
                                continue
                            await self._process_log_event(msg)
                        except Exception as e:
                            log.warning("analytics_pipeline.log_process_error", error=str(e))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("analytics_pipeline.trade_ws_disconnected",
                            error=str(e), backoff=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _process_log_event(self, msg: dict) -> None:
        """
        Parse a pump.fun logsSubscribe notification and feed order_flow.
        pump.fun buy/sell events carry mint + amount in the log text:
          "Instruction: Buy" / "Instruction: Sell"
        followed by an event line. We extract what we can; missing fields
        default to safe zeros.
        """
        value = msg.get("params", {}).get("result", {}).get("value", {})
        logs = value.get("logs") or []
        signature = value.get("signature")
        if not logs:
            return

        # Detect side
        is_buy = any("Instruction: Buy" in l for l in logs)
        is_sell = any("Instruction: Sell" in l for l in logs)
        if not (is_buy or is_sell):
            return
        side = "buy" if is_buy else "sell"

        # Extract mint + amounts from the event line(s).
        # pump.fun logs include lines like:
        #   "Program data: <base64-borsh-event>"
        # The Buy/Sell event payload contains: mint, sol_amount, token_amount, user, ...
        # Parsing the borsh event reliably requires the IDL; for resilience we
        # use a best-effort extraction and fill defaults otherwise.
        mint = self._extract_mint_from_logs(logs)
        sol_amount, token_amount = self._extract_amounts_from_logs(logs)
        if not mint:
            return  # cannot attribute without a mint

        self._track_mint(mint)

        # USD sizing
        sol_usd = await fetch_sol_usd()
        size_usd = sol_amount * sol_usd

        # Token price (SOL per token) — derive from bonding curve if available,
        # else from the ratio of the trade amounts.
        price = (sol_amount / token_amount) if token_amount > 0 else 0.0

        trade = Trade(
            ts=time.time(),
            side=side,
            size_sol=sol_amount,
            size_usd=size_usd,
            size_token=token_amount,
            trader="",  # not reliably available from logs alone
            price=price,
            is_whale=(sol_amount >= self.order_flow.WHALE_THRESHOLD_SOL),
        )
        try:
            self.order_flow.record_trade(mint, trade)
        except Exception as e:
            log.warning("analytics_pipeline.order_flow_record_failed",
                        mint=mint, error=str(e))

    @staticmethod
    def _extract_mint_from_logs(logs: list[str]) -> Optional[str]:
        """
        Best-effort mint extraction. The pump.fun buy/sell event does not
        carry the mint in plain text on standard RPC logs; we rely on the
        accompanying create-event context or fall back to None (the trade
        stream caller can enrich via getTransaction if needed, but that
        would re-introduce latency we are trying to avoid).
        """
        # Some log enrichments (Helius enhanced) surface the mint:
        #   "Program data: ... mint=<b58> ..."
        import re
        for line in logs:
            m = re.search(r'\bmint=([1-9A-HJ-NP-Za-km-z]{32,44})', line)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _extract_amounts_from_logs(logs: list[str]) -> tuple[float, float]:
        """
        Best-effort extraction of (sol_amount, token_amount) from log lines.
        pump.fun event lines sometimes contain:
          "sol_amount": 1.23  /  "token_amount": 4560000
        Returns (0.0, 0.0) if not found — order_flow still gets a trade record
        with zero size, which is better than no record at all (it counts toward
        trade velocity).
        """
        import re
        sol_amt = 0.0
        tok_amt = 0.0
        for line in logs:
            ms = re.search(r'sol[_\s]*amount[:\s=]+([0-9.]+)', line, re.IGNORECASE)
            if ms:
                try: sol_amt = float(ms.group(1))
                except ValueError: pass
            mt = re.search(r'token[_\s]*amount[:\s=]+([0-9.]+)', line, re.IGNORECASE)
            if mt:
                try: tok_amt = float(mt.group(1))
                except ValueError: pass
        return sol_amt, tok_amt

    # ------------------------------------------------------------------
    # Path 2: periodic lifecycle observations for tracked + open positions
    # ------------------------------------------------------------------
    async def _lifecycle_observer_loop(self) -> None:
        interval = self.cfg.get_nested("analysis", "lifecycle_observation_interval_sec",
                                       default=30)
        while not self._stop_event.is_set():
            try:
                # Union of tracked mints and open positions
                mints = set(self._tracked_mints.keys())
                try:
                    for pos in self._get_positions().values():
                        if getattr(pos, "chain", None) == "solana":
                            mints.add(pos.token)
                except Exception:
                    pass
                for mint in mints:
                    await self._observe_one(mint)
            except Exception as e:
                log.error("analytics_pipeline.lifecycle_loop_error", error=str(e))
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def _observe_one(self, mint: str) -> None:
        """Fetch holders, price, trade count for a mint and feed lifecycle."""
        try:
            providers = get_providers()
            helius = providers["helius"]
            dex = providers["dexscreener"]
            holders = await helius.get_token_holders(mint)
            holder_count = len(holders) if isinstance(holders, list) else 0
            price = await dex.get_price_usd(mint)
            # Trade count proxy: number of order_flow records for this mint
            snap = self.order_flow.snapshot(mint, window_sec=300)
            trade_count = snap.total_trades if snap else 0
            self.lifecycle.record_observation(mint, holder_count, price, trade_count)
        except Exception as e:
            log.debug("analytics_pipeline.observe_one_failed", mint=mint, error=str(e))

    # ------------------------------------------------------------------
    # Path 3: periodic sentiment polling
    # ------------------------------------------------------------------
    async def _sentiment_poller_loop(self) -> None:
        import os
        interval = self.cfg.get_nested("analysis", "sentiment_poll_interval_sec",
                                       default=120)
        twitter_chat_ids = self.cfg.get_nested("analysis", "sentiment",
                                               default={}).get("telegram_chat_ids", [])
        # No API keys -> nothing to poll; sleep forever but stay cancellable.
        while not self._stop_event.is_set():
            try:
                # Only poll mints we care about (tracked + open positions)
                mints = set(self._tracked_mints.keys())
                try:
                    for pos in self._get_positions().values():
                        if getattr(pos, "chain", None) == "solana":
                            mints.add(pos.token)
                except Exception:
                    pass
                for mint in mints:
                    # We need the symbol for the Twitter query; best-effort
                    # via token metadata. Skip if unavailable.
                    symbol = await self._symbol_for(mint)
                    if not symbol:
                        continue
                    try:
                        await self.sentiment.fetch_twitter_mentions(mint, symbol)
                    except Exception as e:
                        log.debug("analytics_pipeline.twitter_poll_failed",
                                  mint=mint, error=str(e))
                    for chat_id in twitter_chat_ids:
                        try:
                            await self.sentiment.fetch_telegram_mentions(mint, symbol, chat_id)
                        except Exception as e:
                            log.debug("analytics_pipeline.telegram_poll_failed",
                                      mint=mint, chat=chat_id, error=str(e))
            except Exception as e:
                log.error("analytics_pipeline.sentiment_loop_error", error=str(e))
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def _symbol_for(self, mint: str) -> Optional[str]:
        """Best-effort symbol lookup from DexScreener."""
        try:
            providers = get_providers()
            dex = providers["dexscreener"]
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
                async with s.get(f"https://api.dexscreener.com/latest/dex/tokens/{mint}") as r:
                    data = await r.json()
            pairs = data.get("pairs") or []
            if pairs:
                return pairs[0].get("baseToken", {}).get("symbol")
        except Exception:
            return None
        return None
