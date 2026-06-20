"""
momentum.py
===========
Momentum / breakout strategy:
- Scan tokens on DexScreener every N seconds.
- Filter: 5m volume > threshold, 5m price change > threshold.
- Compute RSI(14) on the 1m klines.
- Emit BUY if RSI < oversold AND price change > threshold (reversal breakout).
- Emit SELL (close existing positions) if RSI > overbought.
"""
from __future__ import annotations

import asyncio
from collections import deque

import aiohttp

from chains.base_chain import BaseChainAdapter
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("momentum")


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def __init__(self, adapter: BaseChainAdapter, signal_queue: asyncio.Queue) -> None:
        super().__init__(adapter)
        self.queue = signal_queue
        self.cfg = Config.get().get_nested("strategies", "momentum", default={})
        self.scan_interval = self.cfg.get("scan_interval_sec", 15)
        self.min_volume = self.cfg.get("min_volume_5m_usd", 50_000)
        self.min_price_change = self.cfg.get("min_price_change_5m_pct", 15.0)
        self.rsi_period = self.cfg.get("rsi_period", 14)
        self.rsi_oversold = self.cfg.get("rsi_oversold", 30)
        self.rsi_overbought = self.cfg.get("rsi_overbought", 75)
        self._session: aiohttp.ClientSession | None = None
        self._price_histories: dict[str, deque] = {}

    async def run(self) -> None:
        self._session = aiohttp.ClientSession()
        try:
            while True:
                await self._scan_once()
                await asyncio.sleep(self.scan_interval)
        except asyncio.CancelledError:
            log.info("momentum.cancelled")
            raise
        finally:
            if self._session:
                await self._session.close()

    async def _scan_once(self) -> None:
        # DexScreener endpoint: top gainers on Solana/Base
        url = "https://api.dexscreener.com/latest/dex/search?q=pump.fun"
        try:
            async with self._session.get(url) as r:
                data = await r.json()
        except Exception as e:
            log.warning("momentum.fetch_failed", error=str(e))
            return
        for pair in data.get("pairs", [])[:50]:
            try:
                chain = pair.get("chainId", "").lower()
                if chain != self.adapter.chain_name:
                    continue
                vol5m = float(pair.get("volume", {}).get("m5", 0))
                change5m = float(pair.get("priceChange", {}).get("m5", 0))
                price_usd = float(pair.get("priceUsd", 0))
                token = pair.get("baseToken", {}).get("address", "")
                if not token or vol5m < self.min_volume or change5m < self.min_price_change:
                    continue

                hist = self._price_histories.setdefault(token, deque(maxlen=self.rsi_period * 5))
                hist.append(price_usd)
                rsi = self._compute_rsi(list(hist)) if len(hist) >= self.rsi_period + 1 else None

                sig = await self.evaluate({
                    "token": token, "chain": chain, "vol5m": vol5m,
                    "change5m": change5m, "rsi": rsi,
                })
                if sig and sig.signal_type != SignalType.HOLD:
                    await self.queue.put(sig)
            except Exception as e:
                log.warning("momentum.pair_parse_error", error=str(e))

    def _compute_rsi(self, prices: list[float]) -> float | None:
        if len(prices) < self.rsi_period + 1:
            return None
        gains, losses = 0.0, 0.0
        for i in range(-self.rsi_period, 0):
            diff = prices[i] - prices[i - 1]
            if diff > 0:
                gains += diff
            else:
                losses -= diff
        if losses == 0:
            return 100.0
        rs = (gains / self.rsi_period) / (losses / self.rsi_period)
        return 100.0 - (100.0 / (1.0 + rs))

    async def evaluate(self, data: dict) -> Signal | None:
        rsi = data.get("rsi")
        if rsi is None:
            return None
        if rsi < self.rsi_oversold and data["change5m"] > self.min_price_change:
            return Signal(
                strategy=self.name,
                chain=data["chain"],
                token_address=data["token"],
                signal_type=SignalType.BUY,
                suggested_size_pct=0.015,
                confidence=min(1.0, (self.rsi_oversold - rsi) / 30),
                reason=f"Momentum breakout RSI={rsi:.1f}, change5m={data['change5m']:.1f}%",
                stop_loss_pct=12.0,
                take_profit_pct=40.0,
            )
        if rsi > self.rsi_overbought:
            return Signal(
                strategy=self.name,
                chain=data["chain"],
                token_address=data["token"],
                signal_type=SignalType.SELL,
                suggested_size_pct=1.0,
                confidence=0.5,
                reason=f"RSI overbought={rsi:.1f}",
            )
        return Signal(strategy=self.name, chain=data["chain"], token_address=data["token"],
                      signal_type=SignalType.HOLD, suggested_size_pct=0, confidence=0, reason="No signal")
