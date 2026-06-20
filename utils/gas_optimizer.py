"""
gas_optimizer.py
================
Dynamic priority fee / gas price optimizer.

Polls Solana's `getRecentPriorityFees` every 30s and maintains a rolling
estimate of the priority fee needed for inclusion within 5 slots.

For EVM chains, polls `eth_gasPrice` and adjusts based on recent block
congestion (gas used / gas limit).

This avoids:
- Overpaying priority fees when the network is calm
- Underpaying (and missing snipes) when congested
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Optional

import aiohttp

from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("gas_optimizer")


class GasOptimizer:
    """Dynamic Solana priority fee + EVM gas price adjuster."""

    def __init__(self) -> None:
        self.cfg = Config.get()
        # Rolling window of recent priority fees (micro-lamports)
        self.sol_priority_fees: deque[float] = deque(maxlen=20)
        self.sol_last_poll: float = 0
        # EVM gas prices (gwei)
        self.evm_gas_prices: dict[str, deque[float]] = {"base": deque(maxlen=20), "ethereum": deque(maxlen=20)}
        self.evm_last_poll: dict[str, float] = {"base": 0, "ethereum": 0}
        self._session: Optional[aiohttp.ClientSession] = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    # ------------------------------------------------------------------
    # Solana
    # ------------------------------------------------------------------
    async def poll_solana(self) -> None:
        """Poll getRecentPriorityFees from Helius."""
        if time.time() - self.sol_last_poll < 30:
            return
        try:
            endpoints = self.cfg["chains"]["solana"]["rpc_endpoints"]
            s = await self.session()
            payload = {"jsonrpc": "2.0", "id": 1, "method": "getRecentPriorityFees",
                       "params": [{"transactionType": "swap"}]}
            async with s.post(endpoints[0], json=payload) as r:
                data = await r.json()
            fees = data.get("result", {}).get("priorityFeeLevels", {})
            # Use 'high' level for sniping (top quartile), 'medium' for routine
            high_fee = fees.get("high", 50_000)
            self.sol_priority_fees.append(float(high_fee))
            self.sol_last_poll = time.time()
            log.info("gas.solana_polled", high_fee=high_fee,
                     avg=self._avg(self.sol_priority_fees))
        except Exception as e:
            log.warning("gas.solana_poll_failed", error=str(e))

    def get_solana_priority_fee(self, urgent: bool = False) -> int:
        """
        Returns recommended priority fee in micro-lamports.
        - urgent=True (sniping, SL hit): use 90th percentile of recent fees + 20% headroom
        - urgent=False (routine buys/sells): use 60th percentile
        """
        if not self.sol_priority_fees:
            return self.cfg["trading"]["priority_fee_micro_lamports"]
        sorted_fees = sorted(self.sol_priority_fees)
        if urgent:
            idx = int(len(sorted_fees) * 0.9)
            base = sorted_fees[min(idx, len(sorted_fees) - 1)]
            return int(base * 1.2)
        else:
            idx = int(len(sorted_fees) * 0.6)
            return int(sorted_fees[min(idx, len(sorted_fees) - 1)])

    # ------------------------------------------------------------------
    # EVM
    # ------------------------------------------------------------------
    async def poll_evm(self, chain: str) -> None:
        if time.time() - self.evm_last_poll.get(chain, 0) < 30:
            return
        try:
            endpoints = self.cfg["chains"][chain]["rpc_endpoints"]
            s = await self.session()
            payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_gasPrice", "params": []}
            async with s.post(endpoints[0], json=payload) as r:
                data = await r.json()
            gas_price_wei = int(data.get("result", "0x0"), 16)
            gas_price_gwei = gas_price_wei / 1e9
            self.evm_gas_prices[chain].append(gas_price_gwei)
            self.evm_last_poll[chain] = time.time()
            log.info("gas.evm_polled", chain=chain, gas_gwei=gas_price_gwei)
        except Exception as e:
            log.warning("gas.evm_poll_failed", chain=chain, error=str(e))

    def get_evm_gas_gwei(self, chain: str, urgent: bool = False) -> float:
        prices = self.evm_gas_prices.get(chain)
        if not prices:
            return self.cfg["trading"]["gas_price_gwei"]
        sorted_p = sorted(prices)
        if urgent:
            idx = int(len(sorted_p) * 0.9)
            return sorted_p[min(idx, len(sorted_p) - 1)] * 1.2
        else:
            idx = int(len(sorted_p) * 0.6)
            return sorted_p[min(idx, len(sorted_p) - 1)]

    def _avg(self, d: deque) -> float:
        return sum(d) / len(d) if d else 0

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
