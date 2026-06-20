"""
data_providers.py
=================
External data providers used by the anti-rugpull strategy and the backtester.

Providers implemented:
- DexScreenerClient  (free, no key)   -> liquidity, price, volume, holders overview
- HeliusClient        (needs API key) -> getTokenHolders, getTokenAccounts
- BirdeyeClient       (needs API key) -> OHLCV for backtesting
- TokenSnifferClient  (free)          -> EVM honeypot + tax info

All clients are async, with retry + circuit breaker (tenacity).
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

import aiohttp
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("data_providers")


# =====================================================================
# DexScreener (free, no key)
# =====================================================================
class DexScreenerClient:
    BASE = "https://api.dexscreener.com"

    def __init__(self) -> None:
        self.cfg = Config.get()
        self._session: Optional[aiohttp.ClientSession] = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._session

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def get_pair(self, token_address: str) -> Optional[dict]:
        """Return the first DexScreener pair for a token, or None."""
        s = await self.session()
        url = f"{self.BASE}/latest/dex/tokens/{token_address}"
        async with s.get(url) as r:
            if r.status != 200:
                return None
            data = await r.json()
        pairs = data.get("pairs") or data.get("pair") or []
        if not pairs:
            return None
        return pairs[0] if isinstance(pairs, list) else pairs

    async def get_liquidity_usd(self, token_address: str) -> float:
        pair = await self.get_pair(token_address)
        if not pair:
            return 0.0
        liq = pair.get("liquidity") or {}
        return float(liq.get("usd", 0))

    async def get_price_usd(self, token_address: str) -> float:
        pair = await self.get_pair(token_address)
        if not pair:
            return 0.0
        return float(pair.get("priceUsd", 0) or 0)

    async def get_volume_5m(self, token_address: str) -> float:
        pair = await self.get_pair(token_address)
        if not pair:
            return 0.0
        return float((pair.get("volume") or {}).get("m5", 0))

    async def get_holders_count(self, token_address: str) -> int:
        pair = await self.get_pair(token_address)
        if not pair:
            return 0
        info = pair.get("info") or {}
        return int(info.get("holders", 0) or 0)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


# =====================================================================
# Helius (Solana RPC + enhanced APIs)
# =====================================================================
class HeliusClient:
    """Wraps Helius RPC + enhanced endpoints (getTokenHolders, etc.)."""

    def __init__(self) -> None:
        self.cfg = Config.get()
        api_key = os.environ.get("HELIUS_API_KEY", "")
        base = self.cfg.get_nested("chains", "solana", "rpc_endpoints", default=[""])[0]
        # If the configured endpoint already contains the key, use it as-is.
        if "api-key=" in base:
            self.rpc_url = base
        elif api_key:
            self.rpc_url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        else:
            self.rpc_url = base
        self._session: Optional[aiohttp.ClientSession] = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self._session

    async def _rpc(self, method: str, params: list) -> dict:
        s = await self.session()
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        async with s.post(self.rpc_url, json=payload) as r:
            return await r.json()

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def get_token_holders(self, mint: str, limit: int = 100) -> list[dict]:
        """
        Returns list of {owner, amount, decimals} for top holders.
        Uses the Helius enhanced API: getPriorityTokenAccounts (DAS).
        Falls back to getTokenLargestAccounts RPC method.
        """
        try:
            resp = await self._rpc("getTokenLargestAccounts", [mint])
            accounts = resp.get("result", {}).get("value", [])
            return [{"owner": a.get("address", ""), "amount": float(a.get("amount", 0) or 0),
                     "decimals": a.get("decimals", 9)} for a in accounts[:limit]]
        except Exception as e:
            log.warning("helius.get_token_holders_failed", mint=mint, error=str(e))
            return []

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def get_account_info(self, mint: str) -> dict:
        resp = await self._rpc("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
        return resp.get("result", {}).get("value", {}) or {}

    async def is_mint_authority_renounced(self, mint: str) -> bool:
        """True if mint authority is null (renounced)."""
        info = await self.get_account_info(mint)
        parsed = (info.get("data") or {}).get("parsed", {}).get("info", {})
        return not parsed.get("mintAuthority")

    async def is_freeze_authority_renounced(self, mint: str) -> bool:
        info = await self.get_account_info(mint)
        parsed = (info.get("data") or {}).get("parsed", {}).get("info", {})
        return not parsed.get("freezeAuthority")

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


# =====================================================================
# Birdeye (OHLCV for backtesting + signal enrichment)
# =====================================================================
class BirdeyeClient:
    BASE = "https://public-api.birdeye.so"

    def __init__(self) -> None:
        self.cfg = Config.get()
        self.api_key = os.environ.get("BIRDEYE_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
        return self._session

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def get_ohlcv(self, token_address: str, interval: str = "1m", limit: int = 1000) -> list[dict]:
        """
        Returns list of {timestamp, open, high, low, close, volume}.
        interval: 1m, 5m, 15m, 1h, 4h, 1d
        """
        if not self.api_key:
            log.warning("birdeye.no_api_key")
            return []
        s = await self.session()
        url = f"{self.BASE}/defi/ohlcv"
        headers = {"X-API-KEY": self.api_key, "x-chain": "solana"}
        params = {"address": token_address, "type": interval, "limit": limit}
        try:
            async with s.get(url, headers=headers, params=params) as r:
                if r.status != 200:
                    log.warning("birdeye.ohlcv_failed", status=r.status, token=token_address)
                    return []
                data = await r.json()
            return data.get("data", {}).get("items", []) or []
        except Exception as e:
            log.warning("birdeye.exception", error=str(e))
            return []

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


# =====================================================================
# TokenSniffer (EVM honeypot + tax info, free)
# =====================================================================
class TokenSnifferClient:
    BASE = "https://api.tokensniffer.com/v2"

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
        return self._session

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def get_token_audit(self, chain: str, address: str) -> Optional[dict]:
        """
        chain: 'base', 'ethereum'
        Returns audit dict including honeypot flag + tax info.
        """
        s = await self.session()
        url = f"{self.BASE}/token/{chain}/{address}"
        try:
            async with s.get(url) as r:
                if r.status != 200:
                    return None
                return await r.json()
        except Exception as e:
            log.warning("tokensniffer.exception", error=str(e))
            return None

    async def is_honeypot(self, chain: str, address: str) -> tuple[bool, str]:
        """Returns (is_honeypot, reason)."""
        audit = await self.get_token_audit(chain, address)
        if not audit:
            return False, "Audit unavailable"
        if audit.get("is_honeypot"):
            return True, "TokenSniffer flagged honeypot"
        buy_tax = audit.get("buy_tax", 0) or 0
        sell_tax = audit.get("sell_tax", 0) or 0
        if buy_tax > 0.10 or sell_tax > 0.10:
            return True, f"Tax too high (buy={buy_tax:.0%}, sell={sell_tax:.0%})"
        return False, "OK"

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


# =====================================================================
# Singleton accessor
# =====================================================================
_providers: dict[str, Any] = {}


def get_providers() -> dict[str, Any]:
    """Lazy-init all providers."""
    global _providers
    if not _providers:
        _providers = {
            "dexscreener": DexScreenerClient(),
            "helius": HeliusClient(),
            "birdeye": BirdeyeClient(),
            "tokensniffer": TokenSnifferClient(),
        }
    return _providers


async def close_all_providers() -> None:
    global _providers
    for p in _providers.values():
        try:
            await p.close()
        except Exception:
            pass
    _providers = {}
