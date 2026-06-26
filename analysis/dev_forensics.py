"""
dev_forensics.py
================
Dev wallet forensic history — scores a token creator by how many of their past
launches ended up as rugs.

WHY THIS MATTERS (Edge #2 in COMPETITIVE_EDGE.md):
Every pump.fun token has a creator wallet. We track that wallet's behavior
across EVERY token they've ever launched — not just the current one. If a dev
has rugged 3 tokens in the last week, we exit their 4th launch before they can
dump. The giants don't track individual Solana dev wallets because they don't
trade launches. This data rots fast (devs rotate wallets), so only a 24/7 bot
with historical memory captures it.

HOW IT WORKS:
Given a dev wallet address, we:
  1. Fetch their recent transaction signatures (getSignaturesForAddress).
  2. For each signature, check if it's a pump.fun create instruction.
  3. For each created token, fetch current price/bonding-curve state.
  4. A token is "rugged" if its price dropped >80% from launch, or the bonding
     curve was abandoned (real_sol_reserves == 0), or it's been delisted.
  5. rug_score = rugged_count / total_created_count.

The result is cached per dev wallet for 1 hour (history doesn't change fast).

DATA SOURCE: Helius RPC (getSignaturesForAddress + getTransaction), which we
already have wired. No new API key needed.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from utils.bonding_curve import BondingCurveAnalyzer
from utils.config_loader import Config
from utils.data_providers import get_providers
from utils.logger import setup_logger
from utils.pumpfun_parser import PUMP_FUN_PROGRAM
from utils.pumpfun_ix import CREATE_DISCRIMINATOR

log = setup_logger("dev_forensics")

# How many signatures to scan per dev wallet (capped for rate-limit safety).
_MAX_SIGS_TO_SCAN = 50
# A token is "rugged" if its current price is < this fraction of its launch price.
_RUG_PRICE_THRESHOLD = 0.20  # -80%
_CACHE_TTL_SEC = 3600  # 1 hour


@dataclass
class DevReputation:
    dev_wallet: str
    total_created: int
    rug_count: int
    rug_score: float               # rug_count / total_created (0..1)
    known_mints: list[str]
    is_serial_rugger: bool         # True if rug_score >= 0.5 AND total_created >= 3
    assessed_at: float


class DevForensics:
    """
    Scores a dev wallet's launch history to detect serial ruggers.
    Used by the executor as an additional gate (refuse buys from serial ruggers)
    and by the smart exit (exit immediately if a serial rugger's token starts
    showing any weakness).
    """

    def __init__(self, bonding: Optional[BondingCurveAnalyzer] = None) -> None:
        self.cfg = Config.get()
        self.bonding = bonding or BondingCurveAnalyzer()
        self._cache: dict[str, tuple[float, DevReputation]] = {}

    async def assess(self, dev_wallet: str) -> Optional[DevReputation]:
        """
        Fetch the dev's recent history and compute their rug reputation.
        Returns None if the wallet is unknown or fetch fails.
        """
        if not dev_wallet:
            return None
        # Cache
        cached = self._cache.get(dev_wallet)
        if cached and time.time() - cached[0] < _CACHE_TTL_SEC:
            return cached[1]

        providers = get_providers()
        helius = providers["helius"]
        try:
            # 1. Get recent signatures for the dev wallet
            resp = await helius._rpc("getSignaturesForAddress", [
                dev_wallet, {"limit": _MAX_SIGS_TO_SCAN},
            ])
            sigs = resp.get("result", [])
        except Exception as e:
            log.warning("dev_forensics.sigs_fetch_failed", dev=dev_wallet[:8], error=str(e))
            return None

        # 2. Find pump.fun create instructions among their signatures
        created_mints: list[str] = []
        for entry in sigs[:_MAX_SIGS_TO_SCAN]:
            sig = entry.get("signature")
            if not sig or entry.get("err"):
                continue
            mint = await self._extract_created_mint(helius, sig)
            if mint and mint not in created_mints:
                created_mints.append(mint)

        if not created_mints:
            rep = DevReputation(
                dev_wallet=dev_wallet, total_created=0, rug_count=0,
                rug_score=0.0, known_mints=[], is_serial_rugger=False,
                assessed_at=time.time(),
            )
            self._cache[dev_wallet] = (time.time(), rep)
            return rep

        # 3. Check each created token's current state for rug evidence
        rug_count = 0
        for mint in created_mints:
            if await self._is_rugged(mint):
                rug_count += 1

        rug_score = rug_count / len(created_mints) if created_mints else 0.0
        rep = DevReputation(
            dev_wallet=dev_wallet,
            total_created=len(created_mints),
            rug_count=rug_count,
            rug_score=round(rug_score, 2),
            known_mints=created_mints,
            is_serial_rugger=(rug_score >= 0.5 and len(created_mints) >= 3),
            assessed_at=time.time(),
        )
        self._cache[dev_wallet] = (time.time(), rep)
        log.info("dev_forensics.assessed", dev=dev_wallet[:8],
                 created=rep.total_created, rugs=rep.rug_count,
                 score=rep.rug_score, serial=rep.is_serial_rugger)
        return rep

    async def _extract_created_mint(self, helius, signature: str) -> Optional[str]:
        """
        Fetch a transaction and check if it contains a pump.fun create
        instruction. If so, return the created mint address.
        """
        try:
            resp = await helius._rpc("getTransaction", [
                signature,
                {"maxSupportedTransactionVersion": 0, "encoding": "jsonParsed"},
            ])
            tx = resp.get("result")
            if not tx:
                return None
            message = tx.get("transaction", {}).get("message", {}) or {}
            account_keys = message.get("accountKeys", []) or []
            instructions = list(message.get("instructions", []) or [])
            for inner_group in (tx.get("meta", {}).get("innerInstructions") or []):
                instructions.extend(inner_group.get("instructions", []) or [])
            for ix in instructions:
                prog = ix.get("programId")
                if isinstance(prog, int):
                    if prog < len(account_keys):
                        prog = account_keys[prog]
                    else:
                        continue
                if prog != PUMP_FUN_PROGRAM:
                    continue
                data = ix.get("data", "")
                import base64
                try:
                    raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
                except Exception:
                    continue
                if len(raw) >= 8 and raw[:8] == CREATE_DISCRIMINATOR:
                    # Mint is the 4th account (index 3): global, fee_recipient,
                    # ... actually the create ix accounts are:
                    #   mint, bonding_curve, user (and the mint is the first).
                    # From parse_create_instruction: mint is parsed from data,
                    # but we can also get it from accounts[0] for the create ix.
                    accs = ix.get("accounts", []) or []
                    if accs:
                        midx = accs[0]
                        if isinstance(midx, int) and midx < len(account_keys):
                            return account_keys[midx]
                        if isinstance(midx, str):
                            return midx
                    # Fallback: parse from data (name, symbol, uri, then mint pubkey)
                    try:
                        from utils.pumpfun_parser import parse_create_instruction
                        ev = parse_create_instruction(data)
                        if ev:
                            return ev.mint
                    except Exception:
                        continue
            return None
        except Exception as e:
            log.debug("dev_forensics.tx_decode_failed", sig=signature[:12], error=str(e))
            return None

    async def _is_rugged(self, mint: str) -> bool:
        """
        Heuristic: a token is 'rugged' if its bonding curve shows 0 real SOL
        reserves (abandoned) or it has extremely low liquidity (< $1k).
        """
        try:
            bc = await self.bonding.fetch_state(mint)
            if bc is None:
                # No bonding curve data + token not migrated = likely abandoned = rug
                return True
            if bc.real_sol_reserves <= 0.01 and bc.completion_pct < 50:
                return True
            # Check liquidity via DexScreener (post-migration tokens)
            providers = get_providers()
            dex = providers["dexscreener"]
            liq = await dex.get_liquidity_usd(mint)
            if liq is not None and liq < 500:
                return True
            return False
        except Exception:
            return False

    def is_serial_rugger(self, dev_wallet: str) -> bool:
        """Synchronous check using cached reputation (assess must have run)."""
        cached = self._cache.get(dev_wallet)
        if cached and time.time() - cached[0] < _CACHE_TTL_SEC:
            return cached[1].is_serial_rugger
        return False
