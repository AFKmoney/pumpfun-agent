"""
jito_client.py
==============
Jito bundle integration for MEV-protected sniping on Solana.

Why Jito?
- Sniping on pump.fun is highly competitive. Public mempool = your tx gets
  frontrun by bots that see your pending buy.
- Jito bundles are atomic: either the whole bundle lands or none of it does.
- Tip-based: you pay a tip to Jito validators for priority inclusion.

Bundle types supported:
1. Single-tx bundle: just the swap, submitted via Jito (no tip instruction).
   - Use when tip is configured as 0, or when testing.
2. Multi-tx bundle: tip ix + swap, atomic.
   - Built by requesting Jupiter's `asLegacyTransaction=true` option so we
     can append a SystemProgram.transfer tip instruction before signing.
   - All-or-nothing: if tip account doesn't get paid, swap doesn't land.

Tip accounts (rotated for fairness): see JITO_TIP_ACCOUNTS below.
"""
from __future__ import annotations

import asyncio
import base64
import random
from typing import Optional

import aiohttp

from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("jito")

# Jito tip accounts (rotated). See https://jito-labs.gitbook.io/mev/tip-account
JITO_TIP_ACCOUNTS = [
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNjsXLcgrl",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pivKeVBBjNhCYWcQfk2A",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
]

# Jito block engine endpoints
JITO_ENDPOINTS = {
    "mainnet": [
        "https://mainnet.block-engine.jito.wtf/api/v1/bundles",
        "https://ny.mainnet.block-engine.jito.wtf/api/v1/bundles",
        "https://frankfurt.mainnet.block-engine.jito.wtf/api/v1/bundles",
        "https://amsterdam.mainnet.block-engine.jito.wtf/api/v1/bundles",
        "https://tokyo.mainnet.block-engine.jito.wtf/api/v1/bundles",
        "https://slc.mainnet.block-engine.jito.wtf/api/v1/bundles",
    ],
}


class JitoClient:
    """Submits atomic bundles to Jito block engines."""

    def __init__(self) -> None:
        self.cfg = Config.get()
        self.enabled = self.cfg.get_nested("jito", "enabled", default=False)
        self.min_size_for_bundle = self.cfg.get_nested("jito", "min_size_sol", default=0.1)
        # tip_lamports is in SOL in config; convert to lamports
        tip_sol = float(self.cfg.get_nested("jito", "tip_lamports", default=0.001))
        self.tip_lamports = int(tip_sol * 1_000_000_000)
        self.endpoints = JITO_ENDPOINTS["mainnet"]
        self._session: Optional[aiohttp.ClientSession] = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
        return self._session

    def should_use_bundle(self, size_sol: float) -> bool:
        return self.enabled and size_sol >= self.min_size_for_bundle

    # ------------------------------------------------------------------
    # Multi-tx bundle construction
    # ------------------------------------------------------------------
    def build_tip_ix(self, payer_pubkey: str, tip_lamports: Optional[int] = None):
        """
        Build a SystemProgram.transfer instruction paying the tip to a random Jito tip account.
        Returns solders.instruction.Instruction.
        """
        from solders.pubkey import Pubkey as Pub
        from solders.system_program import TransferParams, transfer

        tip_account = Pub(random.choice(JITO_TIP_ACCOUNTS))
        payer = Pub(payer_pubkey)
        lamports = tip_lamports if tip_lamports is not None else self.tip_lamports
        return transfer(TransferParams(from_pubkey=payer, to_pubkey=tip_account, lamports=lamports))

    async def build_multi_tx_bundle(self, swap_b64: str, payer_keypair) -> Optional[list[str]]:
        """
        Build a 2-tx Jito bundle: [tip_tx, swap_tx], both signed by the same
        payer, both using a SINGLE fresh blockhash so they are atomic.

        IMPORTANT: For v0 (versioned) transactions, re-signing with a different
        blockhash requires recompiling the message. This implementation therefore
        re-fetches a single fresh blockhash, builds the tip tx on it, and submits
        the swap tx using its OWN blockhash (the one Jupiter returned). Jito
        accepts bundles whose transactions use different blockhashes as long as
        both are still valid (~60s). For true same-blockhash atomicity, the
        caller should instead append the tip ix INTO the swap tx and submit as a
        single-tx bundle — which is what the new pump.fun direct buy path does.

        Args:
            swap_b64: base64 of a SIGNED swap transaction (Jupiter or pump.fun direct).
            payer_keypair: solders.keypair.Keypair of the payer/tip source.

        Returns:
            list of base64-encoded signed transactions, or None on failure.
        """
        try:
            from solders.transaction import Transaction
            from solders.message import Message
            from solana.rpc.async_api import AsyncClient

            # 1. Fresh blockhash for the tip tx
            client = AsyncClient(self.cfg["chains"]["solana"]["rpc_endpoints"][0])
            try:
                recent_blockhash_resp = await client.get_latest_blockhash()
                blockhash = recent_blockhash_resp.value.blockhash
            finally:
                await client.close()

            # 2. Build & sign the tip tx with the fresh blockhash
            tip_ix = self.build_tip_ix(str(payer_keypair.pubkey()))
            tip_msg = Message.new_with_blockhash(
                tip_ix, payer_keypair.pubkey(), blockhash,
            )
            tip_tx = Transaction.new(tip_msg, [payer_keypair])
            tip_b64 = base64.b64encode(bytes(tip_tx)).decode("ascii")

            # 3. Swap tx: submitted as-is (already signed by the caller).
            #    We do NOT re-sign — re-signing a v0 tx with a different blockhash
            #    is invalid. Jito accepts multi-blockhash bundles.
            log.info("jito.multi_tx_built",
                     tip_lamports=self.tip_lamports,
                     tip_account=tip_ix.accounts[1].pubkey)
            return [tip_b64, swap_b64]

        except Exception as e:
            log.error("jito.build_multi_tx_failed", error=str(e))
            return None

    # ------------------------------------------------------------------
    # Bundle submission
    # ------------------------------------------------------------------
    async def submit_bundle(self, signed_txs_b64: list[str]) -> Optional[str]:
        """
        Submit one or more signed transactions as a Jito bundle.
        Returns the bundle UUID, or None on failure.
        """
        s = await self.session()
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "sendBundle",
            "params": [signed_txs_b64],
        }
        last_err = None
        for endpoint in self.endpoints:
            try:
                async with s.post(endpoint, json=payload) as r:
                    data = await r.json()
                if "result" in data:
                    log.info("jito.bundle_submitted", uuid=data["result"],
                             endpoint=endpoint, n_txs=len(signed_txs_b64))
                    return data["result"]
                last_err = data.get("error")
            except Exception as e:
                last_err = str(e)
                continue
        log.error("jito.bundle_submit_failed", error=str(last_err))
        return None

    async def get_bundle_status(self, bundle_uuid: str) -> str:
        """Returns one of: Pending, Landed, Failed, Expired, Unknown."""
        s = await self.session()
        endpoint = self.endpoints[0]
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getInflightBundleStatuses",
            "params": [[bundle_uuid]],
        }
        try:
            async with s.post(endpoint, json=payload) as r:
                data = await r.json()
            results = data.get("result", {}).get("value", [])
            if not results:
                return "Unknown"
            entry = results[0]
            return entry.get("confirmation_status") or entry.get("status") or "Pending"
        except Exception as e:
            log.warning("jito.status_failed", error=str(e))
            return "Unknown"

    async def wait_for_landing(self, bundle_uuid: str, timeout_sec: float = 15.0) -> str:
        """Poll status until landed/failed/expired or timeout."""
        deadline = asyncio.get_event_loop().time() + timeout_sec
        while asyncio.get_event_loop().time() < deadline:
            status = await self.get_bundle_status(bundle_uuid)
            if status in ("Landed", "Failed", "Expired"):
                return status
            await asyncio.sleep(0.5)
        return "Timeout"

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
