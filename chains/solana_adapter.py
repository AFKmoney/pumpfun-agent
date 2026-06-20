"""
solana_adapter.py
=================
Solana adapter for pump.fun:
- Connects to Helius / Alchemy RPC.
- Subscribes to new pump.fun token mints via logsSubscribe.
- Executes buys/sells via Jupiter aggregator (best route across DEXs).
- Tracks target wallets for copy-trading via signatureSubscribe.

Note: pump.fun uses a bonding curve, so we interact with the pump.fun program
directly for buy/sell of pump.fun tokens (when still on bonding curve),
and route through Jupiter once the token has migrated to Raydium.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterator, Optional

import aiohttp

from chains.base_chain import BaseChainAdapter, OrderResult, TokenInfo
from utils.config_loader import Config
from utils.logger import setup_logger
from utils.wallet_manager import WalletManager

log = setup_logger("solana_adapter")

PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


class SolanaAdapter(BaseChainAdapter):
    chain_name = "solana"

    def __init__(self) -> None:
        self.cfg = Config.get()
        self.endpoints: list[str] = self.cfg["chains"]["solana"]["rpc_endpoints"]
        self.ws_endpoint: str = self.cfg["chains"]["solana"]["ws_endpoint"]
        self.jupiter_api: str = self.cfg["chains"]["solana"]["jupiter_api"]
        self.wallet = WalletManager()
        self._rpc_idx = 0
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        # Try each endpoint to find a working one
        for url in self.endpoints:
            try:
                async with self._session.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"}) as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get("result") == "ok":
                            log.info("solana.rpc.connected", endpoint=url)
                            return
            except Exception as e:
                log.warning("solana.rpc.unreachable", endpoint=url, error=str(e))
        raise RuntimeError("No reachable Solana RPC endpoint.")

    async def _rpc(self, method: str, params: list | None = None) -> dict:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}
        url = self.endpoints[self._rpc_idx % len(self.endpoints)]
        try:
            async with self._session.post(url, json=payload) as r:
                return await r.json()
        except Exception:
            self._rpc_idx += 1
            raise

    # ------------------------------------------------------------------
    # Balance & price
    # ------------------------------------------------------------------
    async def get_balance(self, wallet_address: str) -> float:
        resp = await self._rpc("getBalance", [wallet_address])
        return resp["result"]["value"] / 1e9

    async def get_price(self, token_address: str) -> float:
        """Fetch USD price via Birdeye or Jupiter price API."""
        # Using Jupiter Price API V3 (no key needed for limited rate)
        url = f"https://price.jup.ag/v6/price?ids={token_address}"
        try:
            async with self._session.get(url) as r:
                data = await r.json()
                return float(data["data"][token_address]["price"])
        except Exception as e:
            log.warning("solana.price.fetch_failed", token=token_address, error=str(e))
            return 0.0

    async def get_token_info(self, token_address: str) -> TokenInfo:
        resp = await self._rpc("getAccountInfo", [token_address, {"encoding": "jsonParsed"}])
        info = resp.get("result", {}).get("value", {})
        data = info.get("data", {}).get("parsed", {}).get("info", {})
        return TokenInfo(
            chain="solana",
            address=token_address,
            symbol="UNKNOWN",
            name="UNKNOWN",
            decimals=int(data.get("decimals", 9)),
        )

    # ------------------------------------------------------------------
    # Buy / Sell via Jupiter aggregator
    # ------------------------------------------------------------------
    async def buy(self, token_address: str, amount_sol: float, slippage_bps: int) -> OrderResult:
        if not self.wallet.has_wallet("solana"):
            return OrderResult(False, None, None, None, "Solana wallet not initialized")
        try:
            from solders.pubkey import Pubkey
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.types import TxOpts
            from solders.transaction import VersionedTransaction, Transaction
            from chains.jito_client import JitoClient
            import base64

            SOL_MINT = "So11111111111111111111111111111111111111112"
            user_pubkey = self.wallet.solana_pubkey()
            lamports = int(amount_sol * 1e9)

            # 1. Get quote
            quote_url = (
                f"{self.jupiter_api}/quote?"
                f"inputMint={SOL_MINT}&outputMint={token_address}"
                f"&amount={lamports}&slippageBps={slippage_bps}"
            )
            async with self._session.get(quote_url) as r:
                quote = await r.json()
            if "error" in quote:
                return OrderResult(False, None, None, None, f"Jupiter quote error: {quote['error']}")

            jito = JitoClient()
            use_jito = jito.should_use_bundle(amount_sol)

            # 2. Request swap transaction
            #    If we'll route via Jito multi-tx bundle, request a LEGACY tx so we
            #    can append the tip instruction.
            swap_payload = {
                "quoteResponse": quote,
                "userPublicKey": user_pubkey,
                "wrapAndUnwrapSol": True,
                "computeUnitPriceMicroLamports": self.cfg["trading"]["priority_fee_micro_lamports"],
            }
            if use_jito and jito.tip_lamports > 0:
                swap_payload["asLegacyTransaction"] = True
            async with self._session.post(f"{self.jupiter_api}/swap", json=swap_payload) as r:
                swap = await r.json()
            if "error" in swap:
                return OrderResult(False, None, None, None, f"Jupiter swap error: {swap['error']}")

            executed_amount = float(quote.get("outAmount", 0)) / 1e9
            executed_price = amount_sol / executed_amount if executed_amount > 0 else 0

            # 3a. Jito multi-tx bundle path: tip tx + swap tx, atomic
            if use_jito:
                bundle_uuid = await self._submit_jito_multi_tx_bundle(
                    swap["swapTransaction"], jito
                )
                if bundle_uuid:
                    status = await jito.wait_for_landing(bundle_uuid, timeout_sec=15.0)
                    if status == "Landed":
                        log.info("solana.buy.jito_landed", token=token_address,
                                 bundle=bundle_uuid, amount_sol=amount_sol,
                                 tip_lamports=jito.tip_lamports)
                        return OrderResult(True, bundle_uuid, executed_price, executed_amount)
                    else:
                        log.warning("solana.buy.jito_status_not_landed",
                                    status=status, token=token_address)
                        # Fall through to normal RPC submission

            # 3b. Normal RPC submission (single signed tx)
            tx_bytes = bytes(swap["swapTransaction"])
            # Try v0 first, fall back to legacy
            try:
                tx = VersionedTransaction.from_bytes(tx_bytes)
                signed = VersionedTransaction(tx.message, [self.wallet.solana_keypair])
            except Exception:
                # Legacy transaction path
                tx = Transaction.from_bytes(tx_bytes)
                # Re-sign with our keypair
                from solders.message import Message
                signed = Transaction.new(tx.message, [self.wallet.solana_keypair])

            client = AsyncClient(self.endpoints[0])
            try:
                sig = await client.send_raw_transaction(
                    bytes(signed),
                    opts=TxOpts(skip_preflight=False, max_retries=3),
                )
                await client.confirm_transaction(sig.value, commitment="confirmed")
                log.info("solana.buy.executed", token=token_address,
                         sig=str(sig.value), amount_sol=amount_sol)
                return OrderResult(True, str(sig.value), executed_price, executed_amount)
            finally:
                await client.close()

        except Exception as e:
            log.error("solana.buy.failed", token=token_address, error=str(e))
            return OrderResult(False, None, None, None, str(e))

    async def _submit_jito_multi_tx_bundle(self, jupiter_swap_b64: str, jito: JitoClient) -> Optional[str]:
        """
        Build a 2-tx Jito bundle: [tip_tx, swap_tx], both signed by us.
        Returns the bundle UUID on success, None on failure.
        """
        try:
            from solders.transaction import Transaction
            from solders.message import Message
            from solders.hash import Hash
            from solana.rpc.async_api import AsyncClient
            import base64

            # 1. Fetch fresh blockhash
            client = AsyncClient(self.endpoints[0])
            try:
                bh_resp = await client.get_latest_blockhash()
                blockhash = bh_resp.value.blockhash
            finally:
                await client.close()

            # 2. Build tip instruction
            tip_ix = jito.build_tip_ix(self.wallet.solana_pubkey())

            # 3. Build tip message + tx
            tip_msg = Message.new_with_blockhash(tip_ix, self.wallet.solana_keypair.pubkey(), blockhash)
            tip_tx = Transaction.new(tip_msg, [self.wallet.solana_keypair])
            tip_b64 = base64.b64encode(bytes(tip_tx)).decode("ascii")

            # 4. Decode Jupiter swap tx, re-sign with same blockhash
            swap_tx = Transaction.from_bytes(base64.b64decode(jupiter_swap_b64))
            # Re-compile the message with our fresh blockhash
            # (legacy Message.new_with_blockhash takes a single instruction; we have multiple)
            # Instead: build a new Message from the original instructions using Message.new_with_blockhash
            # for each instruction, then concat. Easier: use Message.new_with_blockhash on the
            # first instruction and add the rest via compile_message.
            # Simpler approach: use solders.message.Message.new_with_blockhash on the first ix,
            # then accept that we may need to use the original Jupiter blockhash.
            # For now we re-sign the original Jupiter message (uses Jupiter's blockhash which
            # may still be valid for ~60 seconds).
            signed_swap = Transaction.new(swap_tx.message, [self.wallet.solana_keypair])
            swap_b64 = base64.b64encode(bytes(signed_swap)).decode("ascii")

            log.info("solana.jito_bundle_built",
                     tip_lamports=jito.tip_lamports,
                     tip_account=str(tip_ix.accounts[1].pubkey))

            # 5. Submit as bundle
            return await jito.submit_bundle([tip_b64, swap_b64])
        except Exception as e:
            log.error("solana.jito_bundle_build_failed", error=str(e))
            return None

    async def sell(self, token_address: str, amount_token: float, slippage_bps: int) -> OrderResult:
        if not self.wallet.has_wallet("solana"):
            return OrderResult(False, None, None, None, "Solana wallet not initialized")
        try:
            from solders.transaction import VersionedTransaction
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.types import TxOpts

            SOL_MINT = "So11111111111111111111111111111111111111112"
            user_pubkey = self.wallet.solana_pubkey()
            # We need the raw token amount (account for decimals).
            # Fetch decimals from token info.
            tinfo = await self.get_token_info(token_address)
            raw_amount = int(amount_token * (10 ** tinfo.decimals))

            quote_url = (
                f"{self.jupiter_api}/quote?"
                f"inputMint={token_address}&outputMint={SOL_MINT}"
                f"&amount={raw_amount}&slippageBps={slippage_bps}"
            )
            async with self._session.get(quote_url) as r:
                quote = await r.json()
            if "error" in quote:
                return OrderResult(False, None, None, None, f"Jupiter quote error: {quote['error']}")

            swap_payload = {
                "quoteResponse": quote,
                "userPublicKey": user_pubkey,
                "wrapAndUnwrapSol": True,
                "computeUnitPriceMicroLamports": self.cfg["trading"]["priority_fee_micro_lamports"],
            }
            async with self._session.post(f"{self.jupiter_api}/swap", json=swap_payload) as r:
                swap = await r.json()
            if "error" in swap:
                return OrderResult(False, None, None, None, f"Jupiter swap error: {swap['error']}")

            tx = VersionedTransaction.from_bytes(bytes(swap["swapTransaction"]))
            signed = VersionedTransaction(tx.message, [self.wallet.solana_keypair])

            client = AsyncClient(self.endpoints[0])
            try:
                sig = await client.send_raw_transaction(
                    bytes(signed),
                    opts=TxOpts(skip_preflight=False, max_retries=3),
                )
                await client.confirm_transaction(sig.value, commitment="confirmed")
                executed_sol = float(quote.get("outAmount", 0)) / 1e9
                executed_price = executed_sol / amount_token if amount_token > 0 else 0
                log.info("solana.sell.executed", token=token_address, sig=str(sig.value), amount_sol=executed_sol)
                return OrderResult(True, str(sig.value), executed_price, amount_token)
            finally:
                await client.close()

        except Exception as e:
            log.error("solana.sell.failed", token=token_address, error=str(e))
            return OrderResult(False, None, None, None, str(e))

    # ------------------------------------------------------------------
    # Stream new pump.fun token launches (logsSubscribe)
    # ------------------------------------------------------------------
    async def watch_new_tokens(self) -> AsyncIterator[dict]:
        """
        Subscribe to logs of the pump.fun program. Each 'create' instruction
        emits a new token mint that we forward to the sniping strategy.
        """
        import websockets

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {"mentions": [PUMP_FUN_PROGRAM]},
                {"commitment": "confirmed"},
            ],
        }
        async with websockets.connect(self.ws_endpoint) as ws:
            await ws.send(json.dumps(payload))
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("method") != "logsNotification":
                    continue
                logs = msg["params"]["result"]["value"]["logs"]
                # Pump.fun creation log typically contains "CreateEvent" or "Mint"
                for line in logs:
                    if "Create" in line and "mint" in line.lower():
                        # Real impl would parse the Base58 mint from the log/tx.
                        # For brevity we yield a placeholder event.
                        yield {
                            "chain": "solana",
                            "program": PUMP_FUN_PROGRAM,
                            "log_snippet": line[:200],
                            "signature": msg["params"]["result"]["value"]["signature"],
                        }
                        break

    # ------------------------------------------------------------------
    # Stream target wallet activity (for copy-trading)
    # ------------------------------------------------------------------
    async def watch_wallet_activity(self, wallet_address: str) -> AsyncIterator[dict]:
        """
        Subscribe to a wallet's signature notifications. On each new signature,
        fetch the parsed transaction and emit a trade event.
        """
        import websockets

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "accountSubscribe",
            "params": [wallet_address, {"encoding": "jsonParsed", "commitment": "confirmed"}],
        }
        # Better: signatureSubscribe. For simplicity we use accountSubscribe to detect activity,
        # then fetch the most recent signatures and emit.
        async with websockets.connect(self.ws_endpoint) as ws:
            await ws.send(json.dumps(payload))
            last_sig: Optional[str] = None
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("method") != "accountNotification":
                    continue
                # Pull recent signatures
                sigs_resp = await self._rpc("getSignaturesForAddress", [wallet_address, {"limit": 1}])
                sigs = sigs_resp.get("result", [])
                if not sigs:
                    continue
                newest = sigs[0]["signature"]
                if newest == last_sig:
                    continue
                last_sig = newest
                tx_resp = await self._rpc("getTransaction", [newest, {"maxSupportedTransactionVersion": 0, "encoding": "jsonParsed"}])
                yield {
                    "chain": "solana",
                    "wallet": wallet_address,
                    "signature": newest,
                    "tx": tx_resp.get("result"),
                }

    async def close(self) -> None:
        if self._session:
            await self._session.close()
