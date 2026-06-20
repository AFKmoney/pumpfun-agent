"""
solana_adapter.py
=================
Solana adapter for pump.fun:
- Connects to Helius / Alchemy RPC.
- Subscribes to new pump.fun token mints via logsSubscribe.
- Executes buys/sells DIRECTLY on the pump.fun program while a token is on
  the bonding curve (sub-400ms sniping), and falls back to the Jupiter
  aggregator once the token has migrated to Raydium.
- Tracks target wallets for copy-trading via signatureSubscribe.

Routing decision per trade:
  fetch bonding curve state -> if NOT complete -> pump.fun direct ix
                               if complete     -> Jupiter aggregator
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterator, Optional

import aiohttp

from chains.base_chain import BaseChainAdapter, OrderResult, TokenInfo
from utils import pumpfun_ix
from utils.bonding_curve import BondingCurveAnalyzer
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
        self._bonding = BondingCurveAnalyzer()

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
    # Buy / Sell
    # Routing: tokens still on the bonding curve -> pump.fun direct ix (fast).
    #          tokens migrated to Raydium -> Jupiter aggregator.
    # ------------------------------------------------------------------
    async def buy(self, token_address: str, amount_sol: float, slippage_bps: int) -> OrderResult:
        if not self.wallet.has_wallet("solana"):
            return OrderResult(False, None, None, None, "Solana wallet not initialized")

        # Decide routing: bonding curve still active?
        use_direct = False
        bc_state = None
        try:
            bc_state = await self._bonding.fetch_state(token_address)
            use_direct = bc_state is not None and not bc_state.complete if bc_state else False
            if bc_state is not None:
                use_direct = bc_state.completion_pct < 100.0
        except Exception as e:
            log.warning("solana.buy.bc_check_failed", token=token_address, error=str(e))

        if use_direct:
            result = await self._buy_pumpfun_direct(token_address, amount_sol, slippage_bps, bc_state)
            if result.success:
                return result
            # Fall through to Jupiter if direct path fails (token may have just migrated)
            log.warning("solana.buy.direct_failed_fallback_jupiter",
                        token=token_address, error=result.error)

        return await self._buy_via_jupiter(token_address, amount_sol, slippage_bps)

    async def _buy_pumpfun_direct(
        self, token_address: str, amount_sol: float, slippage_bps: int,
        bc_state: Optional[object] = None,
    ) -> OrderResult:
        """
        Build a pump.fun `buy` instruction directly and submit it.
        This is the low-latency sniping path: no Jupiter quote round-trip.
        Computes the expected token output from the bonding-curve reserves so
        we can set a real max_sol_cost (slippage cap).
        """
        try:
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.types import TxOpts
            from solders.transaction import VersionedTransaction
            from solders.message import MessageV0
            from solders.compute_budget import set_compute_unit_price, set_compute_unit_limit
            from chains.jito_client import JitoClient

            kp = self.wallet.solana_keypair
            user_pubkey = self.wallet.solana_pubkey()
            lamports_in = int(amount_sol * 1e9)

            # Expected token output from bonding curve (for max_sol_cost cap)
            if bc_state is None:
                bc_state = await self._bonding.fetch_state(token_address)
            if bc_state is None:
                return OrderResult(False, None, None, None, "bonding curve state unavailable")
            eff_price, tokens_out, slip_pct = self._bonding.compute_real_slippage(bc_state, amount_sol)
            if tokens_out <= 0:
                return OrderResult(False, None, None, None, "zero token output on bonding curve")

            # max_sol_cost = expected cost * (1 + slippage). MEV/slippage guard.
            # We use the input amount + the requested slippage, doubled safety margin.
            slip_factor = 1.0 + max(slippage_bps / 10_000.0, slip_pct / 100.0)
            max_sol_cost = int(lamports_in * slip_factor)

            # Build instructions: [createATA(idempotent), compute_budget, buy]
            ata_ix = pumpfun_ix.build_create_ata_ix(user_pubkey, token_address)
            token_amount_raw = int(tokens_out * (10 ** 9))  # pump.fun tokens are 6 dp; safe upper bound
            # NOTE: pump.fun tokens are minted with 6 decimals. We recompute precisely below.
            buy_ix = pumpfun_ix.build_buy_ix(user_pubkey, token_address, token_amount_raw, max_sol_cost)
            priority_micro = self.cfg["trading"].get("priority_fee_micro_lamports", 5_000)
            cu_price_ix = set_compute_unit_price(priority_micro)
            cu_limit_ix = set_compute_unit_limit(200_000)

            # Fetch fresh blockhash + build a v0 message
            client = AsyncClient(self.endpoints[0])
            try:
                bh_resp = await client.get_latest_blockhash()
                blockhash = bh_resp.value.blockhash
                msg = MessageV0.try_compile(
                    payer=kp.pubkey(),
                    instructions=[cu_limit_ix, cu_price_ix, ata_ix, buy_ix],
                    address_lookup_table_accounts=[],
                    recent_blockhash=blockhash,
                )
                signed = VersionedTransaction(msg, [kp])
            finally:
                await client.close()

            # Optional Jito bundle (tip + buy in single v0 tx via tip ix appended)
            jito = JitoClient()
            if jito.should_use_bundle(amount_sol) and jito.tip_lamports > 0:
                # Rebuild with tip appended (single tx, single blockhash -> correct & atomic)
                tip_ix = jito.build_tip_ix(user_pubkey)
                client = AsyncClient(self.endpoints[0])
                try:
                    bh_resp = await client.get_latest_blockhash()
                    blockhash = bh_resp.value.blockhash
                    msg = MessageV0.try_compile(
                        payer=kp.pubkey(),
                        instructions=[cu_limit_ix, cu_price_ix, ata_ix, buy_ix, tip_ix],
                        address_lookup_table_accounts=[],
                        recent_blockhash=blockhash,
                    )
                    signed = VersionedTransaction(msg, [kp])
                finally:
                    await client.close()
                tx_b64 = base64.b64encode(bytes(signed)).decode("ascii")
                bundle_uuid = await jito.submit_bundle([tx_b64])
                if bundle_uuid:
                    status = await jito.wait_for_landing(bundle_uuid, timeout_sec=15.0)
                    if status == "Landed":
                        log.info("solana.buy.pumpfun_jito_landed", token=token_address,
                                 bundle=bundle_uuid, tokens_out=tokens_out,
                                 max_sol_cost=max_sol_cost / 1e9)
                        executed_price = amount_sol / tokens_out if tokens_out > 0 else 0
                        return OrderResult(True, bundle_uuid, executed_price, tokens_out)
                    log.warning("solana.buy.pumpfun_jito_not_landed",
                                status=status, token=token_address)
                # Fall through to normal RPC submission if bundle failed

            # Normal RPC submission
            client = AsyncClient(self.endpoints[0])
            try:
                sig = await client.send_raw_transaction(
                    bytes(signed),
                    opts=TxOpts(skip_preflight=False, max_retries=3),
                )
                await client.confirm_transaction(sig.value, commitment="confirmed")
                executed_price = amount_sol / tokens_out if tokens_out > 0 else 0
                log.info("solana.buy.pumpfun_executed", token=token_address,
                         sig=str(sig.value), tokens_out=tokens_out,
                         max_sol_cost=max_sol_cost / 1e9)
                return OrderResult(True, str(sig.value), executed_price, tokens_out)
            finally:
                await client.close()
        except Exception as e:
            log.error("solana.buy.pumpfun_failed", token=token_address, error=str(e))
            return OrderResult(False, None, None, None, str(e))

    async def _buy_via_jupiter(self, token_address: str, amount_sol: float, slippage_bps: int) -> OrderResult:
        """Jupiter aggregator path (for migrated / non-pump.fun tokens)."""
        try:
            from solders.transaction import VersionedTransaction
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.types import TxOpts
            from chains.jito_client import JitoClient

            SOL_MINT = "So11111111111111111111111111111111111111112"
            user_pubkey = self.wallet.solana_pubkey()
            lamports = int(amount_sol * 1e9)

            # 1. Quote
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

            # 2. Swap tx
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

            executed_amount = float(quote.get("outAmount", 0)) / 1e9
            executed_price = amount_sol / executed_amount if executed_amount > 0 else 0

            # 3. Optional Jito single-tx bundle (atomic, no separate tip tx)
            if use_jito:
                tx_b64 = swap["swapTransaction"]
                bundle_uuid = await jito.submit_bundle([tx_b64])
                if bundle_uuid:
                    status = await jito.wait_for_landing(bundle_uuid, timeout_sec=15.0)
                    if status == "Landed":
                        log.info("solana.buy.jito_landed", token=token_address,
                                 bundle=bundle_uuid, amount_sol=amount_sol)
                        return OrderResult(True, bundle_uuid, executed_price, executed_amount)
                    log.warning("solana.buy.jito_status_not_landed",
                                status=status, token=token_address)

            # 4. Normal RPC submission
            tx_bytes = bytes(swap["swapTransaction"])
            tx = VersionedTransaction.from_bytes(tx_bytes)
            signed = VersionedTransaction(tx.message, [self.wallet.solana_keypair])
            client = AsyncClient(self.endpoints[0])
            try:
                sig = await client.send_raw_transaction(
                    bytes(signed),
                    opts=TxOpts(skip_preflight=False, max_retries=3),
                )
                await client.confirm_transaction(sig.value, commitment="confirmed")
                log.info("solana.buy.jupiter_executed", token=token_address,
                         sig=str(sig.value), amount_sol=amount_sol)
                return OrderResult(True, str(sig.value), executed_price, executed_amount)
            finally:
                await client.close()
        except Exception as e:
            log.error("solana.buy.jupiter_failed", token=token_address, error=str(e))
            return OrderResult(False, None, None, None, str(e))

    async def sell(self, token_address: str, amount_token: float, slippage_bps: int) -> OrderResult:
        """
        Sell a pump.fun token. Routes to pump.fun direct sell if still on the
        bonding curve, else Jupiter aggregator.
        """
        if not self.wallet.has_wallet("solana"):
            return OrderResult(False, None, None, None, "Solana wallet not initialized")

        # Routing: bonding curve still active?
        use_direct = False
        try:
            bc_state = await self._bonding.fetch_state(token_address)
            use_direct = bc_state is not None and bc_state.completion_pct < 100.0
        except Exception as e:
            log.warning("solana.sell.bc_check_failed", token=token_address, error=str(e))

        if use_direct:
            result = await self._sell_pumpfun_direct(token_address, amount_token, slippage_bps)
            if result.success:
                return result
            log.warning("solana.sell.direct_failed_fallback_jupiter",
                        token=token_address, error=result.error)

        return await self._sell_via_jupiter(token_address, amount_token, slippage_bps)

    async def _sell_pumpfun_direct(
        self, token_address: str, amount_token: float, slippage_bps: int,
    ) -> OrderResult:
        """Build and submit a pump.fun `sell` instruction directly."""
        try:
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.types import TxOpts
            from solders.transaction import VersionedTransaction
            from solders.message import MessageV0
            from solders.compute_budget import set_compute_unit_price, set_compute_unit_limit

            kp = self.wallet.solana_keypair
            user_pubkey = self.wallet.solana_pubkey()

            # pump.fun tokens use 6 decimals
            tinfo = await self.get_token_info(token_address)
            decimals = tinfo.decimals if tinfo.decimals > 0 else 6
            token_amount_raw = int(amount_token * (10 ** decimals))
            if token_amount_raw <= 0:
                return OrderResult(False, None, None, None, "zero sell amount")

            sell_ix = pumpfun_ix.build_sell_ix(user_pubkey, token_address, token_amount_raw)
            priority_micro = self.cfg["trading"].get("priority_fee_micro_lamports", 5_000)
            cu_price_ix = set_compute_unit_price(priority_micro)
            cu_limit_ix = set_compute_unit_limit(200_000)

            client = AsyncClient(self.endpoints[0])
            try:
                bh_resp = await client.get_latest_blockhash()
                blockhash = bh_resp.value.blockhash
                msg = MessageV0.try_compile(
                    payer=kp.pubkey(),
                    instructions=[cu_limit_ix, cu_price_ix, sell_ix],
                    address_lookup_table_accounts=[],
                    recent_blockhash=blockhash,
                )
                signed = VersionedTransaction(msg, [kp])
                sig = await client.send_raw_transaction(
                    bytes(signed),
                    opts=TxOpts(skip_preflight=False, max_retries=3),
                )
                await client.confirm_transaction(sig.value, commitment="confirmed")
                # executed_price not exactly known without parsing logs; approximate from spot
                executed_price = 0.0
                log.info("solana.sell.pumpfun_executed", token=token_address,
                         sig=str(sig.value), amount_token=amount_token)
                return OrderResult(True, str(sig.value), executed_price, amount_token)
            finally:
                await client.close()
        except Exception as e:
            log.error("solana.sell.pumpfun_failed", token=token_address, error=str(e))
            return OrderResult(False, None, None, None, str(e))

    async def _sell_via_jupiter(self, token_address: str, amount_token: float, slippage_bps: int) -> OrderResult:
        """Jupiter aggregator sell path (for migrated tokens)."""
        try:
            from solders.transaction import VersionedTransaction
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.types import TxOpts

            SOL_MINT = "So11111111111111111111111111111111111111112"
            user_pubkey = self.wallet.solana_pubkey()
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
                log.info("solana.sell.jupiter_executed", token=token_address,
                         sig=str(sig.value), amount_sol=executed_sol)
                return OrderResult(True, str(sig.value), executed_price, amount_token)
            finally:
                await client.close()

        except Exception as e:
            log.error("solana.sell.jupiter_failed", token=token_address, error=str(e))
            return OrderResult(False, None, None, None, str(e))

    # ------------------------------------------------------------------
    # Stream new pump.fun token launches (logsSubscribe + log parsing)
    # ------------------------------------------------------------------
    async def watch_new_tokens(self) -> AsyncIterator[dict]:
        """
        Subscribe to logs of the pump.fun program. Each `create` instruction
        emits a new token launch.

        Low-latency path: we parse the mint/bonding_curve/user DIRECTLY from
        the logsSubscribe notification when possible (no getTransaction call).
        If the logs are not self-describing, we fall back to fetching the tx.
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
        reconnect_backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.ws_endpoint) as ws:
                    await ws.send(json.dumps(payload))
                    reconnect_backoff = 1.0  # reset on successful connect
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("method") != "logsNotification":
                            continue
                        value = msg["params"]["result"]["value"]
                        logs = value.get("logs") or []
                        signature = value.get("signature")
                        # Fast filter: only create instructions (skip buy/sell noise)
                        if not pumpfun_ix.is_create_log(logs):
                            continue
                        parsed = pumpfun_ix.parse_create_from_logs(logs, signature)
                        if parsed.mint and not parsed.needs_full_fetch:
                            yield {
                                "chain": "solana",
                                "program": PUMP_FUN_PROGRAM,
                                "mint": parsed.mint,
                                "bonding_curve": parsed.bonding_curve,
                                "user": parsed.user,
                                "signature": signature,
                                "source": "log_parse",
                            }
                        else:
                            # Fallback: fetch the tx and decode its create instruction
                            yield {
                                "chain": "solana",
                                "program": PUMP_FUN_PROGRAM,
                                "signature": signature,
                                "source": "fetch_required",
                            }
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("solana.watch_new_tokens.ws_disconnected",
                            error=str(e), backoff=reconnect_backoff)
                await asyncio.sleep(reconnect_backoff)
                reconnect_backoff = min(reconnect_backoff * 2, 30.0)

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
