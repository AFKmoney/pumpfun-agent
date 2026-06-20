"""
evm_adapter.py
==============
EVM adapter (Base / Ethereum) for pump.fun-like launches (pumpa.fun, etc).
Uses web3.py + ethers-style providers.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

import aiohttp
from web3 import AsyncWeb3, AsyncHTTPProvider
from web3.middleware import ExtraDataToPOAMiddleware

from chains.base_chain import BaseChainAdapter, OrderResult, TokenInfo
from utils.config_loader import Config
from utils.logger import setup_logger
from utils.wallet_manager import WalletManager

log = setup_logger("evm_adapter")


class EVMAdapter(BaseChainAdapter):
    """Generic EVM adapter. Pass chain_name='base' or 'ethereum'."""

    def __init__(self, chain_name: str) -> None:
        assert chain_name in ("base", "ethereum")
        self.chain_name = chain_name
        self.cfg = Config.get()
        self.endpoints: list[str] = self.cfg["chains"][chain_name]["rpc_endpoints"]
        self.wallet = WalletManager()
        self._w3: Optional[AsyncWeb3] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> None:
        self._w3 = AsyncWeb3(AsyncHTTPProvider(self.endpoints[0]))
        if self.chain_name == "base":
            self._w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        if not await self._w3.is_connected():
            raise RuntimeError(f"EVM adapter for {self.chain_name} failed to connect.")
        self._session = aiohttp.ClientSession()
        log.info("evm.connected", chain=self.chain_name, endpoint=self.endpoints[0])

    # ------------------------------------------------------------------
    async def get_balance(self, wallet_address: str) -> float:
        bal = await self._w3.eth.get_balance(wallet_address)
        return float(bal) / 1e18

    async def get_token_info(self, token_address: str) -> TokenInfo:
        # Minimal ERC-20 ABI
        erc20_abi = [
            {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "name", "outputs": [{"type": "string"}], "type": "function"},
            {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"type": "uint8"}], "type": "function"},
        ]
        contract = self._w3.eth.contract(address=self._w3.to_checksum_address(token_address), abi=erc20_abi)
        try:
            symbol = await contract.functions.symbol().call()
            name = await contract.functions.name().call()
            decimals = await contract.functions.decimals().call()
        except Exception as e:
            log.warning("evm.token_info_failed", token=token_address, error=str(e))
            symbol, name, decimals = "UNKNOWN", "UNKNOWN", 18
        return TokenInfo(self.chain_name, token_address, symbol, name, decimals)

    async def get_price(self, token_address: str) -> float:
        # Use DexScreener API as a fallback price source.
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        try:
            async with self._session.get(url) as r:
                data = await r.json()
            pairs = data.get("pairs", [])
            if not pairs:
                return 0.0
            return float(pairs[0]["priceUsd"])
        except Exception as e:
            log.warning("evm.price.fetch_failed", token=token_address, error=str(e))
            return 0.0

    # ------------------------------------------------------------------
    # Buy / Sell: route via Uniswap V3 router (Base) or Uniswap V2 (Ethereum).
    # For brevity, this shows the V3-style swap. In production, integrate
    # the specific pump.fun-clone contract if applicable.
    # ------------------------------------------------------------------
    async def buy(self, token_address: str, amount_eth: float, slippage_bps: int) -> OrderResult:
        if not self.wallet.has_wallet(self.chain_name):
            return OrderResult(False, None, None, None, f"{self.chain_name} wallet not initialized")
        try:
            acct = self.wallet.evm_accounts[self.chain_name]
            token_addr = self._w3.to_checksum_address(token_address)
            WETH = self._w3.to_checksum_address("0x4200000000000000000000000000000000000006")  # Base WETH
            router = self._w3.to_checksum_address("0x2626664c2603336E57B271c5C0b26F421741e481")  # Uniswap V3 router on Base

            router_abi = [
                {"inputs":[{"name":"recipient","type":"address"},
                            {"name":"tokenIn","type":"address"},
                            {"name":"tokenOut","type":"address"},
                            {"name":"amountIn","type":"uint256"},
                            {"name":"amountOutMin","type":"uint256"},
                            {"name":"sqrtPriceLimitX96","type":"uint160"}],
                 "name":"exactInputSingle",
                 "outputs":[{"name":"amountOut","type":"uint256"}],
                 "type":"function"}
            ]
            contract = self._w3.eth.contract(address=router, abi=router_abi)
            amount_in_wei = self._w3.to_wei(amount_eth, "ether")
            deadline_amount_out_min = 0  # in production: query the pool for expected out, apply slippage

            nonce = await self._w3.eth.get_transaction_count(acct.address)
            tx = await contract.functions.exactInputSingle(
                acct.address, WETH, token_addr, amount_in_wei, deadline_amount_out_min, 0
            ).build_transaction({
                "from": acct.address,
                "nonce": nonce,
                "gas": 250_000,
                "gasPrice": self._w3.to_wei(self.cfg["trading"]["gas_price_gwei"], "gwei"),
                "chainId": await self._w3.eth.chain_id,
            })
            signed = acct.sign_transaction(tx)
            tx_hash = await self._w3.eth.send_raw_transaction(signed.rawTransaction)
            receipt = await self._w3.eth.wait_for_transaction_receipt(tx_hash)
            log.info("evm.buy.executed", chain=self.chain_name, token=token_address, tx=receipt.transactionHash.hex())
            return OrderResult(True, receipt.transactionHash.hex(), None, None)
        except Exception as e:
            log.error("evm.buy.failed", chain=self.chain_name, token=token_address, error=str(e))
            return OrderResult(False, None, None, None, str(e))

    async def sell(self, token_address: str, amount_token: float, slippage_bps: int) -> OrderResult:
        """
        Sell EVM token for WETH via Uniswap V3.
        Flow:
            1. Fetch token decimals.
            2. Check current allowance to Uniswap V3 router.
            3. If allowance < amount: send approve() tx (MaxUint256).
            4. Build + sign + send exactInputSingle(tokenIn=token, tokenOut=WETH).
            5. Wait for receipt; compute executed WETH from swap event log.
        """
        if not self.wallet.has_wallet(self.chain_name):
            return OrderResult(False, None, None, None, f"{self.chain_name} wallet not initialized")
        try:
            acct = self.wallet.evm_accounts[self.chain_name]
            token_addr = self._w3.to_checksum_address(token_address)
            WETH = self._w3.to_checksum_address(
                # Base WETH = 0x4200...0006, Ethereum WETH = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
                "0x4200000000000000000000000000000000000006"
                if self.chain_name == "base"
                else "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
            )
            router = self._w3.to_checksum_address(
                # Uniswap V3 SwapRouter02
                # Base: 0x2626664c2603336E57B271c5C0b26F421741e481
                # Ethereum: 0xE592427A0AEce92De3Edee1F18E0157C05861564
                "0x2626664c2603336E57B271c5C0b26F421741e481"
                if self.chain_name == "base"
                else "0xE592427A0AEce92De3Edee1F18E0157C05861564"
            )

            # ---- 1. Token decimals ----
            tinfo = await self.get_token_info(token_address)
            decimals = tinfo.decimals
            raw_amount = int(amount_token * (10 ** decimals))

            # ---- 2. Check allowance ----
            erc20_abi = [
                {"constant": True, "inputs": [{"name": "owner", "type": "address"},
                                              {"name": "spender", "type": "address"}],
                 "name": "allowance", "outputs": [{"type": "uint256"}], "type": "function"},
                {"inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
                 "name": "approve", "outputs": [{"type": "bool"}], "type": "function"},
            ]
            token_contract = self._w3.eth.contract(address=token_addr, abi=erc20_abi)
            current_allowance = await token_contract.functions.allowance(acct.address, router).call()

            # ---- 3. Approve if needed ----
            if current_allowance < raw_amount:
                log.info("evm.sell.approving", chain=self.chain_name, token=token_address,
                         current=current_allowance, required=raw_amount)
                MAX_UINT256 = 2 ** 256 - 1
                nonce = await self._w3.eth.get_transaction_count(acct.address)
                approve_tx = await token_contract.functions.approve(router, MAX_UINT256).build_transaction({
                    "from": acct.address,
                    "nonce": nonce,
                    "gas": 80_000,
                    "gasPrice": self._w3.to_wei(self.cfg["trading"]["gas_price_gwei"], "gwei"),
                    "chainId": await self._w3.eth.chain_id,
                })
                signed_approve = acct.sign_transaction(approve_tx)
                approve_hash = await self._w3.eth.send_raw_transaction(signed_approve.rawTransaction)
                approve_receipt = await self._w3.eth.wait_for_transaction_receipt(approve_hash)
                if approve_receipt.status != 1:
                    return OrderResult(False, None, None, None, f"Approve tx reverted: {approve_hash.hex()}")
                log.info("evm.sell.approved", chain=self.chain_name, tx=approve_hash.hex())

            # ---- 4. Build swap tx ----
            router_abi = [
                {"inputs":[{"name":"recipient","type":"address"},
                            {"name":"tokenIn","type":"address"},
                            {"name":"tokenOut","type":"address"},
                            {"name":"amountIn","type":"uint256"},
                            {"name":"amountOutMin","type":"uint256"},
                            {"name":"sqrtPriceLimitX96","type":"uint160"}],
                 "name":"exactInputSingle",
                 "outputs":[{"name":"amountOut","type":"uint256"}],
                 "type":"function"}
            ]
            swap_contract = self._w3.eth.contract(address=router, abi=router_abi)
            # amountOutMin: query expected output and apply slippage
            # For simplicity we use 0 here and rely on slippage being checked off-chain.
            # In production: query Quoter contract for expected out, then apply slippage_bps.
            amount_out_min = 0  # TODO: use Quoter contract

            nonce = await self._w3.eth.get_transaction_count(acct.address)
            swap_tx = await swap_contract.functions.exactInputSingle(
                acct.address,           # recipient
                token_addr,             # tokenIn
                WETH,                   # tokenOut
                raw_amount,             # amountIn
                amount_out_min,         # amountOutMin
                0,                      # sqrtPriceLimitX96 (no limit)
            ).build_transaction({
                "from": acct.address,
                "nonce": nonce,
                "gas": 300_000,
                "gasPrice": self._w3.to_wei(self.cfg["trading"]["gas_price_gwei"], "gwei"),
                "chainId": await self._w3.eth.chain_id,
            })
            signed_swap = acct.sign_transaction(swap_tx)
            swap_hash = await self._w3.eth.send_raw_transaction(signed_swap.rawTransaction)
            receipt = await self._w3.eth.wait_for_transaction_receipt(swap_hash)

            if receipt.status != 1:
                return OrderResult(False, None, None, None,
                                   f"Swap tx reverted: {swap_hash.hex()}")

            # ---- 5. Decode executed WETH from swap event ----
            # Uniswap V3 emits a Swap event; we try to find it in logs.
            executed_weth = 0.0
            executed_price = 0.0
            try:
                swap_event_abi = {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": True, "name": "sender", "type": "address"},
                        {"indexed": True, "name": "recipient", "type": "address"},
                        {"indexed": False, "name": "amount0", "type": "int256"},
                        {"indexed": False, "name": "amount1", "type": "int256"},
                        {"indexed": False, "name": "sqrtPriceX96", "type": "uint160"},
                        {"indexed": False, "name": "liquidity", "type": "uint128"},
                        {"indexed": False, "name": "tick", "type": "int24"},
                    ],
                    "name": "Swap",
                    "type": "event",
                }
                # Decode logs
                for log_entry in receipt.logs:
                    if len(log_entry["topics"]) < 1:
                        continue
                    # Swap event topic = keccak256("Swap(address,address,int256,int256,uint160,uint128,int24)")
                    swap_topic = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
                    if log_entry["topics"][0].hex() != swap_topic:
                        continue
                    # amount0/amount1 are int256 at offsets 0 and 32 of data
                    data = bytes(log_entry["data"])
                    if len(data) < 64:
                        continue
                    amount0 = int.from_bytes(data[0:32], "big", signed=True)
                    amount1 = int.from_bytes(data[32:64], "big", signed=True)
                    # WETH is typically token1 on V3 pools with token0 = paired ERC-20.
                    # If WETH is token1, amount1 is the WETH delta (positive = received).
                    out_raw = amount1 if amount1 > 0 else -amount0
                    executed_weth = abs(out_raw) / 1e18
                    executed_price = executed_weth / amount_token if amount_token > 0 else 0
                    break
            except Exception as e:
                log.warning("evm.sell.log_decode_failed", error=str(e))

            log.info("evm.sell.executed", chain=self.chain_name, token=token_address,
                     tx=receipt.transactionHash.hex(), weth_out=executed_weth)
            return OrderResult(True, receipt.transactionHash.hex(), executed_price, amount_token)

        except Exception as e:
            log.error("evm.sell.failed", chain=self.chain_name, token=token_address, error=str(e))
            return OrderResult(False, None, None, None, str(e))

    # ------------------------------------------------------------------
    # Stream new pool creations via DEX Screener / factory contract events.
    # ------------------------------------------------------------------
    async def watch_new_tokens(self) -> AsyncIterator[dict]:
        """Poll DexScreener for new pairs on this chain every N seconds."""
        import asyncio as _asyncio
        url = f"https://api.dexscreener.com/token-boosts/latest/v1"
        while True:
            try:
                async with self._session.get(url) as r:
                    data = await r.json()
                for entry in data:
                    if entry.get("chainId", "").lower() == self.chain_name:
                        yield {"chain": self.chain_name, "token": entry.get("tokenAddress"), "raw": entry}
            except Exception as e:
                log.warning("evm.new_tokens.poll_failed", error=str(e))
            await _asyncio.sleep(15)

    async def watch_wallet_activity(self, wallet_address: str) -> AsyncIterator[dict]:
        """Poll Etherscan/BaseScan for new transactions on a wallet."""
        import asyncio as _asyncio
        # In production use the chain's block explorer API with an API key.
        last_nonce = await self._w3.eth.get_transaction_count(wallet_address)
        while True:
            await _asyncio.sleep(10)
            current = await self._w3.eth.get_transaction_count(wallet_address)
            if current > last_nonce:
                # New tx detected; fetch last block
                block = await self._w3.eth.get_block("latest", full_transactions=True)
                for tx in block.transactions:
                    if tx.get("from", "").lower() == wallet_address.lower():
                        yield {"chain": self.chain_name, "wallet": wallet_address, "tx": dict(tx)}
                last_nonce = current

    async def close(self) -> None:
        if self._session:
            await self._session.close()
