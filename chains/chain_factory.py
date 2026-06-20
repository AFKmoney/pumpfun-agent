"""
chain_factory.py
================
Returns the correct adapter(s) based on config.
"""
from __future__ import annotations

from chains.base_chain import BaseChainAdapter
from chains.solana_adapter import SolanaAdapter
from chains.evm_adapter import EVMAdapter
from utils.config_loader import Config


async def build_adapters() -> dict[str, BaseChainAdapter]:
    cfg = Config.get()
    adapters: dict[str, BaseChainAdapter] = {}
    for chain in cfg["trading"]["chains"]:
        if chain == "solana":
            adapters["solana"] = SolanaAdapter()
        elif chain in ("base", "ethereum"):
            adapters[chain] = EVMAdapter(chain)
        else:
            raise ValueError(f"Unsupported chain: {chain}")
        await adapters[chain].connect()
    return adapters
