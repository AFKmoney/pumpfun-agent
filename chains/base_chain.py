"""
base_chain.py
=============
Abstract base class for chain adapters.
Concrete adapters live in chains/solana_adapter.py and chains/evm_adapter.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenInfo:
    chain: str
    address: str         # mint address (Solana) or contract (EVM)
    symbol: str
    name: str
    decimals: int


@dataclass
class OrderResult:
    success: bool
    tx_hash: Optional[str]
    executed_price: Optional[float]
    executed_amount: Optional[float]
    error: Optional[str] = None


class BaseChainAdapter(ABC):
    """Common interface for all chains."""

    chain_name: str = "abstract"

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def get_balance(self, wallet_address: str) -> float: ...

    @abstractmethod
    async def get_token_info(self, token_address: str) -> TokenInfo: ...

    @abstractmethod
    async def buy(self, token_address: str, amount_base: float, slippage_bps: int) -> OrderResult: ...

    @abstractmethod
    async def sell(self, token_address: str, amount_token: float, slippage_bps: int) -> OrderResult: ...

    @abstractmethod
    async def get_price(self, token_address: str) -> float: ...

    @abstractmethod
    async def watch_new_tokens(self):
        """Async generator yielding newly launched tokens on the chain."""
        ...

    @abstractmethod
    async def watch_wallet_activity(self, wallet_address: str):
        """Async generator yielding trades made by a given wallet (for copy-trading)."""
        ...
