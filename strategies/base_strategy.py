"""
base_strategy.py
================
Abstract strategy interface. Each strategy runs as an async task and emits
'trade signals' that the orchestrator routes through the risk manager
before execution.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from chains.base_chain import BaseChainAdapter


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    strategy: str
    chain: str
    token_address: str
    signal_type: SignalType
    suggested_size_pct: float       # % of allocated capital (0..1)
    confidence: float               # 0..1
    reason: str
    target_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    # Free-form metadata bag. Strategies populate this with context the executor
    # needs but that doesn't fit the fixed fields above:
    #   - "dev_wallet": str  (the token creator's wallet, for DevTracker)
    #   - "token_symbol": str, "token_name": str, "uri": str (for TokenScorer)
    #   - "bonding_curve": str (PDA, for fast direct execution)
    #   - "source_score": dict (per-analyzer scores at signal time, for attribution)
    metadata: Optional[dict] = None


class BaseStrategy(ABC):
    """Common interface for all trading strategies."""

    name: str = "abstract"

    def __init__(self, adapter: BaseChainAdapter) -> None:
        self.adapter = adapter

    @abstractmethod
    async def run(self) -> None:
        """Main async loop. Should respect asyncio.CancelledError for clean shutdown."""
        ...

    @abstractmethod
    async def evaluate(self, *args, **kwargs) -> Signal:
        """Produce a Signal from given market data."""
        ...
