"""
types.py
=========
Shared data types for risk + persistence. Decouples them to avoid
circular imports.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"


@dataclass
class Position:
    chain: str
    token: str
    strategy: str
    entry_price: float
    size_base: float          # in SOL or ETH
    size_token: float
    opened_at: float = field(default_factory=time.time)
    stop_loss_pct: float = 15.0
    take_profit_pct: float = 50.0
    status: TradeStatus = TradeStatus.OPEN
    closed_at: Optional[float] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class TradeRecord:
    chain: str
    token: str
    strategy: str
    side: str
    size_base: float
    price: float
    ts: float
    pnl_pct: Optional[float] = None
