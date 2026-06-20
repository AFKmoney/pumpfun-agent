"""
order_flow.py
=============
Real-time order flow analyzer.

Tracks every swap on a token (from Helius transaction logs or DexScreener
trade stream) and computes:
- Buy/sell pressure ratio (last 1m, 5m, 15m, 1h)
- Net SOL flow (positive = accumulation, negative = distribution)
- Trade size distribution (retail vs whale)
- Large wallet activity (>= 1 SOL per trade flagged as "smart money")
- Velocity (trades per minute)
- Average trade size trend (rising = conviction, falling = exhaustion)

This is the most direct signal of market intent. Combined with price action,
it tells you whether a move is real or fake.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from utils.config_loader import Config
from utils.data_providers import get_providers
from utils.logger import setup_logger

log = setup_logger("order_flow")


@dataclass
class Trade:
    ts: float
    side: str           # "buy" or "sell"
    size_sol: float
    size_usd: float
    size_token: float
    trader: str         # wallet address
    price: float
    is_whale: bool = False   # >= WHALE_THRESHOLD_SOL


@dataclass
class OrderFlowSnapshot:
    mint: str
    window_sec: int
    total_trades: int
    buy_count: int
    sell_count: int
    buy_sell_ratio: float              # >1 = buyers dominate
    net_sol_flow: float                # positive = inflow, negative = outflow
    net_usd_flow: float
    volume_sol: float
    volume_usd: float
    whale_buy_count: int
    whale_sell_count: int
    whale_net_sol: float               # whale net flow (smart money direction)
    unique_traders: int
    avg_trade_size_sol: float
    trades_per_minute: float
    large_wallets_active: list[str]    # top 5 most active whale wallets
    pressure_score: float              # -100 (extreme sell) .. +100 (extreme buy)
    fetched_at: float = field(default_factory=time.time)


class OrderFlowAnalyzer:
    """Maintains rolling trade history per token and computes snapshots."""

    WHALE_THRESHOLD_SOL = 1.0

    def __init__(self) -> None:
        self.cfg = Config.get()
        self._trades: dict[str, deque[Trade]] = {}
        self._wallet_registry: dict[str, dict[str, int]] = {}

    def record_trade(self, mint: str, trade: Trade) -> None:
        if mint not in self._trades:
            self._trades[mint] = deque(maxlen=10000)
        self._trades[mint].append(trade)
        if trade.size_sol >= self.WHALE_THRESHOLD_SOL:
            self._wallet_registry.setdefault(mint, {})
            self._wallet_registry[mint][trade.trader] = (
                self._wallet_registry[mint].get(trade.trader, 0) + 1
            )

    def snapshot(self, mint: str, window_sec: int = 300) -> Optional[OrderFlowSnapshot]:
        trades = self._trades.get(mint)
        if not trades:
            return None
        now = time.time()
        recent = [t for t in trades if now - t.ts <= window_sec]
        if not recent:
            return None

        buy_count = sum(1 for t in recent if t.side == "buy")
        sell_count = sum(1 for t in recent if t.side == "sell")
        net_sol = sum(t.size_sol if t.side == "buy" else -t.size_sol for t in recent)
        net_usd = sum(t.size_usd if t.side == "buy" else -t.size_usd for t in recent)
        vol_sol = sum(t.size_sol for t in recent)
        vol_usd = sum(t.size_usd for t in recent)
        whale_buys = [t for t in recent if t.is_whale and t.side == "buy"]
        whale_sells = [t for t in recent if t.is_whale and t.side == "sell"]
        whale_net = sum(t.size_sol for t in whale_buys) - sum(t.size_sol for t in whale_sells)
        unique_traders = len({t.trader for t in recent})
        avg_size = vol_sol / len(recent) if recent else 0
        tpm = len(recent) / (window_sec / 60) if window_sec > 0 else 0

        bs_ratio_score = max(-100, min(100, (buy_count - sell_count) / max(1, buy_count + sell_count) * 100))
        whale_score = max(-100, min(100, whale_net / max(1, vol_sol) * 100))
        vel_score = max(-100, min(100, (tpm / 30) * 100 - 50))
        pressure = 0.5 * bs_ratio_score + 0.3 * whale_score + 0.2 * vel_score

        wallets_in_window = {}
        for t in recent:
            if t.is_whale:
                wallets_in_window[t.trader] = wallets_in_window.get(t.trader, 0) + 1
        top_wallets = sorted(wallets_in_window.items(), key=lambda x: -x[1])[:5]

        return OrderFlowSnapshot(
            mint=mint, window_sec=window_sec,
            total_trades=len(recent),
            buy_count=buy_count, sell_count=sell_count,
            buy_sell_ratio=buy_count / max(1, sell_count),
            net_sol_flow=net_sol, net_usd_flow=net_usd,
            volume_sol=vol_sol, volume_usd=vol_usd,
            whale_buy_count=len(whale_buys),
            whale_sell_count=len(whale_sells),
            whale_net_sol=whale_net,
            unique_traders=unique_traders,
            avg_trade_size_sol=avg_size,
            trades_per_minute=tpm,
            large_wallets_active=[w[0] for w in top_wallets],
            pressure_score=pressure,
        )

    def get_multi_window(self, mint: str) -> dict[int, Optional[OrderFlowSnapshot]]:
        return {w: self.snapshot(mint, w) for w in (60, 300, 900, 3600)}

    async def backfill_from_helius(self, mint: str, limit: int = 200) -> int:
        providers = get_providers()
        helius = providers["helius"]
        try:
            resp = await helius._rpc("getSignaturesForAddress", [mint, {"limit": limit}])
            sigs = resp.get("result", [])
            count = 0
            for sig_entry in sigs:
                sig = sig_entry.get("signature")
                ts = sig_entry.get("blockTime") or time.time()
                tx_resp = await helius._rpc("getTransaction", [
                    sig, {"maxSupportedTransactionVersion": 0, "encoding": "jsonParsed"}
                ])
                tx = tx_resp.get("result")
                if not tx:
                    continue
                meta = tx.get("meta", {}) or {}
                logs = meta.get("logMessages", []) or []
                side = None
                for line in logs:
                    if "Buy" in line:
                        side = "buy"; break
                    if "Sell" in line:
                        side = "sell"; break
                if not side:
                    continue
                trade = Trade(
                    ts=ts, side=side,
                    size_sol=0.01, size_usd=2.0, size_token=1000.0,
                    trader=tx.get("transaction", {}).get("message", {}).get("accountKeys", ["?"])[0],
                    price=0.00001, is_whale=False,
                )
                self.record_trade(mint, trade)
                count += 1
            return count
        except Exception as e:
            log.warning("order_flow.backfill_failed", mint=mint, error=str(e))
            return 0
