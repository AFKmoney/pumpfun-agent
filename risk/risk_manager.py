"""
risk_manager.py
===============
Central risk gate. Every signal emitted by a strategy must pass through
the RiskManager before any order is sent to the chain.

Implements:
- Fixed position sizing (Strategy A).
- Kelly criterion sizing with half-Kelly safety (Strategy B).
- Daily loss cap (circuit breaker).
- Per-token stop-loss and take-profit.
- Blacklist (tokens that failed rugpull check or hit stop-loss repeatedly).
- Trade history ledger (persisted to SQLite so it survives restarts).
- Positions persisted to SQLite.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from strategies.base_strategy import Signal, SignalType
from risk.types import Position, TradeRecord, TradeStatus
from utils.config_loader import Config
from utils.kill_switch import KillSwitch
from utils.logger import setup_logger
from utils.persistence import Persistence

log = setup_logger("risk_manager")


class RiskManager:
    def __init__(self) -> None:
        self.cfg = Config.get()
        self.rcfg = self.cfg["risk"]
        self.tcfg = self.cfg["trading"]
        self.db = Persistence.get()
        self.positions: dict[str, Position] = {}
        self.history: deque[TradeRecord] = deque(maxlen=500)
        self.daily_pnl_pct: float = 0.0
        self._day_start: float = self._utc_midnight_ts()
        self._lock = asyncio.Lock()

        # Restore state from DB
        self._restore_state()

        self._trade_freq: deque[float] = deque(maxlen=200)
        log.info("risk_manager.initialized",
                 daily_cap=self.rcfg["daily_loss_cap_pct"],
                 kelly_enabled=self.rcfg["kelly"]["enabled"],
                 restored_positions=len(self.positions),
                 restored_daily_pnl=self.daily_pnl_pct)

    # ------------------------------------------------------------------
    def _utc_midnight_ts(self) -> float:
        now = datetime.now(timezone.utc)
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp()

    def _today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _restore_state(self) -> None:
        """Load positions + daily PnL + recent trade history from SQLite."""
        try:
            # Positions
            for key, p in self.db.load_open_positions().items():
                self.positions[key] = Position(
                    chain=p["chain"], token=p["token"], strategy=p["strategy"],
                    entry_price=p["entry_price"], size_base=p["size_base"],
                    size_token=p["size_token"], opened_at=p["opened_at"],
                    stop_loss_pct=p["stop_loss_pct"],
                    take_profit_pct=p["take_profit_pct"],
                )
            # Daily PnL
            today_pnl = self.db.load_daily_pnl(self._today_str())
            if today_pnl is not None:
                self.daily_pnl_pct = today_pnl
            # Recent trade history (for Kelly)
            for t in reversed(self.db.load_recent_trades(limit=500)):
                self.history.append(TradeRecord(
                    chain=t["chain"], token=t["token"], strategy=t["strategy"],
                    side=t["side"], size_base=t["size_base"], price=t["price"],
                    ts=t["ts"], pnl_pct=t["pnl_pct"],
                ))
        except Exception as e:
            log.error("risk_manager.restore_failed", error=str(e))

    def _reset_daily_if_needed(self) -> None:
        today = self._today_str()
        if time.time() - self._day_start > 86_400:
            self._day_start = self._utc_midnight_ts()
            self.daily_pnl_pct = 0.0
            self.db.upsert_daily_pnl(today, 0.0)
            log.info("risk_manager.daily_reset")

    def _is_blacklisted(self, token: str) -> bool:
        return self.db.is_blacklisted(token)

    def _too_many_trades_per_minute(self) -> bool:
        now = time.time()
        while self._trade_freq and now - self._trade_freq[0] > 60:
            self._trade_freq.popleft()
        max_per_min = self.cfg.get_nested("safety", "max_trade_frequency_per_minute", default=10)
        return len(self._trade_freq) >= max_per_min

    # ------------------------------------------------------------------
    async def approve(self, signal: Signal, allocated_capital: float) -> tuple[bool, float, str]:
        """Returns (approved, size_base, reason)."""
        async with self._lock:
            self._reset_daily_if_needed()

            if KillSwitch.is_triggered():
                return False, 0.0, f"Kill switch active: {KillSwitch.reason()}"

            if signal.signal_type == SignalType.SELL:
                key = f"{signal.chain}:{signal.token_address}"
                if key not in self.positions:
                    return False, 0.0, "No open position to sell"
                return True, self.positions[key].size_base, "Sell approved"

            if signal.signal_type == SignalType.HOLD:
                return False, 0.0, "Hold signal"

            # ---- BUY approvals ----
            if self._is_blacklisted(signal.token_address):
                return False, 0.0, "Token blacklisted"

            if self._too_many_trades_per_minute():
                return False, 0.0, "Trade frequency limit"

            if len(self.positions) >= self.tcfg["max_open_positions"]:
                return False, 0.0, "Max open positions reached"

            if self.daily_pnl_pct <= -abs(self.rcfg["daily_loss_cap_pct"]):
                KillSwitch.trigger(f"Daily loss cap reached: {self.daily_pnl_pct:.2f}%")
                return False, 0.0, "Daily loss cap reached"

            # ---- Position sizing ----
            size_base = self._compute_size(signal, allocated_capital)
            if size_base <= 0:
                return False, 0.0, "Computed size <= 0"

            return True, size_base, "Approved"

    def _compute_size(self, signal: Signal, allocated_capital: float) -> float:
        kcfg = self.rcfg["kelly"]
        if not kcfg["enabled"]:
            return self.tcfg.get("fixed_size_sol" if signal.chain == "solana" else "fixed_size_evm", 0.01)

        recent = [r for r in self.history if r.strategy == signal.strategy and r.pnl_pct is not None]
        recent = recent[-kcfg["lookback_trades"]:]
        if len(recent) < 10:
            base = self.tcfg.get("fixed_size_sol" if signal.chain == "solana" else "fixed_size_evm", 0.01)
            return base * max(0.5, signal.confidence)

        wins = [r.pnl_pct for r in recent if r.pnl_pct > 0]
        losses = [r.pnl_pct for r in recent if r.pnl_pct < 0]
        if not wins or not losses:
            base = self.tcfg.get("fixed_size_sol" if signal.chain == "solana" else "fixed_size_evm", 0.01)
            return base * max(0.5, signal.confidence)

        p = len(wins) / len(recent)
        b = (sum(wins) / len(wins)) / abs(sum(losses) / len(losses))
        kelly_f = max(0.0, (p * b - (1 - p)) / b) * kcfg["fraction"]
        kelly_f = max(kcfg["min_size_pct"] / 100, min(kcfg["max_size_pct"] / 100, kelly_f))
        return allocated_capital * kelly_f

    # ------------------------------------------------------------------
    async def open_position(self, signal: Signal, size_base: float, executed_price: float, size_token: float) -> None:
        async with self._lock:
            key = f"{signal.chain}:{signal.token_address}"
            pos = Position(
                chain=signal.chain, token=signal.token_address, strategy=signal.strategy,
                entry_price=executed_price, size_base=size_base, size_token=size_token,
                stop_loss_pct=signal.stop_loss_pct or self.rcfg["per_token_max_loss_pct"],
                take_profit_pct=signal.take_profit_pct or 30.0,
            )
            self.positions[key] = pos
            self._trade_freq.append(time.time())
            rec = TradeRecord(
                chain=signal.chain, token=signal.token_address, strategy=signal.strategy,
                side="BUY", size_base=size_base, price=executed_price, ts=time.time(),
            )
            self.history.append(rec)
            # Persist
            try:
                self.db.upsert_position(pos)
                self.db.insert_trade(rec)
            except Exception as e:
                log.error("risk_manager.persist_open_failed", error=str(e))
            log.info("risk_manager.position_opened", token=signal.token_address, size=size_base, price=executed_price)

    async def close_position(self, chain: str, token: str, exit_price: float, reason: str) -> Optional[float]:
        async with self._lock:
            key = f"{chain}:{token}"
            pos = self.positions.pop(key, None)
            if not pos:
                return None
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
            # Weight the per-token pnl by the position's share of allocated capital.
            # `_allocated_capital_for_chain` returns the SOL/ETH budget for the
            # position's chain. daily_pnl_pct stays a true % of capital.
            cap = self._allocated_capital_for_chain(pos.chain)
            weight = (pos.size_base / cap) if cap > 0 else 0.0
            self.daily_pnl_pct += pnl_pct * weight
            rec = TradeRecord(
                chain=chain, token=token, strategy=pos.strategy,
                side="SELL", size_base=pos.size_base, price=exit_price,
                ts=time.time(), pnl_pct=pnl_pct,
            )
            self.history.append(rec)
            # Persist: delete position + insert trade + update daily PnL
            try:
                self.db.delete_position(chain, token)
                self.db.insert_trade(rec)
                self.db.upsert_daily_pnl(self._today_str(), self.daily_pnl_pct)
                if pnl_pct < -self.rcfg["per_token_max_loss_pct"] * 0.5:
                    self.db.add_blacklist(token, time.time() + self.rcfg["blacklist_duration_minutes"] * 60, reason)
            except Exception as e:
                log.error("risk_manager.persist_close_failed", error=str(e))
            log.info("risk_manager.position_closed", token=token, pnl_pct=round(pnl_pct, 2), reason=reason)
            return pnl_pct

    async def reduce_position(
        self, chain: str, token: str, fraction: float, exit_price: float, reason: str,
    ) -> Optional[float]:
        """
        Partially close a position (e.g. TP ladder tranche). Reduces size_base
        and size_token proportionally, records a SELL trade, updates daily PnL
        for the realized portion, and persists the new reduced position.

        Args:
            fraction: 0..1 of the position to sell.
        Returns: realized pnl_pct for the sold portion, or None if no position.
        """
        if fraction <= 0:
            return None
        if fraction >= 1.0:
            return await self.close_position(chain, token, exit_price, reason)
        async with self._lock:
            key = f"{chain}:{token}"
            pos = self.positions.get(key)
            if not pos:
                return None
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
            cap = self._allocated_capital_for_chain(pos.chain)
            realized_size_base = pos.size_base * fraction
            weight = (realized_size_base / cap) if cap > 0 else 0.0
            self.daily_pnl_pct += pnl_pct * weight
            # Shrink remaining position
            pos.size_base *= (1.0 - fraction)
            pos.size_token *= (1.0 - fraction)
            rec = TradeRecord(
                chain=chain, token=token, strategy=pos.strategy,
                side="SELL", size_base=realized_size_base, price=exit_price,
                ts=time.time(), pnl_pct=pnl_pct,
            )
            self.history.append(rec)
            try:
                self.db.upsert_position(pos)
                self.db.insert_trade(rec)
                self.db.upsert_daily_pnl(self._today_str(), self.daily_pnl_pct)
            except Exception as e:
                log.error("risk_manager.persist_reduce_failed", error=str(e))
            log.info("risk_manager.position_reduced", token=token,
                     fraction=round(fraction, 3), pnl_pct=round(pnl_pct, 2),
                     remaining_base=round(pos.size_base, 4), reason=reason)
            return pnl_pct

    def _allocated_capital_for_chain(self, chain: str) -> float:
        """Allocated capital (SOL or ETH) for a chain, from config. Fallback 1.0."""
        try:
            chains_cfg = self.cfg.get("chains", {})
            key = "allocated_capital_sol" if chain == "solana" else "allocated_capital_eth"
            return float(chains_cfg.get(chain, {}).get(key, 0.0)) or 1.0
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    async def check_exits(self, price_provider) -> None:
        async with self._lock:
            to_close: list[tuple[str, str, float, str]] = []
            for key, pos in list(self.positions.items()):
                price = await price_provider(pos.chain, pos.token)
                if price <= 0:
                    continue
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                if pnl_pct <= -pos.stop_loss_pct:
                    to_close.append((pos.chain, pos.token, price, f"Stop loss @ {pnl_pct:.1f}%"))
                elif pnl_pct >= pos.take_profit_pct:
                    to_close.append((pos.chain, pos.token, price, f"Take profit @ {pnl_pct:.1f}%"))
                elif time.time() - pos.opened_at > self.rcfg["per_token_max_hold_minutes"] * 60:
                    to_close.append((pos.chain, pos.token, price, "Max hold time"))

        for chain, token, price, reason in to_close:
            await self.close_position(chain, token, price, reason)
