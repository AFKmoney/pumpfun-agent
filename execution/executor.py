"""
executor.py
===========
Bridge between RiskManager (approved size) and the chain adapter (actual tx).

Responsibilities:
1. Anti-rugpull gate before any BUY.
2. Token scoring (BUY signals filtered by quality score).
3. Anti-sandwich slippage computation (adaptive per trade).
4. Gas/priority fee optimization (urgent trades get higher fees).
5. Routing the signal to the chain adapter.
6. On position open: register with SmartExitManager, DevTracker, Pyramider.
7. On position close: unregister everywhere.
8. Pyramiding on winners (emits new BUY signals).
"""
from __future__ import annotations

import asyncio
import time

from chains.base_chain import BaseChainAdapter, OrderResult
from risk.risk_manager import RiskManager
from risk.smart_exits import SmartExitManager, ExitSignal
from strategies.anti_rugpull import AntiRugpullStrategy
from strategies.base_strategy import Signal, SignalType
from utils.anti_sandwich import AntiSandwich
from utils.bonding_curve import BondingCurveAnalyzer
from utils.config_loader import Config
from utils.dev_tracker import DevTracker
from utils.gas_optimizer import GasOptimizer
from utils.logger import setup_logger
from utils.notifier import TelegramNotifier
from utils.token_scorer import TokenScorer

log = setup_logger("executor")


class Executor:
    def __init__(
        self,
        adapter: BaseChainAdapter,
        risk_manager: RiskManager,
        notifier: TelegramNotifier,
        anti_rug: AntiRugpullStrategy,
        token_scorer: TokenScorer,
        smart_exits: SmartExitManager,
        dev_tracker: DevTracker,
        gas_optimizer: GasOptimizer,
        anti_sandwich: AntiSandwich,
    ) -> None:
        self.adapter = adapter
        self.risk = risk_manager
        self.notifier = notifier
        self.anti_rug = anti_rug
        self.scorer = token_scorer
        self.smart_exits = smart_exits
        self.dev_tracker = dev_tracker
        self.gas_opt = gas_optimizer
        self.anti_sandwich = anti_sandwich
        self.bonding = BondingCurveAnalyzer()
        self.cfg = Config.get()
        self.paper_mode = self.cfg["trading"]["mode"] == "paper"

    # ------------------------------------------------------------------
    async def execute(self, signal: Signal, allocated_capital: float) -> None:
        try:
            approved, size_base, reason = await self.risk.approve(signal, allocated_capital)
            if not approved:
                log.info("executor.signal_rejected", reason=reason, token=signal.token_address)
                return

            # ---- BUY flow ----
            if signal.signal_type == SignalType.BUY:
                await self._handle_buy(signal, size_base)

            # ---- SELL flow ----
            elif signal.signal_type == SignalType.SELL:
                await self._handle_sell(signal, signal.reason)

        except Exception as e:
            log.error("executor.exception", error=str(e), token=signal.token_address)
            await self.notifier.notify_error(f"Executor exception: {e}")

    # ------------------------------------------------------------------
    async def _handle_buy(self, signal: Signal, size_base: float) -> None:
        meta = signal.metadata or {}
        # 1. Anti-rugpull gate (sniping skips the slow honeypot micro-buy)
        skip_honeypot = (signal.strategy == "sniping")
        verdict = await self.anti_rug.check(signal.token_address, skip_honeypot=skip_honeypot)
        if not verdict.is_safe:
            await self.notifier.notify_rugpull(signal.token_address, verdict.reasons)
            log.warning("executor.rugpull_blocked", token=signal.token_address, reasons=verdict.reasons)
            return

        # 2. Token scoring (skip for pyramiding on existing positions, those are already vetted)
        key = f"{signal.chain}:{signal.token_address}"
        if key not in self.risk.positions:
            try:
                token_meta = {
                    "symbol": meta.get("token_symbol") or "",
                    "name": meta.get("token_name") or "",
                    "uri": meta.get("uri") or "",
                }
                score = await self.scorer.score(signal.token_address, token_meta=token_meta)
                log.info("executor.token_score", token=signal.token_address,
                         score=score.score, recommendation=score.recommendation,
                         confidence=score.confidence)
                if score.recommendation == "AVOID":
                    log.info("executor.skipped_low_score", token=signal.token_address,
                             score=score.score)
                    return
                # Scale size by score (high score = larger position, up to 1.5x)
                if score.recommendation == "BUY_STRONG":
                    size_base *= 1.5
                    log.info("executor.size_boosted", token=signal.token_address,
                             multiplier=1.5, new_size=size_base)
            except Exception as e:
                log.warning("executor.scoring_failed", error=str(e))
                # Fail-open: proceed without score

        # 3. Anti-sandwich slippage computation
        urgent = (signal.strategy == "sniping")
        try:
            bc_state = await self.bonding.fetch_state(signal.token_address)
            liq_sol = bc_state.real_sol_reserves if bc_state else 0
        except Exception:
            bc_state, liq_sol = None, 0

        slip_rec = self.anti_sandwich.compute(
            trade_size_sol=size_base if signal.chain == "solana" else 0,
            liquidity_sol=liq_sol,
            bc_state=bc_state,
            urgent=urgent,
        )
        log.info("executor.slippage_set", token=signal.token_address,
                 bps=slip_rec.slippage_bps, reason=slip_rec.reason, use_jito=slip_rec.use_jito)

        # 4. Execute buy
        if self.paper_mode:
            log.info("executor.paper_buy", token=signal.token_address, size=size_base,
                     slippage_bps=slip_rec.slippage_bps)
            price = await self.adapter.get_price(signal.token_address) or 0.0001
            await self.risk.open_position(signal, size_base, price, size_base / max(price, 1e-9))
            await self._register_position_tracking(signal, size_base, price)
            return

        result: OrderResult = await self.adapter.buy(
            signal.token_address, size_base, slip_rec.slippage_bps
        )
        if result.success:
            await self.risk.open_position(
                signal, size_base,
                result.executed_price or 0.0,
                result.executed_amount or 0.0,
            )
            await self._register_position_tracking(signal, size_base, result.executed_price or 0)
            await self.notifier.notify_trade_opened(
                signal, size_base, result.executed_price or 0, result.tx_hash or ""
            )
        else:
            log.error("executor.buy_failed", error=result.error)
            await self.notifier.notify_error(f"Buy failed: {result.error}")

    # ------------------------------------------------------------------
    async def _handle_sell(self, signal: Signal, reason: str, size_pct: float = 1.0) -> None:
        """size_pct: 0..1, fraction of position to sell."""
        key_pos = self.risk.positions.get(f"{signal.chain}:{signal.token_address}")
        if not key_pos:
            return

        sell_amount_token = key_pos.size_token * size_pct
        urgent = "DEV SOLD" in reason or "stop" in reason.lower()

        # Adaptive slippage for sells too (sells need to land fast on panic)
        slip_bps = self.cfg["trading"]["slippage_bps"]
        try:
            slip_rec = self.anti_sandwich.compute(
                trade_size_sol=key_pos.size_base * size_pct if signal.chain == "solana" else 0,
                liquidity_sol=0,  # we don't have liq for sells in this scope
                urgent=urgent,
            )
            slip_bps = slip_rec.slippage_bps
        except Exception:
            pass

        if self.paper_mode:
            price = await self.adapter.get_price(signal.token_address) or key_pos.entry_price
            if size_pct >= 1.0:
                pnl = await self.risk.close_position(signal.chain, signal.token_address, price, reason)
                self._unregister_position_tracking(signal.chain, signal.token_address)
            else:
                pnl = await self.risk.reduce_position(
                    signal.chain, signal.token_address, size_pct, price, reason,
                )
            if pnl is not None:
                await self.notifier.notify_trade_closed(signal.chain, signal.token_address, pnl, reason)
            return

        result = await self.adapter.sell(signal.token_address, sell_amount_token, slip_bps)
        if result.success:
            if size_pct >= 1.0:
                pnl = await self.risk.close_position(
                    signal.chain, signal.token_address,
                    result.executed_price or key_pos.entry_price, reason,
                )
                self._unregister_position_tracking(signal.chain, signal.token_address)
            else:
                # Partial sell: reduce the position in the bookkeeping (was a no-op before)
                pnl = await self.risk.reduce_position(
                    signal.chain, signal.token_address, size_pct,
                    result.executed_price or key_pos.entry_price, reason,
                )
                log.info("executor.partial_sell_bookkept", token=signal.token_address,
                         pct=size_pct, remaining_base=round(key_pos.size_base, 4))
            if pnl is not None:
                await self.notifier.notify_trade_closed(signal.chain, signal.token_address, pnl, reason)
        else:
            log.error("executor.sell_failed", error=result.error)
            await self.notifier.notify_error(f"Sell failed: {result.error}")

    # ------------------------------------------------------------------
    # Position tracking registration
    # ------------------------------------------------------------------
    def _register_position_tracking(self, signal: Signal, size_base: float, entry_price: float) -> None:
        """Register the new position with SmartExitManager + DevTracker."""
        key = f"{signal.chain}:{signal.token_address}"
        pos = self.risk.positions.get(key)
        if pos:
            self.smart_exits.init_position(pos)
        # DevTracker: now armed. The dev wallet comes from the signal metadata
        # (populated by SnipingStrategy from the create event). Without it we
        # cannot track — log and skip gracefully.
        meta = signal.metadata or {}
        dev_wallet = meta.get("dev_wallet")
        if dev_wallet and signal.chain == "solana":
            try:
                # Initial balance unknown; the tracker will establish it on first ws tick.
                self.dev_tracker.track(signal.chain, signal.token_address, dev_wallet, initial_balance=0.0)
                log.info("executor.dev_tracker_armed",
                         token=signal.token_address, dev_wallet=dev_wallet[:8])
            except Exception as e:
                log.warning("executor.dev_tracker_arm_failed",
                            token=signal.token_address, error=str(e))
        log.info("executor.position_tracking_registered", token=signal.token_address)

    def _unregister_position_tracking(self, chain: str, token: str) -> None:
        self.smart_exits.drop_position(chain, token)
        try:
            self.dev_tracker.untrack(chain, token)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Smart exit signal processing (called by orchestrator)
    # ------------------------------------------------------------------
    async def process_exit_signals(self, exit_signals: list[ExitSignal]) -> None:
        for sig in exit_signals:
            try:
                # Convert ExitSignal to a Signal and route through _handle_sell
                synthetic = Signal(
                    strategy="smart_exit",
                    chain=sig.chain,
                    token_address=sig.token,
                    signal_type=SignalType.SELL,
                    suggested_size_pct=sig.size_pct_to_sell,
                    confidence=1.0,
                    reason=sig.reason,
                )
                await self._handle_sell(synthetic, sig.reason, size_pct=sig.size_pct_to_sell)
            except Exception as e:
                log.error("executor.exit_signal_failed", error=str(e))
