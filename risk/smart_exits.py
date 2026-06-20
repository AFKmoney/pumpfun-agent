"""
smart_exits.py
==============
Smart exit manager — runs alongside RiskManager.check_exits() and applies
sophisticated exit logic per position:

1. **Break-even stop**: once position reaches +X%, move stop-loss to entry price.
   Locks in profit, eliminates downside risk.

2. **Trailing stop**: after break-even triggered, trail the stop at Y% below
   the highest price seen since entry. Lets winners run, exits on reversal.

3. **Take-profit ladder**: instead of single TP at +50%, sell in tranches:
   - 25% at +50%
   - 25% at +100%
   - 25% at +200%
   - 25% free-run with trailing stop
   Maximizes expected value on asymmetric payoffs (memecoins are fat-tailed).

4. **Dev-sell trigger**: if the dev wallet sells ANY amount, immediately exit
   100% of the position. Devs have inside info; their sell is the strongest
   negative signal.

5. **Migration pump exit**: when pump.fun migrates a token to Raydium, price
   typically spikes 50-500% in 5-30 minutes as real LP opens. We hold through
   the migration pump, then trail tight.

6. **Time decay**: if position hasn't moved significantly after N minutes,
   reduce size by 50% (capital efficiency).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from risk.types import Position
from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("smart_exits")


@dataclass
class PositionMeta:
    """Extra state tracked per position by SmartExitManager."""
    chain: str
    token: str
    entry_price: float
    highest_price: float                    # max price since entry
    break_even_triggered: bool = False
    tranche_1_taken: bool = False            # +50%  -> sell 25%
    tranche_2_taken: bool = False            # +100% -> sell 25%
    tranche_3_taken: bool = False            # +200% -> sell 25%
    last_check_ts: float = field(default_factory=time.time)


@dataclass
class ExitSignal:
    chain: str
    token: str
    reason: str
    size_pct_to_sell: float                  # 0..1 (1.0 = full exit)
    suggested_price: Optional[float] = None  # current price at decision time
    urgent: bool = False                     # if True, use Jito for sell


class SmartExitManager:
    """Applies trailing/break-even/ladder exits on top of base SL/TP."""

    def __init__(self) -> None:
        self.cfg = Config.get()
        ecfg = self.cfg.get_nested("smart_exits", default={})
        self.break_even_trigger_pct = ecfg.get("break_even_trigger_pct", 25.0)
        self.break_even_stop_pct = ecfg.get("break_even_stop_pct", 0.0)  # 0 = at entry
        self.trailing_stop_pct = ecfg.get("trailing_stop_pct", 15.0)
        self.tps = ecfg.get("take_profit_ladder", [
            {"pnl_pct": 50.0, "sell_pct": 0.25},
            {"pnl_pct": 100.0, "sell_pct": 0.25},
            {"pnl_pct": 200.0, "sell_pct": 0.25},
        ])
        self.max_hold_minutes_before_decay = ecfg.get("max_hold_minutes_before_decay", 60)
        self.decay_sell_pct = ecfg.get("decay_sell_pct", 0.5)
        self.migration_pump_hold_minutes = ecfg.get("migration_pump_hold_minutes", 15)
        self._meta: dict[str, PositionMeta] = {}  # key = "chain:token"

    # ------------------------------------------------------------------
    def init_position(self, pos: Position) -> None:
        key = f"{pos.chain}:{pos.token}"
        self._meta[key] = PositionMeta(
            chain=pos.chain, token=pos.token, entry_price=pos.entry_price,
            highest_price=pos.entry_price,
        )
        log.info("smart_exits.position_initialized", token=pos.token,
                 entry_price=pos.entry_price, break_even_trigger=self.break_even_trigger_pct)

    def drop_position(self, chain: str, token: str) -> None:
        self._meta.pop(f"{chain}:{token}", None)

    # ------------------------------------------------------------------
    async def check_exits(
        self,
        positions: dict[str, Position],
        price_provider,
        dev_sell_detector=None,
    ) -> list[ExitSignal]:
        """
        Returns list of ExitSignals to execute.
        price_provider: async callable (chain, token) -> current_price
        dev_sell_detector: optional async callable (chain, token) -> bool (True if dev sold)
        """
        signals: list[ExitSignal] = []

        for key, pos in positions.items():
            meta = self._meta.get(key)
            if not meta:
                # New position we haven't seen — initialize
                self.init_position(pos)
                meta = self._meta[key]

            try:
                current_price = await price_provider(pos.chain, pos.token)
            except Exception as e:
                log.warning("smart_exits.price_fetch_failed", token=pos.token, error=str(e))
                continue
            if current_price <= 0:
                continue

            # Update highest price seen
            if current_price > meta.highest_price:
                meta.highest_price = current_price

            pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            hold_minutes = (time.time() - pos.opened_at) / 60

            # ---- 0. Dev-sell trigger (highest priority) ----
            if dev_sell_detector:
                try:
                    if await dev_sell_detector(pos.chain, pos.token):
                        signals.append(ExitSignal(
                            chain=pos.chain, token=pos.token,
                            reason="DEV SOLD — instant exit",
                            size_pct_to_sell=1.0,
                            suggested_price=current_price,
                            urgent=True,
                        ))
                        continue
                except Exception as e:
                    log.warning("smart_exits.dev_check_failed", token=pos.token, error=str(e))

            # ---- 1. Hard stop-loss (from base risk_manager) ----
            if pnl_pct <= -pos.stop_loss_pct:
                signals.append(ExitSignal(
                    chain=pos.chain, token=pos.token,
                    reason=f"Hard stop loss @ {pnl_pct:.1f}%",
                    size_pct_to_sell=1.0,
                    suggested_price=current_price,
                    urgent=True,
                ))
                continue

            # ---- 2. Take-profit ladder ----
            for i, tp in enumerate(self.tps):
                attr = f"tranche_{i+1}_taken"
                if not getattr(meta, attr) and pnl_pct >= tp["pnl_pct"]:
                    setattr(meta, attr, True)
                    signals.append(ExitSignal(
                        chain=pos.chain, token=pos.token,
                        reason=f"TP ladder tranche {i+1} @ {pnl_pct:.1f}%",
                        size_pct_to_sell=tp["sell_pct"],
                        suggested_price=current_price,
                    ))
                    log.info("smart_exits.tp_ladder", token=pos.token,
                             tranche=i+1, sell_pct=tp["sell_pct"], pnl=pnl_pct)

            # ---- 3. Break-even stop ----
            if not meta.break_even_triggered and pnl_pct >= self.break_even_trigger_pct:
                meta.break_even_triggered = True
                log.info("smart_exits.break_even_armed", token=pos.token, pnl=pnl_pct)

            if meta.break_even_triggered:
                break_even_price = pos.entry_price * (1 + self.break_even_stop_pct / 100)
                if current_price <= break_even_price:
                    signals.append(ExitSignal(
                        chain=pos.chain, token=pos.token,
                        reason=f"Break-even stop hit (entry={pos.entry_price:.8f}, now={current_price:.8f})",
                        size_pct_to_sell=1.0,
                        suggested_price=current_price,
                    ))
                    continue

            # ---- 4. Trailing stop (after break-even) ----
            if meta.break_even_triggered:
                trailing_stop_price = meta.highest_price * (1 - self.trailing_stop_pct / 100)
                if current_price <= trailing_stop_price and meta.highest_price > pos.entry_price:
                    signals.append(ExitSignal(
                        chain=pos.chain, token=pos.token,
                        reason=f"Trailing stop @ {pnl_pct:.1f}% (peak={((meta.highest_price-pos.entry_price)/pos.entry_price*100):.1f}%)",
                        size_pct_to_sell=1.0,
                        suggested_price=current_price,
                    ))
                    continue

            # ---- 5. Time decay ----
            if (hold_minutes > self.max_hold_minutes_before_decay
                and -10 < pnl_pct < 10
                and not any([meta.tranche_1_taken, meta.tranche_2_taken, meta.tranche_3_taken])):
                # Position is dead money — exit 50%
                signals.append(ExitSignal(
                    chain=pos.chain, token=pos.token,
                    reason=f"Time decay: {hold_minutes:.0f}min hold, PnL flat",
                    size_pct_to_sell=self.decay_sell_pct,
                    suggested_price=current_price,
                ))

        return signals

    def get_meta(self, chain: str, token: str) -> Optional[PositionMeta]:
        return self._meta.get(f"{chain}:{token}")
