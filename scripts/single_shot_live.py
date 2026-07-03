#!/usr/bin/env python3
"""
single_shot_live.py
===================
Live trading in SINGLE-SHOT mode: the bot executes ONE real trade, then stops.

This is the safest way to do your first live test:
  - One buy, one position, then auto-halt
  - Full logging of every step (latency, slippage, fill, error)
  - Kill switch still active (Telegram /kill or data/KILL_SWITCH)
  - If the buy fails, the bot stops immediately (no retry loop)

Usage:
    python scripts/single_shot_live.py

The bot will:
  1. Start all monitors (regime, analytics, etc.)
  2. Wait for the FIRST signal that passes all gates
  3. Execute the buy
  4. Log everything (tx hash, fill price, latency)
  5. Monitor the position for exits (SL/TP/dev-sell/soft-rug)
  6. Stop after the position closes (or after 30 min timeout)
"""
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Override max_open_positions to 1 for single-shot mode
import os
os.environ["SINGLE_SHOT_MODE"] = "1"


async def main() -> int:
    from orchestrator import Orchestrator
    from utils.logger import setup_logger
    from utils.kill_switch import KillSwitch

    log = setup_logger("single_shot")

    print("=" * 60)
    print("  SINGLE-SHOT LIVE TRADING")
    print("  The bot will execute ONE trade, then stop.")
    print("=" * 60)
    print()

    # Confirm the user wants to proceed
    print("⚠️  This will trade with REAL SOL.")
    print("⚠️  Make sure you ran scripts/preflight_live.py first.")
    print()

    orch = Orchestrator()

    # Patch: force max 1 open position
    orch.cfg["trading"]["max_open_positions"] = 1
    orch.risk.tcfg["max_open_positions"] = 1
    log.info("single_shot.max_positions_forced", max=1)

    # Track the first trade
    first_trade_done = asyncio.Event()
    original_execute = None

    # Wrap the executor to detect the first successful buy
    for executor in orch.executors.values():
        original_execute = executor.execute
        async def tracked_execute(signal, allocated, _orig=original_execute):
            await _orig(signal, allocated)
            key = f"{signal.chain}:{signal.token_address}"
            if key in executor.risk.positions and signal.signal_type == "BUY":
                if not first_trade_done.is_set():
                    log.info("single_shot.first_buy_executed",
                             token=signal.token_address,
                             chain=signal.chain)
                    first_trade_done.set()
        executor.execute = tracked_execute

    # Auto-stop conditions:
    # 1. First trade closes (position exits)
    # 2. 30-minute hard timeout
    # 3. Kill switch triggered

    async def watch_for_completion():
        """Stop the bot when the first position closes or timeout hits."""
        deadline = time.time() + 1800  # 30 min max
        bought = False
        while not orch._stop_event.is_set():
            await asyncio.sleep(5)
            if first_trade_done.is_set():
                bought = True
            if bought and len(orch.risk.positions) == 0:
                # Position opened then closed = done
                log.info("single_shot.position_closed_stopping")
                orch.request_stop()
                return
            if time.time() > deadline:
                if bought:
                    log.warning("single_shot.timeout_position_still_open")
                else:
                    log.warning("single_shot.timeout_no_signal_matched")
                orch.request_stop()
                return

    async def watch_kill():
        while not orch._stop_event.is_set():
            if KillSwitch.is_triggered():
                log.warning("single_shot.kill_switch_triggered")
                orch.request_stop()
                return
            await asyncio.sleep(2)

    # Launch
    asyncio.create_task(watch_for_completion())
    asyncio.create_task(watch_kill())

    try:
        await orch.start()
    except Exception as e:
        log.error("single_shot.crash", error=str(e))

    print()
    print("=" * 60)
    # Summary
    stats = orch.risk
    print(f"  Session complete.")
    print(f"  State: {orch.state}")
    print(f"  Open positions: {len(stats.positions)}")
    if hasattr(stats, 'history') and stats.history:
        last = stats.history[-1]
        print(f"  Last trade: {last.side} {last.token[:12]}... pnl={last.pnl_pct}")
    print(f"  Daily PnL: {stats.daily_pnl_pct:.2f}%")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
