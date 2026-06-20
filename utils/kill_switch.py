"""
kill_switch.py
==============
Hard kill switch. Multiple triggers:
1. Telegram "/kill" command
2. Presence of a sentinel file on disk  (./data/KILL_SWITCH)
3. Daily loss cap reached
4. Manual API call (if monitoring server enabled)

When ANY trigger fires, the agent stops opening new positions, closes any
open positions if requested, and exits the main loop.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("kill_switch")


class KillSwitch:
    _triggered: bool = False
    _reason: Optional[str] = None

    def __init__(self) -> None:
        self.cfg = Config.get()
        self.path = Path(self.cfg.get_nested("safety", "kill_switch_file", default="./data/KILL_SWITCH"))

    @classmethod
    def trigger(cls, reason: str) -> None:
        cls._triggered = True
        cls._reason = reason
        log.critical("kill_switch.triggered", reason=reason)

    @classmethod
    def is_triggered(cls) -> bool:
        return cls._triggered

    @classmethod
    def reset(cls) -> None:
        cls._triggered = False
        cls._reason = None

    async def watch_file(self, interval_sec: float = 2.0) -> None:
        """Watch for the sentinel file. Once it appears, trigger the kill switch."""
        while not KillSwitch.is_triggered():
            if self.path.exists():
                KillSwitch.trigger(f"Sentinel file detected: {self.path}")
                # Auto-remove so manual restart is possible without re-trigger
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                break
            await asyncio.sleep(interval_sec)

    @classmethod
    def reason(cls) -> Optional[str]:
        return cls._reason
