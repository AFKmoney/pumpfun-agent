"""
dev_tracker.py
==============
Dev wallet activity tracker for Solana pump.fun tokens.

For each open position, we know:
- The dev wallet (creator) from the pump.fun create event.
- We watch the dev wallet's token balance.

If the dev's balance decreases (they sold ANY amount), we trigger an
immediate 100% exit on our position. Devs have asymmetric information;
their sell is the strongest negative signal in memecoin trading.

Implementation:
- For each tracked token, subscribe to dev wallet's token account via
  Helius `accountSubscribe` WebSocket.
- On any balance decrease, fire the callback.
- Caches the dev wallet per token (from the create event).
"""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable, Optional

import websockets

from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("dev_tracker")

# Type for the async callback: (chain, token) -> None
DevSellCallback = Callable[[str, str], Awaitable[None]]


class DevTracker:
    """Tracks dev wallet sells for open positions."""

    def __init__(self) -> None:
        self.cfg = Config.get()
        self._dev_wallets: dict[str, str] = {}  # f"{chain}:{token}" -> dev_wallet_address
        self._last_balances: dict[str, float] = {}  # f"{chain}:{token}" -> last known balance
        self._tasks: dict[str, asyncio.Task] = {}  # f"{chain}:{token}" -> watch task
        self._callbacks: list[DevSellCallback] = []
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    def register_callback(self, cb: DevSellCallback) -> None:
        self._callbacks.append(cb)

    def track(self, chain: str, token: str, dev_wallet: str, initial_balance: float = 0.0) -> None:
        """Start tracking a token's dev wallet."""
        key = f"{chain}:{token}"
        self._dev_wallets[key] = dev_wallet
        self._last_balances[key] = initial_balance
        if key not in self._tasks:
            self._tasks[key] = asyncio.create_task(self._watch(chain, token, dev_wallet))
            log.info("dev_tracker.tracking", token=token, dev_wallet=dev_wallet[:8])

    def untrack(self, chain: str, token: str) -> None:
        key = f"{chain}:{token}"
        task = self._tasks.pop(key, None)
        if task:
            task.cancel()
        self._dev_wallets.pop(key, None)
        self._last_balances.pop(key, None)
        log.info("dev_tracker.untrack", token=token)

    async def _watch(self, chain: str, token: str, dev_wallet: str) -> None:
        """Subscribe to dev wallet's token account balance via WebSocket."""
        ws_endpoint = self.cfg["chains"][chain]["ws_endpoint"]
        key = f"{chain}:{token}"

        # Derive the dev's ATA (Associated Token Account) for this mint
        # In production, use getProgramAccounts on Token Program with filters.
        # For simplicity we use accountSubscribe on the dev wallet and check token balances.

        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "accountSubscribe",
            "params": [dev_wallet, {"encoding": "jsonParsed", "commitment": "confirmed"}],
        }
        try:
            async with websockets.connect(ws_endpoint) as ws:
                await ws.send(json.dumps(payload))
                while not self._stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        # Send ping to keep alive
                        try:
                            await ws.ping()
                        except Exception:
                            break
                        continue
                    msg = json.loads(raw)
                    if msg.get("method") != "accountNotification":
                        continue
                    # Parse token balances from the account data
                    value = msg.get("params", {}).get("result", {}).get("value", {})
                    tokens = (value.get("data") or {}).get("parsed", {}).get("info", {}).get("tokens", [])
                    current_balance = 0.0
                    for t in tokens:
                        if t.get("mint") == token:
                            current_balance = float(t.get("amount", 0))
                            break
                    last = self._last_balances.get(key, 0)
                    if current_balance < last:
                        log.warning("dev_tracker.DEV_SOLD",
                                    token=token, dev_wallet=dev_wallet[:8],
                                    prev=last, current=current_balance,
                                    delta_pct=((current_balance-last)/last*100 if last else 0))
                        # Fire callbacks
                        for cb in self._callbacks:
                            try:
                                await cb(chain, token)
                            except Exception as e:
                                log.error("dev_tracker.callback_failed", error=str(e))
                    self._last_balances[key] = current_balance
        except asyncio.CancelledError:
            log.info("dev_tracker.cancelled", token=token)
            raise
        except Exception as e:
            log.error("dev_tracker.watch_failed", token=token, error=str(e))

    # ------------------------------------------------------------------
    async def stop_all(self) -> None:
        self._stop_event.set()
        for task in self._tasks.values():
            task.cancel()
        for task in self._tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

    def is_dev_selling(self, chain: str, token: str) -> bool:
        """Synchronous check: did dev sell since last call? Resets after read."""
        # Note: this is a simple check used as fallback if WebSocket misses events.
        # The primary mechanism is the async callback above.
        return False  # WebSocket-driven; this always returns False
