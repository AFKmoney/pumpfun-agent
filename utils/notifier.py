"""
notifier.py
===========
Telegram notifications + kill switch command listener.
Posts trade events, errors, and rugpull alerts to a Telegram chat.

The Telegram bot ALSO listens for the `/kill` command from the configured
chat_id, which triggers the KillSwitch instantly.
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from strategies.base_strategy import Signal, SignalType
from utils.config_loader import Config
from utils.kill_switch import KillSwitch
from utils.logger import setup_logger

log = setup_logger("notifier")


class TelegramNotifier:
    def __init__(self) -> None:
        self.cfg = Config.get()
        ncfg = self.cfg.get_nested("notifications", "telegram", default={})
        self.enabled = ncfg.get("enabled", False)
        self.token = os.environ.get(ncfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN"), "") if self.enabled else ""
        self.chat_id = os.environ.get(ncfg.get("chat_id_env", "TELEGRAM_CHAT_ID"), "") if self.enabled else ""
        self.notify_on = ncfg.get("notify_on", {})
        self.kill_cmd = ncfg.get("kill_switch_command", "/kill")
        self._app: Optional[Application] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        if self.enabled and (not self.token or not self.chat_id):
            log.warning("notifier.disabled_missing_creds")
            self.enabled = False

    async def start(self) -> None:
        if not self.enabled:
            log.info("notifier.disabled")
            return
        self._app = Application.builder().token(self.token).build()
        self._app.add_handler(CommandHandler("kill", self._handle_kill))
        self._app.add_handler(CommandHandler("status", self._handle_status))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        asyncio.create_task(self._sender_loop())
        log.info("notifier.started", chat_id=self.chat_id)

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def notify_trade_opened(self, signal: Signal, size: float, price: float, tx_hash: str) -> None:
        if not self._should("trade_opened"):
            return
        msg = (
            f"*{signal.signal_type.value} {signal.chain.upper()}*\n"
            f"Strategy: `{signal.strategy}`\n"
            f"Token: `{signal.token_address}`\n"
            f"Size: `{size}`\n"
            f"Price: `{price}`\n"
            f"Reason: {signal.reason}\n"
            f"TX: `{tx_hash}`"
        )
        await self._queue.put(msg)

    async def notify_trade_closed(self, chain: str, token: str, pnl_pct: float, reason: str) -> None:
        event = "take_profit_hit" if pnl_pct > 0 else "stop_loss_hit"
        if not self._should(event):
            return
        emoji = "🟢" if pnl_pct > 0 else "🔴"
        msg = f"{emoji} *CLOSED {chain.upper()}*\nToken: `{token}`\nPnL: `{pnl_pct:.2f}%`\nReason: {reason}"
        await self._queue.put(msg)

    async def notify_rugpull(self, token: str, reasons: list[str]) -> None:
        if not self._should("rugpull_detected"):
            return
        msg = f"⚠️ *RUGPULL DETECTED*\nToken: `{token}`\nReasons: {', '.join(reasons)}"
        await self._queue.put(msg)

    async def notify_daily_loss_cap(self) -> None:
        if not self._should("daily_loss_cap_reached"):
            return
        await self._queue.put("🛑 *DAILY LOSS CAP REACHED*\nAgent halted. Review positions manually.")

    async def notify_error(self, error: str) -> None:
        if not self._should("system_error"):
            return
        await self._queue.put(f"❌ *ERROR*\n```\n{error[:500]}\n```")

    # ------------------------------------------------------------------
    # Internal sender loop
    # ------------------------------------------------------------------
    async def _sender_loop(self) -> None:
        while True:
            msg = await self._queue.get()
            try:
                await self._app.bot.send_message(
                    chat_id=self.chat_id, text=msg,
                    parse_mode="Markdown", disable_web_page_preview=True,
                )
            except Exception as e:
                log.warning("notifier.send_failed", error=str(e))
            await asyncio.sleep(0.5)  # avoid Telegram rate limit

    def _should(self, event: str) -> bool:
        return self.enabled and self.notify_on.get(event, True)

    # ------------------------------------------------------------------
    # Telegram command handlers
    # ------------------------------------------------------------------
    async def _handle_kill(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if str(update.effective_chat.id) != self.chat_id:
            return  # ignore unauthorized
        KillSwitch.trigger("Manual /kill from Telegram")
        await update.message.reply_text("🛑 Kill switch triggered. Agent will halt.")

    async def _handle_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if str(update.effective_chat.id) != self.chat_id:
            return
        from orchestrator import Orchestrator  # lazy import
        status = Orchestrator.get_status()
        await update.message.reply_text(
            f"Status: {status['state']}\n"
            f"Open positions: {status['open_positions']}\n"
            f"Daily PnL: {status['daily_pnl_pct']:.2f}%\n"
            f"Kill switch: {'ACTIVE' if KillSwitch.is_triggered() else 'inactive'}"
        )
