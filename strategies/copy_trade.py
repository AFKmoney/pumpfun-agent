"""
copy_trade.py
=============
Copy-trading + anti-copy (fade) strategy.

Two modes, both driven by the same wallet-tailing infrastructure:

1. COPY mode (target_wallets):
   Replicate buys from wallets in the config watchlist. Decode the pump.fun
   buy/sell instruction from each new signature and emit a matching BUY signal.

2. ANTI-COPY mode (fade_wallets):
   Fade wallets that lose systematically — when they BUY, we AVOID that token,
   and optionally we can short/sell-if-held. Loser wallets are more stable
   than winner wallets (winners rotate strategies; losers repeat mistakes),
   which makes fading them a surprisingly durable edge.

Both modes decode the real pump.fun instruction data via the buy/sell
discriminators (utils.pumpfun_ix), not a placeholder. When the instruction
cannot be decoded from the tx (non-pump.fun swap, or unknown program), we
skip silently.

Wallet reputation from SocialGraphAnalyzer gates both modes:
- COPY: only follow a wallet the social_graph considers "smart_money" OR that
  is in the explicit target_wallets list (trust the operator).
- ANTI-COPY: only fade wallets flagged with a negative reputation_score, OR in
  the explicit fade_wallets list.
"""
from __future__ import annotations

import asyncio
import base64
import re
from typing import Optional

from chains.base_chain import BaseChainAdapter
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from utils.config_loader import Config
from utils.logger import setup_logger
from utils.pumpfun_parser import PUMP_FUN_PROGRAM, parse_create_instruction
from utils.pumpfun_ix import BUY_DISCRIMINATOR, SELL_DISCRIMINATOR

log = setup_logger("copy_trade")

# Base58 regex for mint extraction from instruction account lists / logs.
_B58 = re.compile(r'\b([1-9A-HJ-NP-Za-km-z]{32,44})\b')


def _decode_pumpfun_swap(tx: dict) -> Optional[dict]:
    """
    Walk a parsed Solana transaction and find a pump.fun buy or sell instruction.
    Returns {"side": "buy"|"sell", "mint": <addr>, "amount_raw": int} or None.

    Handles both top-level and inner instructions. Uses the accountKeys /
    accountKeys index to resolve the mint account.
    """
    if not tx:
        return None
    message = tx.get("transaction", {}).get("message", {}) or {}
    account_keys = message.get("accountKeys", []) or []
    instructions = list(message.get("instructions", []) or [])
    meta = tx.get("meta", {}) or {}
    for inner_group in (meta.get("innerInstructions") or []):
        instructions.extend(inner_group.get("instructions", []) or [])

    for ix in instructions:
        # Resolve program id
        prog = ix.get("programId")
        if prog is None:
            pidx = ix.get("programIdIndex")
            if isinstance(pidx, int) and pidx < len(account_keys):
                prog = account_keys[pidx]
        if prog != PUMP_FUN_PROGRAM:
            continue
        data = ix.get("data", "")
        try:
            raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
        except Exception:
            continue
        if len(raw) < 16:
            continue
        disc = raw[:8]
        if disc == BUY_DISCRIMINATOR:
            side = "buy"
            # buy data: disc(8) + amount(u64) + max_sol_cost(u64)
            if len(raw) < 24:
                continue
            amount_raw = int.from_bytes(raw[8:16], "little")
        elif disc == SELL_DISCRIMINATOR:
            side = "sell"
            # sell data: disc(8) + amount(u64)
            amount_raw = int.from_bytes(raw[8:16], "little")
        else:
            continue
        # The mint is the 3rd account in both buy/sell (index 2): global,
        # fee_recipient, mint, bonding_curve, ...
        accounts_idx = ix.get("accounts", []) or []
        mint = None
        if len(accounts_idx) >= 3:
            midx = accounts_idx[2]
            if isinstance(midx, int) and midx < len(account_keys):
                mint = account_keys[midx]
            elif isinstance(midx, str):
                mint = midx
        if not mint:
            continue
        return {"side": side, "mint": mint, "amount_raw": amount_raw}
    return None


class CopyTradeStrategy(BaseStrategy):
    name = "copy_trade"

    def __init__(
        self,
        adapter: BaseChainAdapter,
        signal_queue: asyncio.Queue,
        social_graph=None,
    ) -> None:
        super().__init__(adapter)
        self.queue = signal_queue
        self.cfg = Config.get().get_nested("strategies", "copy_trade", default={})
        self.social_graph = social_graph
        self.target_wallets: list[str] = self.cfg.get("target_wallets", [])
        self.fade_wallets: list[str] = self.cfg.get("fade_wallets", [])
        self.follow_delay_ms_max = self.cfg.get("follow_delay_ms_max", 3000)
        self.max_follow_size = self.cfg.get("max_follow_size_sol", 0.2)
        self.anti_copy_enabled: bool = self.cfg.get("anti_copy_enabled", True)
        # In-memory set of tokens faded by loser wallets (avoid them on other
        # strategies too). Exposed for other components to consult.
        self.faded_tokens: set[str] = set()

    def is_faded(self, token: str) -> bool:
        """Public check: is this token on the anti-copy fade list?"""
        return token in self.faded_tokens

    async def run(self) -> None:
        wallets = list(self.target_wallets) + (list(self.fade_wallets) if self.anti_copy_enabled else [])
        log.info("copy_trade.started", chain=self.adapter.chain_name,
                 copy_targets=self.target_wallets, fade_targets=self.fade_wallets)
        if not wallets:
            log.warning("copy_trade.no_wallets_configured")
            return
        tasks = [asyncio.create_task(self._tail_wallet(w)) for w in wallets]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            log.info("copy_trade.cancelled")
            raise

    async def _tail_wallet(self, wallet: str) -> None:
        async for event in self.adapter.watch_wallet_activity(wallet):
            try:
                sig = await self.evaluate(event)
                if sig and sig.signal_type != SignalType.HOLD:
                    await self.queue.put(sig)
            except Exception as e:
                log.error("copy_trade.evaluate_failed", wallet=wallet, error=str(e))

    async def evaluate(self, event: dict) -> Signal | None:
        wallet = event.get("wallet", "")
        tx = event.get("tx")
        chain = event.get("chain", self.adapter.chain_name)
        decoded = _decode_pumpfun_swap(tx)
        if not decoded:
            return None

        is_copy_target = wallet in self.target_wallets
        is_fade_target = self.anti_copy_enabled and wallet in self.fade_wallets

        # Reputation gate (optional): if we have a social_graph, trust it.
        if self.social_graph is not None and not is_copy_target and not is_fade_target:
            prof = self.social_graph.get_wallet_profile(wallet)
            if prof:
                if "smart_money" in prof.tags:
                    is_copy_target = True
                elif prof.reputation_score < 0 and self.anti_copy_enabled:
                    is_fade_target = True

        if decoded["side"] == "buy":
            if is_copy_target:
                return Signal(
                    strategy=self.name, chain=chain, token_address=decoded["mint"],
                    signal_type=SignalType.BUY, suggested_size_pct=0.01,
                    confidence=0.6,
                    reason=f"COPY buy: wallet {wallet[:8]} bought",
                    stop_loss_pct=20.0, take_profit_pct=50.0,
                    metadata={"copied_wallet": wallet},
                )
            if is_fade_target:
                # Don't buy this token on any strategy; record it for the
                # executor / other strategies to consult.
                self.faded_tokens.add(decoded["mint"])
                log.info("copy_trade.fade_recorded",
                         token=decoded["mint"], wallet=wallet[:8])
                return None
        elif decoded["side"] == "sell":
            if is_copy_target:
                # Replicate the exit too (mirrors the winner's trade management).
                return Signal(
                    strategy=self.name, chain=chain, token_address=decoded["mint"],
                    signal_type=SignalType.SELL, suggested_size_pct=1.0,
                    confidence=0.7,
                    reason=f"COPY sell: wallet {wallet[:8]} exited",
                    metadata={"copied_wallet": wallet},
                )
        return None
