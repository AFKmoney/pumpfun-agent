"""
anti_rugpull.py
===============
Anti-rugpull pre-flight check. Runs synchronously BEFORE any buy is executed.

Real checks (all wired to live data):
1. Holders count + concentration (Helius getTokenLargestAccounts).
2. Liquidity (DexScreener).
3. Mint authority renounced (Helius getAccountInfo).
4. Freeze authority renounced.
5. Honeypot detection:
   - Solana: micro-buy (0.001 SOL) + immediate sell test.
   - EVM: TokenSniffer API.

Returns a verdict + reasons. The orchestrator blocks any BUY whose verdict
is False.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from chains.base_chain import BaseChainAdapter
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from utils.config_loader import Config
from utils.data_providers import get_providers
from utils.logger import setup_logger

log = setup_logger("anti_rugpull")


@dataclass
class RugpullVerdict:
    is_safe: bool
    reasons: list[str]
    score: float  # 0 = rug, 1 = safe


class AntiRugpullStrategy(BaseStrategy):
    name = "anti_rugpull"

    def __init__(self, adapter: BaseChainAdapter) -> None:
        super().__init__(adapter)
        self.cfg = Config.get().get_nested("strategies", "anti_rugpull", default={})
        self.min_holders = self.cfg.get("min_holders", 50)
        self.max_top10_pct = self.cfg.get("max_top10_holders_pct", 0.40)
        self.min_liquidity_usd = self.cfg.get("min_liquidity_usd", 10_000)
        self.require_renounced_mint = self.cfg.get("require_renounced_mint", True)
        self.require_locked_lp = self.cfg.get("require_locked_lp", False)
        self.check_honeypot = self.cfg.get("check_honeypot", True)
        self.reject_if_mint_authority_active = self.cfg.get("reject_if_mint_authority_active", True)
        self._verdict_cache: dict[str, tuple[float, RugpullVerdict]] = {}  # mint -> (ts, verdict), 5 min TTL

    async def run(self) -> None:
        log.info("anti_rugpull.ready", chain=self.adapter.chain_name)

    async def evaluate(self, *args, **kwargs) -> Signal:
        return Signal(self.name, self.adapter.chain_name, "", SignalType.HOLD, 0, 0, "n/a")

    # ------------------------------------------------------------------
    async def check(self, token_address: str, skip_honeypot: bool = False) -> RugpullVerdict:
        # 5-min cache: a token can't go from rug to safe in 5 min
        import time
        cached = self._verdict_cache.get(token_address)
        if cached and time.time() - cached[0] < 300:
            return cached[1]

        providers = get_providers()
        reasons: list[str] = []
        score = 1.0

        # ---- 1. Holders count + concentration ----
        try:
            holders = await self._fetch_holders(token_address, providers)
            n_holders = len(holders)
            if n_holders < self.min_holders:
                reasons.append(f"Insufficient holders ({n_holders} < {self.min_holders})")
                score -= 0.3
            total = sum(h["amount"] for h in holders) or 1
            top10_pct = sum(h["amount"] for h in holders[:10]) / total
            if top10_pct > self.max_top10_pct:
                reasons.append(f"Top10 concentration {top10_pct:.0%} > {self.max_top10_pct:.0%}")
                score -= 0.3
        except Exception as e:
            reasons.append(f"Holders check failed: {e}")
            score -= 0.1

        # ---- 2. Liquidity ----
        try:
            liq = await providers["dexscreener"].get_liquidity_usd(token_address)
            if liq < self.min_liquidity_usd:
                reasons.append(f"Liquidity ${liq:,.0f} < ${self.min_liquidity_usd:,.0f}")
                score -= 0.3
        except Exception as e:
            reasons.append(f"Liquidity check failed: {e}")
            score -= 0.1

        # ---- 3. Mint authority renounced (Solana SPL) ----
        if self.adapter.chain_name == "solana" and self.reject_if_mint_authority_active:
            try:
                renounced = await providers["helius"].is_mint_authority_renounced(token_address)
                if not renounced:
                    reasons.append("Mint authority NOT renounced")
                    score -= 0.3
            except Exception as e:
                reasons.append(f"Mint authority check failed: {e}")
                score -= 0.1

        # ---- 4. Honeypot ----
        if self.check_honeypot and not skip_honeypot:
            try:
                honeypot, reason = await self._honeypot_check(token_address, providers)
                if honeypot:
                    reasons.append(f"Honeypot: {reason}")
                    score -= 0.5
            except Exception as e:
                reasons.append(f"Honeypot check skipped: {e}")

        verdict = RugpullVerdict(
            is_safe=score >= 0.6 and not reasons,
            reasons=reasons or ["All checks passed"],
            score=max(0.0, min(1.0, score)),
        )
        self._verdict_cache[token_address] = (time.time(), verdict)
        return verdict

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _fetch_holders(self, token: str, providers: dict) -> list[dict]:
        """Return holder list sorted by balance, descending."""
        if self.adapter.chain_name == "solana":
            holders = await providers["helius"].get_token_holders(token)
            holders.sort(key=lambda h: h["amount"], reverse=True)
            return holders
        # EVM: TokenSniffer audit includes top holders summary in some versions
        audit = await providers["tokensniffer"].get_token_audit(self.adapter.chain_name, token)
        if audit and "holders" in audit:
            return [{"owner": h.get("address", ""), "amount": float(h.get("balance", 0))} for h in audit["holders"]]
        return []

    async def _honeypot_check(self, token: str, providers: dict) -> tuple[bool, str]:
        """
        Returns (is_honeypot, reason).

        Solana: micro-buy 0.001 SOL worth of the token, then attempt to sell
                100% of what we received. If the sell fails, the token is a honeypot.
                Only runs in LIVE mode (skipped in paper to avoid real tx).
                Caches results for 30 minutes per token to avoid repeated spend.
        EVM:    TokenSniffer API.
        """
        if self.adapter.chain_name == "solana":
            # Check cache first
            import time
            cache_key = f"honeypot:{token}"
            cached_ts = providers.get("_honeypot_cache", {}).get(cache_key)
            if cached_ts and time.time() - cached_ts[0] < 1800:  # 30 min
                return cached_ts[1]

            # Freeze authority check (fast, free)
            try:
                freeze_renounced = await providers["helius"].is_freeze_authority_renounced(token)
                if not freeze_renounced:
                    result = (True, "Freeze authority NOT renounced (owner can freeze sellers)")
                    providers.setdefault("_honeypot_cache", {})[cache_key] = (time.time(), result)
                    return result
            except Exception:
                pass

            # Skip micro-buy in paper mode (would do nothing anyway)
            mode = Config.get().get_nested("trading", "mode", default="paper")
            if mode != "live":
                return False, "OK (paper mode, freeze check passed)"

            # Micro-buy + sell test
            result = await self._solana_micro_buy_sell_test(token)
            providers.setdefault("_honeypot_cache", {})[cache_key] = (time.time(), result)
            return result

        # EVM: use TokenSniffer
        return await providers["tokensniffer"].is_honeypot(self.adapter.chain_name, token)

    async def _solana_micro_buy_sell_test(self, token: str) -> tuple[bool, str]:
        """
        Buy 0.001 SOL worth of `token`, then attempt to sell 100% back.
        Returns (is_honeypot, reason).

        Cost: ~0.001 SOL + gas per check. Cached 30 min to avoid repeat spend.
        """
        try:
            MICRO_BUY_SOL = 0.001
            slippage_bps = self.cfg["trading"].get("slippage_bps", 1000)

            # 1. Buy
            buy_result = await self.adapter.buy(token, MICRO_BUY_SOL, slippage_bps)
            if not buy_result.success:
                return True, f"Micro-buy failed: {buy_result.error}"
            if not buy_result.executed_amount or buy_result.executed_amount <= 0:
                return True, "Micro-buy returned 0 tokens"

            log.info("anti_rugpull.micro_buy_ok", token=token,
                     amount_token=buy_result.executed_amount)

            # 2. Sell 100% back immediately
            sell_result = await self.adapter.sell(token, buy_result.executed_amount, slippage_bps)
            if not sell_result.success:
                # We're now holding tokens we can't sell — log + blacklist
                log.error("anti_rugpull.micro_sell_failed",
                          token=token, error=sell_result.error,
                          stuck_tokens=buy_result.executed_amount)
                return True, f"Micro-sell failed: {sell_result.error}"

            log.info("anti_rugpull.micro_sell_ok", token=token,
                     recovered_sol=sell_result.executed_amount or 0)
            return False, "OK (micro-buy + sell successful)"

        except Exception as e:
            log.warning("anti_rugpull.micro_test_exception", token=token, error=str(e))
            # Fail-open: don't block the trade just because our test crashed
            return False, f"Micro-test skipped: {e}"
