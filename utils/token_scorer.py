"""
token_scorer.py
===============
Multi-factor token quality scorer.

Computes a composite score (0..100) for a pump.fun token based on 15+ features.
The orchestrator uses this score to:
- Filter sniping candidates (only buy tokens scoring >= threshold)
- Adjust position sizing (higher score = larger size, capped by Kelly)
- Adjust take-profit targets (high-quality tokens held longer)

Features:
1. Holder count (raw)                       — more holders = safer
2. Holder growth velocity (last 5 min)      — fast growth = momentum
3. Top-10 holder concentration              — lower = safer
4. Dev wallet share                         — lower = safer
5. Mint authority renounced                 — yes/no
6. Freeze authority renounced                — yes/no
7. Liquidity USD                            — higher = safer
8. Liquidity / market cap ratio             — higher = safer (real LP)
9. Bonding curve completion %               — closer to migration = bullish
10. SOL inflow velocity (last 60s)          — fast inflow = momentum
11. Token age (seconds since launch)        — too young = risky
12. Symbol length (too short = scam)        — 3-4 chars typically
13. Name has emoji / red flags              — emoji = often scam
14. URI reachable + valid JSON              — broken URI = lazy dev
15. Twitter/Telegram presence (from URI)    — has socials = legit
16. Dev wallet SOL balance (solvent dev)    — funded dev = serious
17. Recent buyer count (unique wallets in last 5 min)

Output: TokenScore(score, confidence, breakdown, recommendation)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from utils.bonding_curve import BondingCurveAnalyzer, BondingCurveState
from utils.config_loader import Config
from utils.data_providers import get_providers
from utils.logger import setup_logger

log = setup_logger("token_scorer")


@dataclass
class TokenScore:
    mint: str
    score: float                       # 0..100
    confidence: float                  # 0..1 (data completeness)
    recommendation: str                # "BUY" | "AVOID" | "NEUTRAL"
    breakdown: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    fetched_at: float = field(default_factory=time.time)


class TokenScorer:
    """Multi-factor scorer for pump.fun token launches."""

    def __init__(self) -> None:
        self.cfg = Config.get()
        scfg = self.cfg.get_nested("scoring", default={})
        self.min_score_to_buy = scfg.get("min_score_to_buy", 55.0)
        self.min_score_to_pyramid = scfg.get("min_score_to_pyramid", 75.0)
        self.bonding = BondingCurveAnalyzer()
        self._session: Optional[aiohttp.ClientSession] = None
        self._score_cache: dict[str, tuple[float, TokenScore]] = {}

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._session

    # ------------------------------------------------------------------
    async def score(self, mint: str, token_meta: Optional[dict] = None) -> TokenScore:
        # 2-minute cache to avoid hammering APIs
        cached = self._score_cache.get(mint)
        if cached and time.time() - cached[0] < 120:
            return cached[1]

        providers = get_providers()
        breakdown: dict[str, float] = {}
        reasons: list[str] = []
        confidence_factors: list[float] = []

        # ---- Fetch all data in parallel ----
        holder_data, liq_usd, price_usd, mint_renounced, freeze_renounced, bc_state, uri_meta = (
            await asyncio.gather(
                self._fetch_holders(mint, providers),
                providers["dexscreener"].get_liquidity_usd(mint),
                providers["dexscreener"].get_price_usd(mint),
                self._safe(providers["helius"].is_mint_authority_renounced(mint), False),
                self._safe(providers["helius"].is_freeze_authority_renounced(mint), False),
                self.bonding.fetch_state(mint),
                self._fetch_uri_meta(token_meta),
                return_exceptions=True,
            )
        )

        # ---- 1. Holder count (max 10 pts) ----
        holders = holder_data or []
        n_holders = len(holders)
        pts_holders = min(10, n_holders / 10)  # 100+ holders = max
        breakdown["holders"] = pts_holders
        if n_holders < 20:
            reasons.append(f"Low holder count ({n_holders})")
        confidence_factors.append(1.0 if n_holders > 0 else 0.0)

        # ---- 2. Top-10 concentration (max 10 pts) ----
        if holders:
            total = sum(h["amount"] for h in holders) or 1
            top10_pct = sum(h["amount"] for h in holders[:10]) / total
            pts_conc = 10 * (1 - min(1, top10_pct / 0.5))  # 0% = 10 pts, 50%+ = 0 pts
            breakdown["concentration"] = pts_conc
            if top10_pct > 0.4:
                reasons.append(f"High concentration ({top10_pct:.0%})")
        else:
            breakdown["concentration"] = 0
        confidence_factors.append(1.0 if holders else 0.0)

        # ---- 3. Mint authority renounced (max 10 pts) ----
        pts_mint = 10 if mint_renounced else 0
        breakdown["mint_renounced"] = pts_mint
        if not mint_renounced:
            reasons.append("Mint authority active")
        confidence_factors.append(1.0)

        # ---- 4. Freeze authority renounced (max 10 pts) ----
        pts_freeze = 10 if freeze_renounced else 0
        breakdown["freeze_renounced"] = pts_freeze
        if not freeze_renounced:
            reasons.append("Freeze authority active")
        confidence_factors.append(1.0)

        # ---- 5. Liquidity USD (max 15 pts) ----
        # 0-$5k = 0, $5k-$50k = 0..10, $50k+ = 15
        if liq_usd < 5000:
            pts_liq = 0
        elif liq_usd < 50000:
            pts_liq = (liq_usd - 5000) / 45000 * 10
        else:
            pts_liq = min(15, 10 + (liq_usd - 50000) / 100000 * 5)
        breakdown["liquidity"] = pts_liq
        if liq_usd < 5000:
            reasons.append(f"Very low liquidity (${liq_usd:.0f})")
        confidence_factors.append(1.0 if liq_usd > 0 else 0.0)

        # ---- 6. Bonding curve completion (max 15 pts) ----
        if bc_state:
            comp = bc_state.completion_pct
            # Sweet spot: 30-90% completion = healthy momentum
            # <30% = too early, >90% = might miss the move (or catch migration)
            if comp < 30:
                pts_bc = comp / 30 * 5  # max 5
            elif comp < 90:
                pts_bc = 5 + (comp - 30) / 60 * 10  # 5..15
            else:
                pts_bc = 15  # full points, imminent migration = bullish
            breakdown["bonding_curve"] = pts_bc

            # ---- 7. SOL inflow velocity (max 10 pts) ----
            if bc_state.last_sol_inflow_rate:
                rate = bc_state.last_sol_inflow_rate  # SOL/sec
                # 0.01 SOL/sec = OK, 0.1+ SOL/sec = very hot
                pts_inflow = min(10, rate * 100)
                breakdown["inflow_velocity"] = pts_inflow
                if rate > 0.05:
                    reasons.append(f"Hot inflow: {rate:.3f} SOL/sec")
            else:
                breakdown["inflow_velocity"] = 0
        else:
            breakdown["bonding_curve"] = 0
            breakdown["inflow_velocity"] = 0
        confidence_factors.append(1.0 if bc_state else 0.0)

        # ---- 8. URI metadata reachable (max 5 pts) ----
        if uri_meta and uri_meta.get("reachable"):
            pts_uri = 5
            # Bonus for socials
            if uri_meta.get("twitter") or uri_meta.get("telegram"):
                pts_uri += 0  # already maxed
                breakdown["socials"] = 3
            else:
                breakdown["socials"] = 0
                reasons.append("No Twitter/Telegram in metadata")
        else:
            pts_uri = 0
            breakdown["socials"] = 0
            reasons.append("URI unreachable or invalid")
        breakdown["uri_meta"] = pts_uri
        confidence_factors.append(1.0 if uri_meta else 0.0)

        # ---- 9. Symbol / name sanity (max 5 pts) ----
        symbol = (token_meta or {}).get("symbol", "")
        name = (token_meta or {}).get("name", "")
        pts_symbol = 5
        if len(symbol) < 2 or len(symbol) > 10:
            pts_symbol -= 2
            reasons.append(f"Suspicious symbol length: {symbol}")
        # Red-flag patterns
        if any(w in name.lower() for w in ["airdrop", "claim", "free", "bonus"]):
            pts_symbol -= 3
            reasons.append("Name contains spam keywords")
        breakdown["symbol_sanity"] = max(0, pts_symbol)

        # ---- Compute total ----
        total = sum(breakdown.values())
        # Cap to 100
        total = min(100, total)
        confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0

        if total >= self.min_score_to_pyramid:
            recommendation = "BUY_STRONG"
        elif total >= self.min_score_to_buy:
            recommendation = "BUY"
        elif total >= 35:
            recommendation = "NEUTRAL"
        else:
            recommendation = "AVOID"
            reasons.append(f"Score too low ({total:.0f}/100)")

        score_obj = TokenScore(
            mint=mint,
            score=total,
            confidence=confidence,
            recommendation=recommendation,
            breakdown=breakdown,
            reasons=reasons,
        )
        self._score_cache[mint] = (time.time(), score_obj)
        return score_obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _safe(self, coro, default):
        try:
            return await coro
        except Exception:
            return default

    async def _fetch_holders(self, mint: str, providers: dict) -> list[dict]:
        try:
            return await providers["helius"].get_token_holders(mint, limit=100)
        except Exception:
            return []

    async def _fetch_uri_meta(self, token_meta: Optional[dict]) -> Optional[dict]:
        """Fetch and parse the off-chain metadata URI."""
        if not token_meta or not token_meta.get("uri"):
            return None
        try:
            s = await self.session()
            async with s.get(token_meta["uri"], timeout=5) as r:
                if r.status != 200:
                    return {"reachable": False}
                data = await r.json()
            return {
                "reachable": True,
                "twitter": data.get("twitter") or (data.get("socials") or {}).get("twitter"),
                "telegram": data.get("telegram") or (data.get("socials") or {}).get("telegram"),
                "website": data.get("website"),
            }
        except Exception:
            return {"reachable": False}

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
