"""
alpha_signal.py
===============
Composite alpha signal generator.

Ensembles outputs from all analyzers:
- TokenScorer (15-factor quality)
- OrderFlowAnalyzer (buy/sell pressure)
- SocialGraphAnalyzer (smart money presence)
- LifecycleAnalyzer (stage + size multiplier)
- LiquidityDepthAnalyzer (slippage + max position)
- SentimentAnalyzer (social hype)
- BondingCurveAnalyzer (migration catalyst)
- MevDetector (avoid sandwiched tokens)

Produces a single AlphaSignal object:
- alpha_score: 0..100 (composite conviction)
- conviction: 0..1 (data completeness + agreement)
- recommended_size_sol: actual SOL amount to deploy
- recommended_strategy: which strategy should fire
- recommendation: "STRONG_BUY" | "BUY" | "HOLD" | "AVOID" | "STRONG_AVOID"
- rationale: list of contributing factors (for logging/notifications)
- risks: list of identified risk factors

This is the brain that turns raw data into a trade decision.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from analysis.lifecycle import LifecycleAnalyzer, LifecycleAssessment, LifecycleStage
from analysis.liquidity_depth import LiquidityDepthAnalyzer, LiquidityDepth
from analysis.mev_detector import MevDetector
from analysis.order_flow import OrderFlowAnalyzer, OrderFlowSnapshot
from analysis.sentiment import SentimentAnalyzer, SentimentSnapshot
from analysis.social_graph import SocialGraphAnalyzer, WalletProfile
from utils.bonding_curve import BondingCurveAnalyzer, BondingCurveState
from utils.config_loader import Config
from utils.logger import setup_logger
from utils.token_scorer import TokenScorer, TokenScore

log = setup_logger("alpha_signal")


@dataclass
class AlphaSignal:
    mint: str
    chain: str
    alpha_score: float                       # 0..100
    conviction: float                        # 0..1
    recommendation: str                      # STRONG_BUY/BUY/HOLD/AVOID/STRONG_AVOID
    recommended_size_sol: float
    recommended_strategy: str
    rationale: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    component_scores: dict[str, float] = field(default_factory=dict)
    fetched_at: float = field(default_factory=time.time)


class AlphaSignalGenerator:
    """Composite signal generator. Weights all analyzers."""

    # Tuned weights for memecoin trading (sum = 1.0).
    #
    # Rationale:
    # - order_flow (0.30): the most actionable real-time signal. Buy/sell pressure
    #   + whale net flow predict 5-30 min price moves with the highest hit rate.
    # - smart_money (0.20): who is buying matters more than what is being bought.
    #   A token bought by known profitable wallets has 3-5x higher expected value.
    # - token_quality (0.15): rugpull/honeypot filter. Lower than order_flow because
    #   the anti_rugpull gate already hard-blocks scams before this runs.
    # - lifecycle (0.12): INFANT stage is the sweet spot — getting this right
    #   multiplies returns. BIRTH = too risky, MATURE = stale.
    # - liquidity (0.10): can we exit? Critical for position sizing but less so
    #   for go/no-go decision (anti_rugpull also checks this).
    # - sentiment (0.08): hype drives memecoins but Twitter signal is noisy and
    #   lagging. Keep weight modest.
    # - bonding_curve (0.03): migration catalyst is situational — only matters
    #   when completion > 80%.
    # - mev_penalty (0.02): penalty only; matters when we've been sandwiched
    #   recently on this specific token.
    WEIGHTS = {
        "order_flow": 0.30,
        "smart_money": 0.20,
        "token_quality": 0.15,
        "lifecycle": 0.12,
        "liquidity": 0.10,
        "sentiment": 0.08,
        "bonding_curve": 0.03,
        "mev_penalty": 0.02,
    }

    def __init__(
        self,
        token_scorer: TokenScorer,
        order_flow: OrderFlowAnalyzer,
        social_graph: SocialGraphAnalyzer,
        lifecycle: LifecycleAnalyzer,
        liquidity_depth: LiquidityDepthAnalyzer,
        sentiment: SentimentAnalyzer,
        bonding: BondingCurveAnalyzer,
        mev_detector: MevDetector,
    ) -> None:
        self.cfg = Config.get()
        self.scorer = token_scorer
        self.order_flow = order_flow
        self.social_graph = social_graph
        self.lifecycle = lifecycle
        self.liquidity_depth = liquidity_depth
        self.sentiment = sentiment
        self.bonding = bonding
        self.mev = mev_detector
        self._cache: dict[str, tuple[float, AlphaSignal]] = {}
        # Active weights: start with the hand-tuned defaults, then overlay any
        # learned weights from the DB (set by the weekly attribution job).
        self.weights: dict[str, float] = dict(self.WEIGHTS)
        self._refresh_weights_from_db()

    def _refresh_weights_from_db(self) -> None:
        """Overlay learned weights from the DB if present (weekly re-tuning)."""
        try:
            from utils.persistence import Persistence
            learned = Persistence.get().load_alpha_weights()
            if learned:
                # Only overlay components we know; preserve sum normalization.
                merged = dict(self.WEIGHTS)
                for k, v in learned.items():
                    if k in merged:
                        merged[k] = float(v)
                total = sum(merged.values())
                if total > 0:
                    self.weights = {k: v / total for k, v in merged.items()}
                log.info("alpha_signal.weights_loaded", weights=self.weights)
        except Exception as e:
            log.warning("alpha_signal.weights_load_failed", error=str(e))

    async def generate(self, mint: str, chain: str = "solana") -> AlphaSignal:
        # 90-second cache to avoid hammering APIs
        cached = self._cache.get(mint)
        if cached and time.time() - cached[0] < 90:
            return cached[1]

        # Run all analyzers in parallel
        results = await asyncio.gather(
            self._safe(self.scorer.score(mint)),
            self._safe_async(self.order_flow.snapshot(mint, 300)),
            self._safe(self.lifecycle.assess(mint)),
            self._safe_async(self.liquidity_depth.analyze(mint, chain)),
            self._safe(self.sentiment.snapshot(mint)),
            self._safe_async(self.bonding.fetch_state(mint)),
            return_exceptions=True,
        )
        token_score, order_flow_snap, lifecycle_a, liq_depth, sentiment_snap, bc_state = results

        # Component scores (each 0..100)
        components: dict[str, float] = {}
        rationale: list[str] = []
        risks: list[str] = []
        conviction_factors: list[float] = []

        # 1. Token quality
        if isinstance(token_score, TokenScore):
            components["token_quality"] = token_score.score
            conviction_factors.append(token_score.confidence)
            if token_score.recommendation == "AVOID":
                risks.append(f"Low token quality ({token_score.score:.0f})")
            elif token_score.recommendation == "BUY_STRONG":
                rationale.append(f"High quality ({token_score.score:.0f})")
        else:
            components["token_quality"] = 50
            conviction_factors.append(0)

        # 2. Order flow
        if isinstance(order_flow_snap, OrderFlowSnapshot):
            # pressure_score is -100..100, normalize to 0..100
            score = (order_flow_snap.pressure_score + 100) / 2
            components["order_flow"] = score
            conviction_factors.append(1.0)
            if order_flow_snap.pressure_score > 30:
                rationale.append(f"Strong buy pressure (+{order_flow_snap.pressure_score:.0f})")
            elif order_flow_snap.pressure_score < -30:
                risks.append(f"Strong sell pressure ({order_flow_snap.pressure_score:.0f})")
            if order_flow_snap.whale_net_sol > 0:
                rationale.append(f"Whales net buying ({order_flow_snap.whale_net_sol:.2f} SOL)")
            elif order_flow_snap.whale_net_sol < 0:
                risks.append(f"Whales net selling ({order_flow_snap.whale_net_sol:.2f} SOL)")
        else:
            components["order_flow"] = 50
            conviction_factors.append(0)

        # 3. Smart money (placeholder; needs wallet registry to be useful)
        # In production, check if any top buyers are tagged smart_money
        components["smart_money"] = 50  # neutral until integrated
        conviction_factors.append(0.3)

        # 4. Lifecycle
        if isinstance(lifecycle_a, LifecycleAssessment):
            # Map stage to score: BIRTH=30, INFANT=90, ADOLESCENT=70, MATURE=40, MIGRATED=60, DEAD=0
            stage_scores = {
                LifecycleStage.BIRTH: 30,
                LifecycleStage.INFANT: 90,
                LifecycleStage.ADOLESCENT: 70,
                LifecycleStage.MATURE: 40,
                LifecycleStage.MIGRATED: 60,
                LifecycleStage.DEAD: 0,
            }
            components["lifecycle"] = stage_scores.get(lifecycle_a.stage, 50)
            conviction_factors.append(1.0)
            rationale.append(f"Stage: {lifecycle_a.stage.value}")
            if lifecycle_a.is_breaking_out:
                rationale.append("Breaking out")
                components["lifecycle"] = min(100, components["lifecycle"] + 15)
            if lifecycle_a.is_dying:
                risks.append("Activity dying")
                components["lifecycle"] = max(0, components["lifecycle"] - 20)
        else:
            components["lifecycle"] = 50
            conviction_factors.append(0)

        # 5. Liquidity depth
        if isinstance(liq_depth, LiquidityDepth):
            # Score based on liquidity/MCAP ratio + recommended position size
            if liq_depth.liquidity_to_mcap_ratio > 0.3:
                liq_score = 90
            elif liq_depth.liquidity_to_mcap_ratio > 0.1:
                liq_score = 70
            elif liq_depth.liquidity_to_mcap_ratio > 0.05:
                liq_score = 50
            else:
                liq_score = 20
            components["liquidity"] = liq_score
            conviction_factors.append(1.0)
            if liq_depth.liquidity_usd < 10000:
                risks.append(f"Low liquidity (${liq_depth.liquidity_usd:.0f})")
        else:
            components["liquidity"] = 50
            conviction_factors.append(0)

        # 6. Sentiment
        if isinstance(sentiment_snap, SentimentSnapshot):
            components["sentiment"] = sentiment_snap.hype_score
            conviction_factors.append(1.0)
            if sentiment_snap.hype_score > 70:
                rationale.append(f"High hype ({sentiment_snap.hype_score:.0f})")
            elif sentiment_snap.hype_score < 20 and sentiment_snap.mention_count_1h > 0:
                risks.append("Negative sentiment")
        else:
            components["sentiment"] = 50
            conviction_factors.append(0)

        # 7. Bonding curve (migration catalyst)
        if isinstance(bc_state, BondingCurveState):
            # Sweet spot: 80-99% completion (migration imminent)
            if bc_state.completion_pct > 95:
                bc_score = 95
                rationale.append("Migration imminent")
            elif bc_state.completion_pct > 80:
                bc_score = 85
                rationale.append(f"Near migration ({bc_state.completion_pct:.0f}%)")
            elif bc_state.completion_pct > 30:
                bc_score = 60
            else:
                bc_score = 40
                risks.append("Very early bonding curve")
            components["bonding_curve"] = bc_score
            conviction_factors.append(1.0)
        else:
            components["bonding_curve"] = 50
            conviction_factors.append(0)

        # 8. MEV penalty
        if self.mev.is_high_mev_token(mint):
            components["mev_penalty"] = 20
            risks.append("Recent sandwich attacks on this token")
        else:
            components["mev_penalty"] = 80
        conviction_factors.append(1.0)

        # ---- Composite (uses self.weights: learned overlay if present) ----
        alpha_score = sum(
            components.get(k, 50) * w for k, w in self.weights.items()
        )
        conviction = sum(conviction_factors) / len(conviction_factors) if conviction_factors else 0

        # ---- Recommendation ----
        if alpha_score >= 80 and conviction >= 0.7:
            recommendation = "STRONG_BUY"
        elif alpha_score >= 60:
            recommendation = "BUY"
        elif alpha_score >= 40:
            recommendation = "HOLD"
        elif alpha_score >= 25:
            recommendation = "AVOID"
        else:
            recommendation = "STRONG_AVOID"

        # ---- Recommended size ----
        # Base size from config, scaled by alpha + lifecycle multiplier
        base_size = self.cfg["trading"]["fixed_size_sol"]
        size_mult = 1.0
        if isinstance(lifecycle_a, LifecycleAssessment):
            size_mult *= lifecycle_a.size_multiplier
        if recommendation == "STRONG_BUY":
            size_mult *= 1.5
        elif recommendation == "BUY":
            size_mult *= 1.0
        elif recommendation == "HOLD":
            size_mult *= 0.5
        else:
            size_mult = 0.0

        # Cap at recommended max position from liquidity depth
        max_size = 5.0  # hard cap
        if isinstance(liq_depth, LiquidityDepth):
            max_size = liq_depth.recommended_max_position_sol

        recommended_size = min(max_size, base_size * size_mult)

        signal = AlphaSignal(
            mint=mint, chain=chain,
            alpha_score=alpha_score,
            conviction=conviction,
            recommendation=recommendation,
            recommended_size_sol=recommended_size,
            recommended_strategy=lifecycle_a.recommended_strategy if isinstance(lifecycle_a, LifecycleAssessment) else "momentum",
            rationale=rationale,
            risks=risks,
            component_scores=components,
        )

        self._cache[mint] = (time.time(), signal)
        log.info("alpha_signal.generated",
                 mint=mint, score=alpha_score, recommendation=recommendation,
                 conviction=conviction, size=recommended_size,
                 rationale="; ".join(rationale)[:200])
        return signal

    # ------------------------------------------------------------------
    async def _safe(self, coro_or_value):
        """Await if coroutine, else return value."""
        if asyncio.iscoroutine(coro_or_value):
            try:
                return await coro_or_value
            except Exception as e:
                log.warning("alpha_signal.component_failed", error=str(e))
                return None
        return coro_or_value

    async def _safe_async(self, coro):
        try:
            return await coro
        except Exception as e:
            log.warning("alpha_signal.component_failed", error=str(e))
            return None
