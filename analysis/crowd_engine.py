"""
crowd_engine.py
===============
Crowd Positioning Engine for pump.fun — adapted from the OMEGA perps design
to the signals that actually exist on Solana memecoins.

THE THESIS (unchanged from the OMEGA design):
> We don't trade the price. We trade the positioning of the crowd: detect when
> it's overcrowded at an extreme, and take the trade that hurts the most people
> (the reverse cascade).

On Binance perps the crowd signals are funding rate + L/S ratio + sentiment.
On pump.fun there are no perpetuals, so "crowd positioning" is measured from:

  1. ORDER FLOW pressure  — buy/sell ratio on the bonding curve. An extreme
     buy/sell ratio (>4:1) means the crowd is all-in long on a token that
     has nowhere to go but down (everyone who will buy has bought).
  2. HOLDER VELOCITY     — how fast new wallets are piling in. A vertical
     spike in unique holders signals FOMO overcrowding (late money, weak
     hands, cascade-prone).
  3. SENTIMENT HYPE       — Twitter/Telegram mention velocity. Parabolic
     hype = retail FOMO peak = the crowd is at its most overcrowded.
  4. BONDING CURVE %      — near-completion (>90%) means the token is about
     to migrate; the crowd expects a pump but migration often disappoints
     (sell the news).

Each signal produces a score in [-1, +1]:
  +1 = crowd is maximally overcrowded LONG (FOMO, euphoria) → we FADE (sell/avoid)
  -1 = crowd is maximally overcrowded SHORT (capitulation, fear) → we FADE (buy)
   0 = neutral / no edge

The engine fuses them into a single CrowdPositioningEvent with a conviction
that is REDUCED when signals disagree (divergent crowd = weak setup).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from analysis.order_flow import OrderFlowAnalyzer
from analysis.sentiment import SentimentAnalyzer
from analysis.lifecycle import LifecycleAnalyzer
from utils.bonding_curve import BondingCurveAnalyzer
from utils.data_providers import get_providers
from utils.logger import setup_logger

log = setup_logger("crowd_engine")


@dataclass(frozen=True)
class CrowdPositioningEvent:
    """Normalized crowd positioning signal for a single token."""
    token: str
    timestamp: str
    # Composite score [-1, +1]. Positive = crowd overcrowded LONG (fade by
    # selling/avoiding). Negative = crowd overcrowded SHORT (fade by buying).
    crowd_score: float
    # 0..1 — how statistically significant the extreme is. High conviction
    # requires all signals to agree on the SAME direction.
    conviction: float
    # "minutes" | "hours" — expected holding period for the contrarian trade.
    horizon: str
    # Per-signal breakdown for audit / LLM autopsy.
    components: dict        # {"order_flow": +0.8, "holder_velocity": +0.6, ...}
    # Human-readable regime label.
    regime_hint: str        # "cascade_imminent" | "euphoria" | "fear" | "neutral"
    # Expected size of the reverse move, in basis points (heuristic).
    expected_move_bps: float


class CrowdPositioningEngine:
    """
    Fuses order-flow + holder-velocity + sentiment + bonding-curve signals
    into a single CrowdPositioningEvent per token.

    Weights are tunable; the fusion deliberately penalizes divergence (3
    signals agreeing = high conviction; 1 signal screaming while others are
    neutral = low conviction, probably noise).
    """

    # Fusion weights (sum to 1.0). Order flow is the most reliable on-chain
    # signal for memecoins; sentiment is noisy but catches retail FOMO peaks.
    WEIGHTS = {
        "order_flow": 0.35,
        "holder_velocity": 0.20,
        "sentiment": 0.25,
        "bonding_curve": 0.20,
    }

    # Thresholds for the individual signal scorers.
    BUY_SELL_EXTREME = 3.0      # b/s ratio >= 3 = crowd all-in long
    HOLDER_VEL_PER_MIN = 10.0   # >=10 new holders/min = FOMO spike
    HYPE_VEL_MULT = 4.0         # mention velocity 4x baseline = euphoria
    BC_COMPLETION_EXTREME = 90.0  # >=90% = migration sell-the-news risk

    def __init__(
        self,
        order_flow: OrderFlowAnalyzer,
        sentiment: SentimentAnalyzer,
        lifecycle: LifecycleAnalyzer,
        bonding: BondingCurveAnalyzer,
    ) -> None:
        self.order_flow = order_flow
        self.sentiment = sentiment
        self.lifecycle = lifecycle
        self.bonding = bonding

    async def assess(self, token: str) -> CrowdPositioningEvent:
        """
        Compute the crowd positioning for a token. Each sub-signal is scored
        [-1,+1] then fused. Returns a frozen CrowdPositioningEvent.
        """
        scores: dict[str, float] = {}

        # 1. Order flow pressure (buy/sell ratio)
        of = self.order_flow.snapshot(token, 600)  # last 10 min
        scores["order_flow"] = self._score_order_flow(of)

        # 2. Holder velocity (from lifecycle observations)
        lc = await self.lifecycle.assess(token)
        scores["holder_velocity"] = self._score_holder_velocity(lc)

        # 3. Sentiment hype
        sent = self.sentiment.snapshot(token)
        scores["sentiment"] = self._score_sentiment(sent)

        # 4. Bonding curve completion
        bc = await self.bonding.fetch_state(token)
        scores["bonding_curve"] = self._score_bonding_curve(bc)

        # --- Fusion ---
        crowd_score = sum(
            self.WEIGHTS.get(k, 0) * v for k, v in scores.items()
        )
        crowd_score = max(-1.0, min(1.0, crowd_score))

        # Conviction: high when signals agree on direction AND magnitude.
        # Divergence penalty: if some signals are +0.8 and others -0.3, that's
        # a weak setup even if the weighted average is high.
        active = [v for v in scores.values() if abs(v) > 0.15]
        if not active:
            conviction = 0.0
        else:
            # Mean direction strength
            mean_abs = sum(abs(v) for v in active) / len(active)
            # Agreement: are all active signals the same sign?
            signs = set(1 if v > 0 else -1 for v in active)
            agreement = 1.0 if len(signs) == 1 else (0.4 if len(active) <= 2 else 0.2)
            conviction = min(1.0, mean_abs * agreement)

        # Horizon: a cascade (all signals screaming) is acute (minutes);
        # mild euphoria is longer (hours).
        if conviction >= 0.7 and abs(crowd_score) >= 0.6:
            horizon = "minutes"
            regime_hint = "cascade_imminent"
        elif abs(crowd_score) >= 0.5:
            horizon = "hours"
            regime_hint = "euphoria" if crowd_score > 0 else "fear"
        elif abs(crowd_score) >= 0.25:
            horizon = "hours"
            regime_hint = "neutral"
        else:
            horizon = "hours"
            regime_hint = "neutral"

        # Expected move: heuristic. Crowds at extreme positioning reverse hard.
        # Base ~200bps for a moderate extreme, scaling up for high conviction.
        expected_move_bps = abs(crowd_score) * conviction * 1500.0

        return CrowdPositioningEvent(
            token=token,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            crowd_score=round(crowd_score, 3),
            conviction=round(conviction, 3),
            horizon=horizon,
            components={k: round(v, 3) for k, v in scores.items()},
            regime_hint=regime_hint,
            expected_move_bps=round(expected_move_bps, 0),
        )

    # ------------------------------------------------------------------
    # Individual signal scorers (each returns [-1, +1])
    # ------------------------------------------------------------------
    def _score_order_flow(self, snap) -> float:
        """
        Extreme buy/sell ratio = crowd all-in long = +1 (we fade by selling).
        Extreme sell-heavy = crowd capitulating = -1 (we fade by buying).
        """
        if snap is None or snap.total_trades < 5:
            return 0.0
        if snap.sell_count == 0:
            return 1.0 if snap.buy_count > 0 else 0.0
        bs_ratio = snap.buy_count / max(1, snap.sell_count)
        # tanh-like saturation at the extremes
        import math
        return max(-1.0, min(1.0, math.tanh((bs_ratio - 1.0) / (self.BUY_SELL_EXTREME - 1.0))))

    def _score_holder_velocity(self, lc) -> float:
        """
        Vertical holder velocity = FOMO pile-in = +1.
        We approximate from lifecycle trade_velocity (a proxy).
        """
        if lc is None:
            return 0.0
        # Use trade_velocity as a proxy for holder velocity (they correlate on
        # fresh launches). High velocity + breakout flag = euphoria.
        tv = getattr(lc, "trade_velocity", 0.0) or 0.0
        import math
        score = math.tanh(tv / self.HOLDER_VEL_PER_MIN)
        # If the lifecycle says "breaking out", amplify (crowd is piling in NOW)
        if getattr(lc, "is_breaking_out", False):
            score = max(score, 0.5) * 1.2
        if getattr(lc, "is_dying", False):
            score = min(score, -0.3)
        return max(-1.0, min(1.0, score))

    def _score_sentiment(self, sent) -> float:
        """
        Parabolic hype score = euphoria = +1. Dead sentiment = slightly negative
        (the crowd has given up = contrarian buy opportunity, but weak).
        """
        if sent is None:
            return 0.0
        hype = getattr(sent, "hype_score", 50.0) or 50.0
        velocity = getattr(sent, "mentions_per_minute", 0.0) or 0.0
        # High hype + high velocity = FOMO peak
        import math
        hype_score = (hype - 50.0) / 50.0  # -1..1
        vel_score = math.tanh(velocity / 5.0)  # saturate around 5 mentions/min
        return max(-1.0, min(1.0, 0.6 * hype_score + 0.4 * vel_score))

    def _score_bonding_curve(self, bc) -> float:
        """
        Near-completion (>90%) = crowd expects migration pump = sell-the-news
        risk = +1. Low completion is neutral (too early to call).
        """
        if bc is None:
            return 0.0
        pct = bc.completion_pct
        if pct < 50:
            return 0.0
        # Linear ramp from 50% (0.0) to 100% (+1.0)
        return max(0.0, min(1.0, (pct - 50.0) / 50.0))


# ----------------------------------------------------------------------
# The ContrarianAgent — fades crowd extremes
# ----------------------------------------------------------------------
from strategies.base_strategy import Signal, SignalType


EXTREME_THRESHOLD = 0.5  # |crowd_score| >= 0.5 = an extreme worth fading
HORIZON_BARS = {"minutes": 12, "hours": 60, "days": 240}  # 1m bars


class ContrarianAgent:
    """
    Fades crowd extremes. The heart of the crowd-positioning thesis.

    On a CrowdPositioningEvent:
      - crowd_score > +0.5 (crowd overcrowded LONG) → emit SELL
      - crowd_score < -0.5 (crowd overcrowded SHORT) → emit BUY
      - conviction scales the suggested size
      - TP is asymmetric: big (the cascade is the payoff), SL is tight
        (a contrarian is wrong often but wins big when right)
    """

    def __init__(self, extreme_threshold: float = EXTREME_THRESHOLD) -> None:
        self.extreme_threshold = extreme_threshold

    def on_positioning(
        self, event: CrowdPositioningEvent, chain: str = "solana",
    ) -> list[Signal]:
        if abs(event.crowd_score) < self.extreme_threshold:
            return []
        # Fade the crowd
        if event.crowd_score > 0:
            # Crowd is long → we SELL (or avoid buying)
            signal_type = SignalType.SELL
            reason = (f"Fade crowd {event.regime_hint}: "
                      f"score={event.crowd_score:+.2f} conv={event.conviction:.2f}")
            # For SELL we exit fully; sizing is handled by the executor
            suggested = 1.0
        else:
            # Crowd is short/afraid → we BUY (contrarian accumulation)
            signal_type = SignalType.BUY
            reason = (f"Contrarian buy (crowd fear): "
                      f"score={event.crowd_score:+.2f} conv={event.conviction:.2f}")
            suggested = min(0.02, event.conviction * 0.03)  # small, conviction-scaled

        # TP/SL asymmetric: aim for the cascade (big), stop tight
        tp_pct = event.expected_move_bps / 100.0  # bps → %
        sl_pct = tp_pct * 0.3                     # 30% of expected move

        return [Signal(
            strategy="contrarian",
            chain=chain,
            token_address=event.token,
            signal_type=signal_type,
            suggested_size_pct=suggested,
            confidence=min(0.85, event.conviction),
            reason=reason,
            stop_loss_pct=max(5.0, sl_pct),
            take_profit_pct=max(10.0, tp_pct),
            metadata={
                "crowd_score": event.crowd_score,
                "conviction": event.conviction,
                "regime_hint": event.regime_hint,
                "components": event.components,
            },
        )]
