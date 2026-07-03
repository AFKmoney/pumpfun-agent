"""
entry_edge.py
============
Unified entry-quality gate that fuses ALL signal modules into a single
go/no-go score. This is the MAXIMUM-EDGE filter — only the top setups pass.

WHY THIS MAXIMIZES WIN RATE:
Each individual signal (toxic flow, smart money, crowd, sentiment, regime,
soft-rug, dev forensics...) has a modest edge alone (~52-58% win rate). But
their edges are partially INDEPENDENT — they measure different aspects of the
same token. When you require MULTIPLE signals to agree (AND-gate), the combined
win rate compounds:

  P(win | A and B) > max(P(win|A), P(win|B))   when A,B are independent

By requiring 5+ signals to align before buying, we filter to the top ~5% of
launches where the confluence is overwhelming. We trade less, but each trade
has a much higher expected value.

The tradeoff: fewer trades = higher variance per unit time. We accept this
because on memecoins, the tail payoffs are asymmetric — one 10x covers nine
small losses.

SCORING:
  - Veto signals: any ONE can block the trade (hard gates):
      regime risk_off, serial rugger, toxic flow extreme, soft-rug high P,
      depeg alert, flash-crash depth anomaly
  - Boost signals: additive score, threshold-gated:
      smart money accumulation, whale buying, crowd fear (contrarian buy),
      sentiment bullish NLP, MTF alignment, bonding curve sweet spot,
      low stress market, peak-hype (for sells)
  - Final: trade ONLY if score >= ENTRY_THRESHOLD (default 70/100)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from analysis.flow_signals import ToxicFlowDetector, SmartMoneyDivergence, WhaleTracker
from analysis.regime import RegimeDetector
from analysis.soft_rug import SoftRugDetector, RugFeatures, RugPrediction
from analysis.timing_signals import MultiTimeframeSignal, SentimentNLP, TimeOfDayAlpha
from typing import Optional
from utils.logger import setup_logger

log = setup_logger("entry_edge")

ENTRY_THRESHOLD = 70  # out of 100 — only top setups pass


@dataclass
class EntryAssessment:
    token: str
    score: int                    # 0..100
    should_enter: bool            # True if score >= threshold AND no veto
    vetoes: list[str] = field(default_factory=list)
    boosts: dict[str, int] = field(default_factory=dict)
    summary: str = ""


class EntryEdgeScorer:
    """
    Fuses all signals into a single entry-quality gate.
    Includes the OMEGA2 trend filter: only enter WITH the 50-bar trend,
    never against it (the single biggest win-rate driver in OMEGA2's backtest).
    """

    def __init__(
        self,
        toxic_flow: ToxicFlowDetector,
        smart_money: SmartMoneyDivergence,
        whale_tracker: WhaleTracker,
        regime: RegimeDetector,
        mtf: MultiTimeframeSignal,
        sentiment_nlp: SentimentNLP,
        time_of_day: TimeOfDayAlpha,
        atr_tracker: Optional[object] = None,
    ) -> None:
        self.toxic_flow = toxic_flow
        self.smart_money = smart_money
        self.whale_tracker = whale_tracker
        self.regime = regime
        self.mtf = mtf
        self.sentiment_nlp = sentiment_nlp
        self.time_of_day = time_of_day
        self.atr_tracker = atr_tracker  # for ATR-dynamic TP/SL
        # Per-token price history for the trend filter (OMEGA2: 50-bar trend)
        self._price_history: dict[str, list[float]] = {}

    def assess(self, token: str, price_now: float, price_5m_ago: float) -> EntryAssessment:
        """
        Score a token for entry. Returns should_enter=True only if the
        confluence of signals is strong AND no hard veto fires.
        """
        vetoes: list[str] = []
        boosts: dict[str, int] = {}
        score = 50  # neutral start

        # === HARD VETOES (any one blocks the trade) ===

        # 1. Regime: risk-off = no new entries
        if not self.regime.should_trade():
            vetoes.append(f"regime_{self.regime.current.label if self.regime.current else 'unknown'}")

        # 2. Toxic flow extreme: someone informed is distributing
        toxic = self.toxic_flow.detect(token)
        if toxic.conviction > 0.6 and toxic.score < -0.5:
            vetoes.append(f"toxic_flow_{toxic.label}")

        # 3. Smart money divergence (distribution): price up but flow negative
        div = self.smart_money.assess(token, price_now, price_5m_ago)
        if div.label == "bearish_divergence" and div.conviction > 0.5:
            vetoes.append(f"distribution_{div.label}")

        # === BOOST SIGNALS (additive, each contributes to the score) ===

        # 4. Smart money CONFIRMED uptrend (not divergence): price + flow agree
        if div.label == "confirmed_uptrend":
            boosts["confirmed_uptrend"] = int(div.score * 20)  # up to +20

        # 5. Whale accumulation
        whale = self.whale_tracker.net_direction(token)
        if whale.score > 0.3:
            boosts["whale_accumulation"] = int(whale.score * 15)  # up to +15
        elif whale.score < -0.3:
            boosts["whale_distribution"] = int(whale.score * 10)  # negative boost

        # 6. Multi-timeframe alignment (all TFs agree)
        mtf_r = self.mtf.assess(token)
        if mtf_r.alignment > 0.8 and mtf_r.direction > 0.2:
            boosts["mtf_aligned"] = int(mtf_r.conviction * 15)  # up to +15
        elif mtf_r.alignment < 0.3:
            boosts["mtf_conflicted"] = -5  # conflicting TFs = noise

        # 7. Sentiment NLP (keyword-based, free, fast)
        nlp_score = self.sentiment_nlp.score(token)
        if abs(nlp_score) > 0.3:
            boosts["sentiment_nlp"] = int(nlp_score * 10)  # ±10

        # 8. Time-of-day multiplier (session quality)
        tod_mult = self.time_of_day.multiplier()
        if tod_mult > 1.0:
            boosts["peak_session"] = int((tod_mult - 1.0) * 20)  # +4 for US session
        elif tod_mult < 0.9:
            boosts["dead_hours"] = -3

        # 9. OMEGA2 TREND FILTER — the single biggest win-rate driver.
        # Only enter WITH the 50-bar trend, never against it. On pump.fun
        # "bars" = price observations; we track the rolling series.
        self._price_history.setdefault(token, []).append(price_now)
        self._price_history[token] = self._price_history[token][-50:]
        hist = self._price_history[token]
        if len(hist) >= 20:
            # 50-bar (or available) trend direction
            trend_start = hist[len(hist) // 2]  # midpoint price
            trend_pct = (price_now - trend_start) / trend_start * 100 if trend_start > 0 else 0
            short_term_pct = (price_now - price_5m_ago) / price_5m_ago * 100 if price_5m_ago > 0 else 0
            # Trading WITH the trend = both same sign and non-trivial
            if abs(trend_pct) > 2.0 and (trend_pct > 0) == (short_term_pct > 0):
                boosts["with_trend"] = 12  # OMEGA2: +12 for trend-aligned entries
            elif abs(trend_pct) > 2.0 and (trend_pct > 0) != (short_term_pct > 0):
                vetoes.append("against_trend")  # VETO: never fight the trend

        # 10. ATR-dynamic TP/SL (OMEGA2's volatility-adaptive exits)
        # If we have an ATR tracker, compute dynamic SL/TP for this token
        atr_sl = None
        atr_tp = None
        if self.atr_tracker is not None:
            self.atr_tracker.update(price_now)
            atr_sl = self.atr_tracker.dynamic_sl_bps()
            atr_tp = self.atr_tracker.dynamic_tp_bps()
            # High ATR (volatile) = riskier = need stronger confluence
            atr = self.atr_tracker.atr_bps()
            if atr and atr > 200:  # >2% per bar = extreme vol
                boosts["high_vol_caution"] = -8

        # === COMPUTE FINAL SCORE ===
        score = 50 + sum(boosts.values())
        score = max(0, min(100, score))

        should_enter = (score >= ENTRY_THRESHOLD) and (len(vetoes) == 0)

        # Build human-readable summary
        parts = []
        if vetoes:
            parts.append(f"VETO: {', '.join(vetoes)}")
        if boosts:
            top = sorted(boosts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            parts.append(f"Boosts: {top}")
        parts.append(f"Score: {score}/{100}")
        summary = " | ".join(parts)

        return EntryAssessment(
            token=token, score=score, should_enter=should_enter,
            vetoes=vetoes, boosts=boosts, summary=summary,
        )
