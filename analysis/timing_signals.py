"""
timing_signals.py
=================
Batch E — Timing & signal-quality modules adapted from OMEGA.

1. TimeOfDayAlpha (B12):
   Memecoin activity is NOT uniform across the day. US session (14:00-21:00 UTC)
   and Asia session (00:00-08:00 UTC) have the highest volume/volatility.
   Funding time (00:00/08:00/16:00 UTC on perps) doesn't apply to spot, but
   the session-driven volume peaks DO. We modulate conviction by time.

2. MultiTimeframeSignal (B15):
   Checks alignment across multiple timeframes. On pump.fun, "timeframes" map
   to order-flow windows: 1min, 5min, 15min, 1h. If all point the same
   direction = max conviction. Conflicting = noise, reduce conviction.

3. SentimentNLP (B24):
   Lightweight keyword-based sentiment (no LLM, no API key cost). Scans the
   text of mentions already in the SentimentAnalyzer buffer for crypto-
   specific bullish/bearish keywords. Fast, free, and surprisingly effective
   as a complement to the hype_score.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from analysis.order_flow import OrderFlowAnalyzer
from analysis.sentiment import SentimentAnalyzer
from utils.logger import setup_logger

log = setup_logger("timing_signals")


# =====================================================================
# B12 — TimeOfDayAlpha
# =====================================================================
class TimeOfDayAlpha:
    """
    Conviction multiplier based on UTC hour. Memecoin volume concentrates in
    the US afternoon and Asian evening. We boost conviction during high-
    activity windows and dampen during dead hours.

    Returns a multiplier 0.7..1.2 to apply to signal conviction.
    """
    # (start_hour, end_hour, multiplier) in UTC
    SESSIONS = [
        (14, 21, 1.20),   # US afternoon/evening — peak retail memecoin activity
        (0, 8, 1.10),     # Asia session — high SOL activity
        (8, 14, 0.90),    # EU morning — moderate
        (21, 24, 0.80),   # Late US/early Asia gap — dead hours
    ]

    def multiplier(self, ts: Optional[float] = None) -> float:
        hour = time.gmtime(ts or time.time()).tm_hour
        for start, end, mult in self.SESSIONS:
            if start <= hour < end:
                return mult
        return 0.90  # default


# =====================================================================
# B15 — MultiTimeframeSignal
# =====================================================================
@dataclass
class MTFReading:
    direction: float        # -1..+1 (net direction across TFs)
    alignment: float        # 0..1 (1 = all TFs agree)
    conviction: float       # alignment * |direction|


class MultiTimeframeSignal:
    """
    Checks order-flow direction across 1m/5m/15m/1h windows. Full alignment =
    max conviction. A token that's bullish on 1m but bearish on 1h is noise.
    """
    WINDOWS = [60, 300, 900, 3600]  # 1m, 5m, 15m, 1h

    def __init__(self, order_flow: OrderFlowAnalyzer) -> None:
        self.order_flow = order_flow

    def assess(self, token: str) -> MTFReading:
        directions = []
        for window in self.WINDOWS:
            snap = self.order_flow.snapshot(token, window)
            if snap is None or snap.total_trades < 3:
                directions.append(0.0)
                continue
            # Direction = normalized buy/sell balance
            if snap.sell_count == 0:
                d = 1.0 if snap.buy_count > 0 else 0.0
            else:
                d = (snap.buy_count - snap.sell_count) / (snap.buy_count + snap.sell_count)
            directions.append(d)

        # Net direction (weighted toward longer TFs for stability)
        weights = [0.15, 0.25, 0.30, 0.30]
        net = sum(w * d for w, d in zip(weights, directions))

        # Alignment: do all non-zero directions share the same sign?
        active = [d for d in directions if abs(d) > 0.1]
        if not active:
            alignment = 0.0
        else:
            signs = set(1 if d > 0 else -1 for d in active)
            alignment = 1.0 if len(signs) == 1 else 0.3

        conviction = alignment * abs(net)
        return MTFReading(
            direction=round(net, 3),
            alignment=round(alignment, 3),
            conviction=round(conviction, 3),
        )


# =====================================================================
# B24 — SentimentNLP (keyword-based, no LLM)
# =====================================================================
# Curated crypto/memecoin keyword lexicons. Tuned for pump.fun context.
BULLISH_KEYWORDS = {
    "moon", "pump", "bullish", "buy", "long", "send", "based", "gem", "alpha",
    "early", "degen", "ape", "wagmi", "lfg", "going up", "accumulate",
    "undervalued", "breakout", "run", "fly", "100x", "next pepe", "dev is based",
}
BEARISH_KEYWORDS = {
    "rug", "dump", "sell", "bearish", "scam", "honey", "honeypot", "dead",
    "abandoned", "jeet", "paper", "soft rug", "exit", "fade", "short",
    "crash", "reckt", "rekt", "liquidation", "dev sold", "sell walls",
}


class SentimentNLP:
    """
    Keyword-based sentiment scorer. Scans mention text in the SentimentAnalyzer
    buffer for bullish/bearish crypto keywords. Returns a -1..+1 score.
    No API calls, no LLM — pure string matching, very fast.
    """
    def __init__(self, sentiment: SentimentAnalyzer) -> None:
        self.sentiment = sentiment

    def score(self, token: str) -> float:
        """
        Scan recent mentions for keywords. Returns net sentiment -1..+1.
        """
        snap = self.sentiment.snapshot(token)
        if snap is None:
            return 0.0
        # Access the raw mentions buffer
        mentions = getattr(self.sentiment, "_mentions", {}).get(token, [])
        if not mentions:
            return 0.0
        # Only last 30 min
        now = time.time()
        recent = [m for m in mentions if now - getattr(m, "ts", 0) < 1800]
        if not recent:
            return 0.0
        bullish = 0
        bearish = 0
        for m in recent:
            text = (getattr(m, "text", "") or "").lower()
            if any(kw in text for kw in BULLISH_KEYWORDS):
                bullish += 1
            if any(kw in text for kw in BEARISH_KEYWORDS):
                bearish += 1
        total = bullish + bearish
        if total == 0:
            return 0.0
        return max(-1.0, min(1.0, (bullish - bearish) / total))
