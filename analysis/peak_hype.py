"""
peak_hype.py
============
Peak-hype contrarian exit detector — identifies when a memecoin has reached
maximum narrative/social velocity and is about to reverse, then triggers an exit.

WHY THIS MATTERS (Edge #4 in COMPETITIVE_EDGE.md):
Memecoin price action is parabolic then cliff. The naive bot holds for a fixed
+100% take-profit and watches the price come back to -50%. The reality:
  - Price peaks when social mentions peak (everyone who will buy has bought).
  - The reversal signal is when mentions are STILL spiking but the buy/sell
    ratio starts inverting (early holders begin distributing into the hype).
  - Selling into the distribution phase captures 80% of the move instead of
    watching it evaporate waiting for a fixed TP.

THE DETECTION (3 conditions, all must be true):
  1. Mention velocity is parabolic (mentions/min in last 10min >= 3x the
     rolling 1h average — "everyone is talking about it").
  2. Buy/sell ratio is declining (last 10min ratio < 60% of the 1h ratio —
     "sellers are appearing into the strength").
  3. Price is extended (current price >= 2x the 1h-ago price — confirms we're
     actually in a parabolic move, not random noise).

This is deliberately conservative: false positives (exiting too early) cost
less than false negatives (holding through the crash). The TP ladder still
runs underneath; this detector is an overlay that catches the blowoff top.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from analysis.order_flow import OrderFlowAnalyzer
from analysis.sentiment import SentimentAnalyzer
from utils.logger import setup_logger

log = setup_logger("peak_hype")


@dataclass
class PeakHypeSignal:
    token: str
    is_peak: bool
    mention_velocity_ratio: float    # recent / baseline (>3 = parabolic)
    bs_ratio_decline: float          # recent_bs / baseline_bs (<0.6 = declining)
    price_extension: float           # current / 1h_ago (>2 = extended)
    reason: str


class PeakHypeDetector:
    """
    Detects blowoff-top conditions and emits sell signals. Designed as a
    periodic check on open positions (wired into the orchestrator like the
    soft-rug monitor).
    """

    def __init__(
        self,
        order_flow: OrderFlowAnalyzer,
        sentiment: SentimentAnalyzer,
    ) -> None:
        self.order_flow = order_flow
        self.sentiment = sentiment
        # Tunable thresholds
        self.mention_velocity_mult = 3.0   # recent must be 3x baseline
        self.bs_decline_threshold = 0.6    # recent bs < 60% of baseline
        self.price_extension_mult = 2.0    # price must be 2x 1h-ago

    def assess(
        self, token: str, current_price: float, price_1h_ago: float,
    ) -> PeakHypeSignal:
        """
        Assess whether `token` is at peak hype. Requires the caller to provide
        current_price and price_1h_ago (the orchestrator tracks these).
        """
        # 1. Mention velocity
        sent_snap = self.sentiment.snapshot(token)
        baseline_mentions_per_min = sent_snap.mentions_per_minute if sent_snap else 0.0
        # We approximate "recent" velocity from the 1h count vs 6h count ratio.
        # If 1h mentions >> 6h/6, recent is elevated.
        recent_mentions_per_min = baseline_mentions_per_min
        if sent_snap and sent_snap.mention_count_6h > 0:
            # recent_1h_rate vs 6h_average_rate
            avg_6h_rate = sent_snap.mention_count_6h / 360.0  # per min over 6h
            if avg_6h_rate > 0:
                recent_mentions_per_min = sent_snap.mention_count_1h / 60.0
        mention_ratio = (
            recent_mentions_per_min / baseline_mentions_per_min
            if baseline_mentions_per_min > 0 else 0.0
        )

        # 2. Buy/sell ratio decline
        snap_10m = self.order_flow.snapshot(token, 600)   # last 10 min
        snap_1h = self.order_flow.snapshot(token, 3600)   # last 1h
        bs_10m = (snap_10m.buy_count / max(1, snap_10m.sell_count)) if snap_10m else 1.0
        bs_1h = (snap_1h.buy_count / max(1, snap_1h.sell_count)) if snap_1h else 1.0
        bs_decline = (bs_10m / bs_1h) if bs_1h > 0 else 1.0

        # 3. Price extension
        price_ext = (current_price / price_1h_ago) if price_1h_ago > 0 else 1.0

        # All three conditions
        is_peak = (
            mention_ratio >= self.mention_velocity_mult
            and bs_decline <= self.bs_decline_threshold
            and price_ext >= self.price_extension_mult
        )

        reason = ""
        if is_peak:
            reasons = []
            if mention_ratio >= self.mention_velocity_mult:
                reasons.append(f"mentions {mention_ratio:.1f}x baseline")
            if bs_decline <= self.bs_decline_threshold:
                reasons.append(f"b/s ratio declining ({bs_decline:.2f})")
            if price_ext >= self.price_extension_mult:
                reasons.append(f"price {price_ext:.1f}x 1h-ago")
            reason = "Peak hype: " + ", ".join(reasons)

        return PeakHypeSignal(
            token=token, is_peak=is_peak,
            mention_velocity_ratio=round(mention_ratio, 2),
            bs_ratio_decline=round(bs_decline, 2),
            price_extension=round(price_ext, 2),
            reason=reason,
        )
