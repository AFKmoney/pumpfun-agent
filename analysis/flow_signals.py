"""
flow_signals.py
===============
Batch A — Flow & divergence signals adapted from OMEGA's CEX techniques to
pump.fun's on-chain reality.

On Binance perps these read CVD (cumulative volume delta) and CEX whale flows.
On pump.fun there's no CVD feed, but the same CONCEPTS transfer because the
underlying truth is identical: WHO is buying/selling matters more than how much.

Three detectors, each returns a normalized signal:

1. ToxicFlowDetector (B6):
   Detects CVD acceleration — a spike in net sell volume that diverges from
   price action. On pump.fun: sell volume accelerating while price holds =
   someone informed is distributing into retail buys. "Toxic" because taking
   the other side loses.

2. SmartMoneyDivergence (B7):
   Price rising but net flow turning negative = classic distribution. Smart
   money sells into retail FOMO. On pump.fun: price up 50% in 10min BUT the
   buy/sell ratio inverted from 4:1 to 1:2 = whales exiting.

3. WhaleTracker (B3):
   Unified whale flow. On pump.fun: tracks large trades (>= whale threshold)
   and computes net whale direction. Whales accumulating = bullish; whales
   distributing = the top is near.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from analysis.order_flow import OrderFlowAnalyzer, OrderFlowSnapshot
from utils.logger import setup_logger

log = setup_logger("flow_signals")


@dataclass
class FlowSignal:
    """Normalized flow signal [-1, +1]. Positive = bullish flow, negative = bearish."""
    score: float
    conviction: float       # 0..1
    label: str              # human-readable
    detail: dict            # for audit


# =====================================================================
# B6 — ToxicFlowDetector
# =====================================================================
class ToxicFlowDetector:
    """
    Detects toxic (informed) flow: sell volume accelerating against a stable
    or rising price. Someone knows something and is exiting.

    On pump.fun this manifests as: the bonding curve price holds (retail keeps
    buying) but SELL count and SELL size spike — the smart money is feeding
    shares into the bid. The cascade comes when retail exhausts.

    Signal: -1 (strongly toxic/bearish) .. +1 (clean/bullish flow).
    """
    # Sell volume must be >= this multiple of buy volume to flag toxicity
    TOXIC_SELL_RATIO = 1.8
    # Minimum trades to consider the snapshot statistically meaningful
    MIN_TRADES = 15

    def __init__(self, order_flow: OrderFlowAnalyzer) -> None:
        self.order_flow = order_flow

    def detect(self, token: str) -> FlowSignal:
        snap = self.order_flow.snapshot(token, 300)  # 5-min window
        if snap is None or snap.total_trades < self.MIN_TRADES:
            return FlowSignal(0.0, 0.0, "insufficient_data", {})

        # Net sell pressure = sell volume / total volume
        sell_share = snap.sell_count / max(1, snap.total_trades)
        # Toxic if sells dominate AND whale sells present
        whale_sell_pressure = snap.whale_sell_count / max(1, snap.total_trades)
        net_sol = snap.net_sol_flow  # negative = net outflow (bearish)

        # Score: more sells + negative net flow + whale sells = more toxic
        toxicity = 0.0
        if snap.buy_count > 0 and snap.sell_count / max(1, snap.buy_count) >= self.TOXIC_SELL_RATIO:
            toxicity += 0.4
        if net_sol < 0:
            toxicity += min(0.4, abs(net_sol) / 10.0)
        if whale_sell_pressure > 0.05:
            toxicity += min(0.3, whale_sell_pressure * 3.0)

        score = max(-1.0, -toxicity)  # toxic = negative
        conviction = min(1.0, toxicity)
        label = "toxic_distribution" if toxicity > 0.5 else (
            "mild_toxic" if toxicity > 0.2 else "clean_flow"
        )
        return FlowSignal(
            score=round(score, 3), conviction=round(conviction, 3),
            label=label,
            detail={
                "sell_share": round(sell_share, 3),
                "net_sol_flow": round(net_sol, 3),
                "whale_sells": snap.whale_sell_count,
                "total_trades": snap.total_trades,
            },
        )


# =====================================================================
# B7 — SmartMoneyDivergence
# =====================================================================
class SmartMoneyDivergence:
    """
    Detects price/flow divergence: price UP but net flow DOWN = distribution.

    This is THE classic smart-money exit signature. The price keeps climbing
    (retail FOMO) but the cumulative net flow turns negative (smart money
    selling into the rally). The reversal follows.

    On pump.fun we need price history (caller provides current vs N-min-ago)
    and the order flow snapshot. Divergence = price change sign != flow sign.

    Signal: -1 (strong bearish divergence) .. +1 (confirmed uptrend, flow agrees).
    """
    def __init__(self, order_flow: OrderFlowAnalyzer) -> None:
        self.order_flow = order_flow

    def assess(self, token: str, price_now: float, price_5m_ago: float) -> FlowSignal:
        snap = self.order_flow.snapshot(token, 300)
        if snap is None or snap.total_trades < 10 or price_5m_ago <= 0:
            return FlowSignal(0.0, 0.0, "insufficient_data", {})

        price_change_pct = (price_now - price_5m_ago) / price_5m_ago * 100
        # Normalize flow to a direction: positive = net buying
        bs_ratio = snap.buy_count / max(1, snap.sell_count)
        flow_bullish = bs_ratio > 1.0
        price_bullish = price_change_pct > 2.0  # ignore noise < 2%

        if not price_bullish:
            # Price flat/down — no divergence to detect (or it's just bearish)
            score = max(-0.3, min(0.3, price_change_pct / 20.0))
            return FlowSignal(round(score, 3), 0.2, "no_divergence_flat",
                              {"price_pct": round(price_change_pct, 2)})

        if not flow_bullish:
            # BEARISH DIVERGENCE: price up, flow negative = distribution
            divergence_strength = min(1.0, abs(price_change_pct) / 30.0)
            flow_neg = min(1.0, (1.0 / bs_ratio) / 3.0) if bs_ratio > 0 else 1.0
            score = -min(1.0, divergence_strength * flow_neg)
            conviction = min(1.0, divergence_strength * 0.8)
            return FlowSignal(
                round(score, 3), round(conviction, 3),
                "bearish_divergence",
                {"price_pct": round(price_change_pct, 2),
                 "bs_ratio": round(bs_ratio, 2),
                 "net_sol": round(snap.net_sol_flow, 2)},
            )

        # Confirmed uptrend: price up AND flow bullish
        score = min(1.0, (price_change_pct / 30.0) * (bs_ratio / 3.0))
        return FlowSignal(round(score, 3), 0.4, "confirmed_uptrend",
                          {"price_pct": round(price_change_pct, 2),
                           "bs_ratio": round(bs_ratio, 2)})


# =====================================================================
# B3 — WhaleTracker
# =====================================================================
class WhaleTracker:
    """
    Unified whale flow direction. On pump.fun: counts large trades (>= whale
    threshold in SOL) and computes net whale direction (buys minus sells).

    Whales accumulating (net whale buy) while retail is still hesitant = early
    bullish. Whales distributing (net whale sell) while retail FOMOs = the top.

    Signal: -1 (whales dumping) .. +1 (whales accumulating).
    """
    def __init__(self, order_flow: OrderFlowAnalyzer) -> None:
        self.order_flow = order_flow

    def net_direction(self, token: str) -> FlowSignal:
        snap = self.order_flow.snapshot(token, 600)  # 10-min window
        if snap is None or (snap.whale_buy_count + snap.whale_sell_count) == 0:
            return FlowSignal(0.0, 0.0, "no_whale_activity", {})

        whale_buys = snap.whale_buy_count
        whale_sells = snap.whale_sell_count
        total_whale = whale_buys + whale_sells
        # Net direction: +1 if all buys, -1 if all sells
        net = (whale_buys - whale_sells) / total_whale
        conviction = min(1.0, total_whale / 5.0)  # more whales = more conviction

        if net > 0.3:
            label = "whales_accumulating"
        elif net < -0.3:
            label = "whales_distributing"
        else:
            label = "whales_neutral"

        return FlowSignal(
            score=round(net, 3), conviction=round(conviction, 3),
            label=label,
            detail={"whale_buys": whale_buys, "whale_sells": whale_sells,
                    "total_whale_trades": total_whale},
        )
