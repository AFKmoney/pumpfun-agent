"""
risk_signals.py
===============
Batch B — Adaptive risk management adapted from OMEGA.

1. VolatilityForecast (B8):
   GARCH(1,1)-style / EWMA volatility forecast per token. Predicts the vol
   regime to adjust position sizing (high vol = smaller size).

2. StressIndex (B20):
   Composite 0-100 "VIX of crypto" from BTC vol + SOL vol + market breadth.
   Feeds the AdaptiveRiskManager to throttle sizing in stressed markets.

3. AdaptiveRiskManager (B22):
   Dynamic Kelly fraction that responds to: stress index, per-token vol,
   and loss streak. When stressed or in a losing streak, size down. When
   winning and calm, size up. Replaces the static fixed-size with a
   self-regulating one.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.config_loader import Config
from utils.logger import setup_logger
from utils.persistence import Persistence

log = setup_logger("risk_signals")


# =====================================================================
# B8 — VolatilityForecast (EWMA / GARCH-lite)
# =====================================================================
class VolatilityForecast:
    """
    EWMA volatility forecast per token. Tracks the price series and computes
    a forward-looking vol estimate. High forecast vol -> smaller position size.

    Uses the standard EWMA (lambda=0.94, RiskMetrics convention) which is a
    GARCH(1,1) special case and requires no fitting.
    """
    LAMBDA = 0.94  # RiskMetrics standard

    def __init__(self) -> None:
        # token -> list of (ts, price)
        self._prices: dict[str, list[tuple[float, float]]] = {}

    def update(self, token: str, price: float) -> None:
        """Record a new price observation for the token."""
        if price <= 0:
            return
        hist = self._prices.setdefault(token, [])
        hist.append((time.time(), price))
        # Keep last 2 hours
        now = time.time()
        self._prices[token] = [(t, p) for t, p in hist if now - t < 7200]

    def forecast_vol(self, token: str) -> Optional[float]:
        """
        Return the EWMA volatility forecast (as a per-bar stdev of log returns).
        None if insufficient data.
        """
        hist = self._prices.get(token, [])
        if len(hist) < 10:
            return None
        prices = [p for _, p in hist if p > 0]
        if len(prices) < 10:
            return None
        rets = [math.log(prices[i] / prices[i - 1])
                for i in range(1, len(prices)) if prices[i - 1] > 0]
        if len(rets) < 5:
            return None
        # EWMA: sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_{t-1}^2
        var = rets[0] ** 2
        for r in rets[1:]:
            var = self.LAMBDA * var + (1 - self.LAMBDA) * r ** 2
        return math.sqrt(var)


# =====================================================================
# B20 — StressIndex
# =====================================================================
@dataclass
class StressReading:
    score: int               # 0..100 (higher = more stressed)
    label: str               # "calm" | "elevated" | "stressed" | "panic"
    btc_vol: float
    sol_vol: float
    breadth: float           # fraction of our recent trades that lost


class StressIndex:
    """
    Composite market stress index (the "VIX of crypto" for our context).
    Combines:
      - BTC volatility (proxy for global crypto risk)
      - SOL volatility (our chain's risk)
      - Our own trade breadth (are WE losing? -> market is hostile)

    Feeds AdaptiveRiskManager to throttle sizing when stressed.
    """
    def __init__(self, vol_forecast: VolatilityForecast) -> None:
        self.vol_forecast = vol_forecast
        self._btc_prices: list[tuple[float, float]] = []
        self._sol_prices: list[tuple[float, float]] = []

    def update_macro(self, btc_price: float, sol_price: float) -> None:
        now = time.time()
        self._btc_prices.append((now, btc_price))
        self._sol_prices.append((now, sol_price))
        self._btc_prices = [(t, p) for t, p in self._btc_prices if now - t < 7200]
        self._sol_prices = [(t, p) for t, p in self._sol_prices if now - t < 7200]

    def compute(self) -> StressReading:
        btc_vol = self._stdev(self._btc_prices)
        sol_vol = self._stdev(self._sol_prices)
        breadth = self._our_breadth()
        # Normalize each to 0..1 then blend. BTC vol ~3% = elevated, ~8% = panic.
        btc_component = min(1.0, btc_vol / 0.08) if btc_vol else 0.3
        sol_component = min(1.0, sol_vol / 0.10) if sol_vol else 0.3
        breadth_component = max(0.0, 1.0 - breadth)  # low breadth = stressed
        raw = 0.40 * btc_component + 0.35 * sol_component + 0.25 * breadth_component
        score = int(min(100, max(0, raw * 100)))
        if score >= 75:
            label = "panic"
        elif score >= 55:
            label = "stressed"
        elif score >= 35:
            label = "elevated"
        else:
            label = "calm"
        return StressReading(
            score=score, label=label,
            btc_vol=round(btc_vol or 0, 4),
            sol_vol=round(sol_vol or 0, 4),
            breadth=round(breadth, 3),
        )

    @staticmethod
    def _stdev(history: list[tuple[float, float]]) -> Optional[float]:
        prices = [p for _, p in history if p > 0]
        if len(prices) < 5:
            return None
        rets = [math.log(prices[i] / prices[i - 1])
                for i in range(1, len(prices)) if prices[i - 1] > 0]
        if len(rets) < 3:
            return None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        return math.sqrt(var)

    @staticmethod
    def _our_breadth() -> float:
        """Fraction of our last 20 closed trades that were profitable."""
        try:
            db = Persistence.get()
            rows = db.load_closed_trade_features(since_ts=0.0, limit=20)
            if not rows:
                return 0.5
            wins = sum(1 for r in rows if (r.get("profitable") or 0) == 1)
            return wins / len(rows)
        except Exception:
            return 0.5


# =====================================================================
# B22 — AdaptiveRiskManager
# =====================================================================
class AdaptiveRiskManager:
    """
    Dynamic position sizer. Adjusts the Kelly fraction based on:
      - Market stress (StressIndex): stressed -> size down to 30% of base.
      - Per-token volatility forecast: high vol -> size down.
      - Loss streak: consecutive losses -> size down (tilt protection).

    This does NOT replace the RiskManager (which enforces hard limits). It
    MODULATES the suggested size before the RiskManager's approval gate.
    Returns a multiplier 0..1 to apply to the base size.
    """
    def __init__(self, stress: StressIndex, vol_forecast: VolatilityForecast) -> None:
        self.stress = stress
        self.vol_forecast = vol_forecast
        self._loss_streak = 0
        self._max_streak = 5

    def record_trade_outcome(self, won: bool) -> None:
        """Feed the loss-streak tracker. Call on every closed trade."""
        if won:
            self._loss_streak = 0
        else:
            self._loss_streak = min(self._max_streak, self._loss_streak + 1)

    def size_multiplier(self, token: Optional[str] = None) -> float:
        """
        Return a multiplier 0..1 to apply to the base position size.
        Combines stress, volatility, and loss-streak dampening.
        """
        stress = self.stress.compute()
        # Stress dampening: calm(0) -> 1.0, panic(100) -> 0.3
        stress_mult = 1.0 - (stress.score / 100.0) * 0.7

        # Volatility dampening: high forecast vol -> smaller size
        vol_mult = 1.0
        if token:
            fvol = self.vol_forecast.forecast_vol(token)
            if fvol and fvol > 0:
                # 2% per-bar vol = full size; 10% = 20% of size
                vol_mult = max(0.2, min(1.0, 0.02 / fvol))

        # Loss streak dampening: each consecutive loss -> -12% size
        streak_mult = max(0.3, 1.0 - self._loss_streak * 0.12)

        return max(0.1, min(1.0, stress_mult * vol_mult * streak_mult))
