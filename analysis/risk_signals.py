"""
risk_signals.py
===============
Adaptive risk management — VolatilityForecast + StressIndex + AdaptiveRiskManager
(with Monte Carlo de-risking ported from OMEGA2).

1. VolatilityForecast (B8):
   EWMA volatility forecast per token (RiskMetrics lambda=0.94).

2. StressIndex (B20):
   Composite 0-100 from BTC vol + SOL vol + market breadth.

3. ATRTracker (ported from OMEGA2 risk.py):
   Rolling Average True Range in bps. Used for ATR-dynamic TP/SL — scale
   stops/targets with realized volatility instead of fixed percentages.
   OMEGA2's best backtested idea: SL = 1.5×ATR, TP = 2.5×SL.

4. AdaptiveRiskManager (B22 + OMEGA2 MC):
   Dynamic Kelly fraction × stress × vol × loss-streak × Monte Carlo
   drawdown probability. The MC layer (ported from OMEGA2 risk.py:357)
   simulates 2000 bootstrap paths from our trade history and shrinks size
   when drawdown probability exceeds 30%.
"""
from __future__ import annotations

import math
import time
from collections import deque
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
# ATRTracker (ported from OMEGA2 risk.py) — ATR-dynamic TP/SL
# =====================================================================
class ATRTracker:
    """
    Rolling Average True Range in basis points. OMEGA2's key insight: scale
    stop-loss and take-profit with realized volatility, not fixed %.

    ATR-dynamic TP/SL (the OMEGA2 winning parameter set):
      SL = max(60 bps, 1.5 × ATR_bps)     # tighter in calm markets
      TP = 2.5 × SL                        # 2.5:1 R/R minimum

    On pump.fun, "True Range" per bar = |price_now - price_prev| (we don't
    have separate H/L on a bonding curve, so we approximate with close-to-close).
    """
    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._closes: deque = deque(maxlen=period + 1)

    def update(self, price: float) -> None:
        if price > 0:
            self._closes.append(price)

    def atr_bps(self) -> Optional[float]:
        """
        Return the ATR in basis points. None if insufficient data.
        Approximates TR as |close[i] - close[i-1]| (no intrabar H/L available).
        """
        if len(self._closes) < self.period + 1:
            return None
        closes = list(self._closes)
        # True Range = |close[i] - close[i-1]| (bonding curve approximation)
        trs = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                trs.append(abs(closes[i] - closes[i - 1]) / closes[i - 1] * 10_000)
        if not trs:
            return None
        atr = sum(trs[-self.period:]) / min(self.period, len(trs))
        # Clamp to a sane range (OMEGA2 used 30-300 bps)
        return max(10.0, min(500.0, atr))

    def dynamic_sl_bps(self) -> float:
        """ATR-dynamic stop-loss: 1.5×ATR, floored at 60 bps."""
        atr = self.atr_bps()
        if atr is None:
            return 150.0  # default 1.5% if no data
        return max(60.0, 1.5 * atr)

    def dynamic_tp_bps(self) -> float:
        """ATR-dynamic take-profit: 2.5× the stop-loss."""
        return 2.5 * self.dynamic_sl_bps()


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
        # Monte Carlo de-risking pool (ported from OMEGA2 risk.py)
        # Stores per-trade returns for bootstrap simulation.
        self._mc_returns: deque = deque(maxlen=500)
        self._mc_multiplier: float = 1.0

    def record_trade_outcome(self, won: bool, pnl_pct: float = 0.0) -> None:
        """Feed the loss-streak tracker + MC pool. Call on every closed trade."""
        if won:
            self._loss_streak = 0
        else:
            self._loss_streak = min(self._max_streak, self._loss_streak + 1)
        # Feed the MC pool for drawdown-probability estimation
        if pnl_pct != 0.0:
            self._mc_returns.append(pnl_pct / 100.0)
            self._recompute_mc_multiplier()

    def _recompute_mc_multiplier(self) -> None:
        """
        Monte Carlo de-risking (ported from OMEGA2 risk.py:357).
        Simulates 2000 bootstrap paths over a 20-trade horizon from our
        historical returns. If >30% of paths hit a 10% drawdown, we shrink
        the position size proportionally. This is drawdown-aware adaptive
        sizing — the more our recent history shows drawdown risk, the smaller
        we trade.
        """
        if len(self._mc_returns) < 30:
            self._mc_multiplier = 1.0
            return
        import numpy as np
        returns = np.array(list(self._mc_returns))
        # Outlier filter: drop |r| >= 5×std (prevents fat-tail domination)
        std = returns.std()
        if std > 0:
            returns = returns[np.abs(returns) < 5 * std]
        if len(returns) < 20:
            self._mc_multiplier = 1.0
            return
        # Bootstrap: 2000 paths × 20 trades
        n_paths = 2000
        horizon = 20
        rng = np.random.default_rng()
        sampled = rng.choice(returns, size=(n_paths, horizon), replace=True)
        # Cumulative equity curves
        equity = np.cumprod(1 + sampled, axis=1)
        # Max drawdown per path
        peak = np.maximum.accumulate(equity, axis=1)
        drawdown = (equity - peak) / peak
        max_dd = np.min(drawdown, axis=1)  # most negative per path
        # Probability of a 10%+ drawdown
        prob_dd = np.mean(max_dd <= -0.10)
        # Map probability to multiplier: <0.3 → 1.0, >0.8 → 0.2, linear
        if prob_dd < 0.3:
            self._mc_multiplier = 1.0
        elif prob_dd > 0.8:
            self._mc_multiplier = 0.2
        else:
            self._mc_multiplier = 1.0 - (prob_dd - 0.3) / 0.5 * 0.8

    def size_multiplier(self, token: Optional[str] = None) -> float:
        """
        Return a multiplier 0..1 to apply to the base position size.
        Combines stress, volatility, loss-streak, and Monte Carlo drawdown
        probability dampening.
        """
        stress = self.stress.compute()
        # Stress dampening: calm(0) -> 1.0, panic(100) -> 0.3
        stress_mult = 1.0 - (stress.score / 100.0) * 0.7

        # Volatility dampening: high forecast vol -> smaller size
        vol_mult = 1.0
        if token:
            fvol = self.vol_forecast.forecast_vol(token)
            if fvol and fvol > 0:
                vol_mult = max(0.2, min(1.0, 0.02 / fvol))

        # Loss streak dampening
        streak_mult = max(0.3, 1.0 - self._loss_streak * 0.12)

        # Monte Carlo drawdown-probability dampening (ported from OMEGA2)
        mc_mult = self._mc_multiplier

        return max(0.1, min(1.0, stress_mult * vol_mult * streak_mult * mc_mult))
