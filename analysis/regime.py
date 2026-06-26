"""
regime.py
=========
Market regime detector — classifies the Solana memecoin market as risk-on or
risk-off in real-time, and gates trading accordingly.

WHY THIS MATTERS (Edge #5 in COMPETITIVE_EDGE.md):
In a dead market (few launches, low SOL volatility, BTC trending down), the
expected value of EVERY trade is negative — the winners don't pump enough to
cover the losers because there's no inflow to sustain moves. Most retail bots
trade the same regardless of regime and bleed out in chop. We auto-disable
sniping in risk-off conditions, preserving capital for when conditions favor us.

SIGNALS (each 0..1, weighted into a composite regime score):
  - launch_velocity    : pump.fun launches per hour (from our trade stream).
                         High = lots of new activity = risk-on. Low = dead.
  - sol_volatility     : recent SOL price volatility. High = risk-appetite.
  - sol_trend          : SOL 1h price change. Positive = risk-on.
  - btc_trend          : BTC 1h price change (crypto beta). Positive = risk-on.
  - our_recent_winrate : last 20 closed trades win rate. High = our edge is
                         working in current conditions.

The regime score is a weighted blend; >0.5 = risk-on (trade normally), <0.35 =
risk-off (pause new snipes), between = cautious (reduce size).

Data sources:
  - launch_velocity: counts create-events seen by our analytics pipeline.
  - sol/btc trend+vol: fetched from a price API (CoinGecko, free, no key).
  - our winrate: from the trade_features table.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from utils.config_loader import Config
from utils.logger import setup_logger
from utils.persistence import Persistence

log = setup_logger("regime")


@dataclass
class Regime:
    score: float                # 0..1 composite (higher = more risk-on)
    label: str                  # "risk_on" | "cautious" | "risk_off"
    launch_velocity: float      # launches/hour normalized 0..1
    sol_volatility: float       # 0..1
    sol_trend: float            # 0..1
    btc_trend: float            # 0..1
    our_winrate: float          # 0..1
    should_trade: bool          # True if score >= pause_threshold
    fetched_at: float


class RegimeDetector:
    """
    Polls market indicators every N minutes and classifies the regime.
    The orchestrator consults regime.should_trade before emitting snipe signals.
    """

    # Weights (sum to 1.0). Tunable via config.
    WEIGHTS = {
        "launch_velocity": 0.30,
        "sol_volatility": 0.15,
        "sol_trend": 0.20,
        "btc_trend": 0.15,
        "our_winrate": 0.20,
    }

    def __init__(self, launch_counter=None) -> None:
        self.cfg = Config.get()
        rcfg = self.cfg.get_nested("analysis", "regime", default={})
        self.pause_threshold = float(rcfg.get("pause_threshold", 0.35))
        self.caution_threshold = float(rcfg.get("caution_threshold", 0.50))
        self.poll_interval = int(rcfg.get("poll_interval_sec", 300))
        self.coingecko_sol = "https://api.coingecko.com/api/v3/simple/price?ids=solana,bitcoin&vs_currencies=usd&include_24hr_change=true"
        # launch_counter: a callable () -> int returning launches seen in last hour.
        # The analytics pipeline can provide this; default to a simple counter.
        self._launch_counter = launch_counter
        self._current: Optional[Regime] = None
        self._sol_price_history: list[tuple[float, float]] = []  # (ts, price)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    @property
    def current(self) -> Optional[Regime]:
        return self._current

    def should_trade(self) -> bool:
        """Synchronous gate: True if regime allows new snipes."""
        if self._current is None:
            return True  # unknown regime -> default to allowing (fail-open)
        return self._current.should_trade

    async def start(self) -> None:
        # Initial assessment before first poll interval
        await self.assess()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
                return
            except asyncio.TimeoutError:
                pass
            try:
                await self.assess()
            except Exception as e:
                log.error("regime.assess_failed", error=str(e))

    async def assess(self) -> Regime:
        """Compute the current regime from all signals."""
        prices = await self._fetch_sol_btc()
        sol_price = prices.get("sol", 0.0)
        btc_price = prices.get("btc", 0.0)
        sol_change = prices.get("sol_change_24h", 0.0)
        btc_change = prices.get("btc_change_24h", 0.0)

        # Track SOL price history for volatility
        now = time.time()
        self._sol_price_history.append((now, sol_price))
        self._sol_price_history = [(t, p) for t, p in self._sol_price_history if now - t < 3600]
        sol_vol = self._volatility(self._sol_price_history)

        # Launch velocity (from the counter, if wired)
        launches_per_hour = 0.0
        if self._launch_counter:
            try:
                launches_per_hour = float(self._launch_counter())
            except Exception:
                pass

        # Our recent winrate (from trade_features)
        winrate = await self._our_recent_winrate()

        # Normalize each signal to 0..1
        launch_norm = min(1.0, launches_per_hour / 100.0)   # 100+/hr = max risk-on
        sol_vol_norm = min(1.0, sol_vol / 0.05)             # 5% volatility = max
        sol_trend_norm = self._trend_to_unit(sol_change)    # -10%..+10% -> 0..1
        btc_trend_norm = self._trend_to_unit(btc_change)
        wr_norm = max(0.0, min(1.0, winrate))

        score = (
            self.WEIGHTS["launch_velocity"] * launch_norm +
            self.WEIGHTS["sol_volatility"] * sol_vol_norm +
            self.WEIGHTS["sol_trend"] * sol_trend_norm +
            self.WEIGHTS["btc_trend"] * btc_trend_norm +
            self.WEIGHTS["our_winrate"] * wr_norm
        )
        if score >= self.caution_threshold:
            label = "risk_on"
        elif score >= self.pause_threshold:
            label = "cautious"
        else:
            label = "risk_off"

        self._current = Regime(
            score=round(score, 3), label=label,
            launch_velocity=round(launch_norm, 3),
            sol_volatility=round(sol_vol_norm, 3),
            sol_trend=round(sol_trend_norm, 3),
            btc_trend=round(btc_trend_norm, 3),
            our_winrate=round(wr_norm, 3),
            should_trade=(score >= self.pause_threshold),
            fetched_at=now,
        )
        log.info("regime.assessed", **{k: v for k, v in self._current.__dict__.items()})
        return self._current

    async def _fetch_sol_btc(self) -> dict:
        """Fetch SOL + BTC prices and 24h change from CoinGecko (free, no key)."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                async with s.get(self.coingecko_sol) as r:
                    data = await r.json()
            sol = data.get("solana", {})
            btc = data.get("bitcoin", {})
            return {
                "sol": float(sol.get("usd", 0)),
                "btc": float(btc.get("usd", 0)),
                "sol_change_24h": float(sol.get("usd_24h_change", 0)),
                "btc_change_24h": float(btc.get("usd_24h_change", 0)),
            }
        except Exception as e:
            log.warning("regime.price_fetch_failed", error=str(e))
            return {"sol": 0, "btc": 0, "sol_change_24h": 0, "btc_change_24h": 0}

    async def _our_recent_winrate(self) -> float:
        """Win rate of our last 20 closed trades (from trade_features)."""
        try:
            db = Persistence.get()
            rows = db.load_closed_trade_features(since_ts=0.0, limit=20)
            if not rows:
                return 0.4  # neutral default
            wins = sum(1 for r in rows if (r.get("profitable") or 0) == 1)
            return wins / len(rows)
        except Exception:
            return 0.4

    @staticmethod
    def _volatility(history: list[tuple[float, float]]) -> float:
        """Annualized-ish volatility from a price series (stdev of log returns)."""
        if len(history) < 3:
            return 0.0
        prices = [p for _, p in history if p > 0]
        if len(prices) < 3:
            return 0.0
        import math
        rets = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices)) if prices[i - 1] > 0]
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        return math.sqrt(var)

    @staticmethod
    def _trend_to_unit(change_pct: float) -> float:
        """Map a -10%..+10% change to 0..1 (0% = 0.5)."""
        return max(0.0, min(1.0, 0.5 + change_pct / 20.0))
