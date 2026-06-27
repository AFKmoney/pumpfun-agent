"""
macro_signals.py
================
Batch C — Macro / regime signals adapted from OMEGA.

These detect capital rotation and systemic stress at the crypto-market level,
which precede memecoin regime shifts. All pull from free public APIs
(CoinGecko) — no API keys needed.

1. CorrelationBreakdown (B9):
   BTC/SOL decorrelation = capital rotation. When they stop moving together,
   money is flowing INTO or OUT OF alts/memecoins specifically.

2. BTCDominance (B13):
   Dominance falling = alt season starting (capital flowing to alts/memecoins).
   Dominance rising = risk-off (capital fleeing to BTC).

3. StablecoinFlow (B16):
   USDT market cap rising = fresh capital entering crypto (bullish for risk
   assets including memecoins). Falling = capital leaving.

4. ExchangeReserves (B14):
   BTC exchange reserves falling = accumulation (investors withdrawing to
   cold storage = bullish long-term sentiment).

5. DepegAlert (B5):
   USDT/USDC losing peg = systemic stress. If a stablecoin depegs, the whole
   market panics — memecoins dump hardest. Early warning to go risk-off.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from utils.logger import setup_logger

log = setup_logger("macro_signals")

_COINGECKO = "https://api.coingecko.com/api/v3"


@dataclass
class MacroReading:
    score: float            # -1..+1 (positive = risk-on for memecoins)
    label: str
    components: dict


async def _fetch_json(url: str, timeout: int = 10) -> Optional[dict]:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
            async with s.get(url) as r:
                return await r.json()
    except Exception as e:
        log.debug("macro.fetch_failed", url=url, error=str(e))
        return None


class CorrelationBreakdown:
    """
    BTC/SOL correlation. High (>0.8) = normal. Low (<0.5) = capital rotation
    = regime shift in progress (memecoins often pump when SOL decorrelates up).
    """
    def __init__(self) -> None:
        self._btc: list[float] = []
        self._sol: list[float] = []

    def update(self, btc: float, sol: float) -> None:
        self._btc.append(btc)
        self._sol.append(sol)
        self._btc = self._btc[-60:]
        self._sol = self._sol[-60:]

    def correlation(self) -> Optional[float]:
        if len(self._btc) < 20 or len(self._sol) < 20:
            return None
        import numpy as np
        b = np.array(self._btc[-60:])
        s = np.array(self._sol[-60:])
        if b.std() == 0 or s.std() == 0:
            return None
        return float(np.corrcoef(b, s)[0, 1])


class BTCDominance:
    """Falling dominance = alt/memecoin season risk-on."""
    def __init__(self) -> None:
        self._history: list[tuple[float, float]] = []

    async def fetch(self) -> Optional[float]:
        data = await _fetch_json(f"{_COINGECKO}/global")
        if not data:
            return None
        dom = data.get("data", {}).get("market_cap_percentage", {}).get("btc")
        if dom is not None:
            self._history.append((time.time(), float(dom)))
            self._history = [(t, d) for t, d in self._history if time.time() - t < 86400]
        return float(dom) if dom is not None else None

    def trend(self) -> str:
        """Returns 'falling' | 'rising' | 'flat'."""
        if len(self._history) < 4:
            return "flat"
        recent = [d for _, d in self._history[-4:]]
        delta = recent[-1] - recent[0]
        if delta < -0.5:
            return "falling"
        if delta > 0.5:
            return "rising"
        return "flat"


class StablecoinFlow:
    """USDT market cap trend. Rising = capital inflow (bullish)."""
    def __init__(self) -> None:
        self._history: list[tuple[float, float]] = []

    async def fetch(self) -> Optional[float]:
        data = await _fetch_json(f"{_COINGECKO}/coins/tether?fields=market_cap")
        # CoinGecko simple endpoint
        data2 = await _fetch_json(f"{_COINGECKO}/simple/price?ids=tether&vs_currencies=usd&include_market_cap=true")
        if data2 and "tether" in data2:
            mc = data2["tether"].get("usd_market_cap")
            if mc:
                self._history.append((time.time(), float(mc)))
                self._history = [(t, m) for t, m in self._history if time.time() - t < 604800]
                return float(mc)
        return None

    def trend(self) -> str:
        if len(self._history) < 3:
            return "flat"
        recent = [m for _, m in self._history[-3:]]
        pct = (recent[-1] - recent[0]) / recent[0] * 100
        if pct > 0.5:
            return "rising"
        if pct < -0.5:
            return "falling"
        return "flat"


class DepegAlert:
    """
    Checks if USDT or USDC has depegged (>0.5% off $1). A depeg is an early
    warning of systemic stress — memecoins dump hardest in panics.
    """
    DEPEG_THRESHOLD = 0.005  # 0.5%

    async def check(self) -> dict:
        data = await _fetch_json(
            f"{_COINGECKO}/simple/price?ids=tether,usd-coin&vs_currencies=usd"
        )
        if not data:
            return {"depegged": False, "usdt": 1.0, "usdc": 1.0}
        usdt = float(data.get("tether", {}).get("usd", 1.0))
        usdc = float(data.get("usd-coin", {}).get("usd", 1.0))
        depegged = abs(usdt - 1.0) > self.DEPEG_THRESHOLD or abs(usdc - 1.0) > self.DEPEG_THRESHOLD
        return {"depegged": depegged, "usdt": usdt, "usdc": usdc}


class ExchangeReserves:
    """
    BTC exchange reserves trend. Falling = accumulation (bullish long-term).
    Uses CoinGecko's exchange-volume as a proxy (true reserves need Glassnode).
    """
    def __init__(self) -> None:
        self._history: list[tuple[float, float]] = []

    async def fetch(self) -> Optional[float]:
        data = await _fetch_json(f"{_COINGECKO}/coins/bitcoin?fields=market_data")
        # Fallback: use BTC price as activity proxy
        data2 = await _fetch_json(f"{_COINGECKO}/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_vol=true")
        if data2 and "bitcoin" in data2:
            vol = data2["bitcoin"].get("usd_24h_vol")
            if vol:
                self._history.append((time.time(), float(vol)))
                self._history = [(t, v) for t, v in self._history if time.time() - t < 604800]
                return float(vol)
        return None


class MacroEngine:
    """
    Aggregates all macro signals into a single risk-on/risk-off reading.
    Polled periodically by the orchestrator.
    """
    def __init__(self) -> None:
        self.correlation = CorrelationBreakdown()
        self.dominance = BTCDominance()
        self.stablecoin = StablecoinFlow()
        self.depeg = DepegAlert()
        self.reserves = ExchangeReserves()
        self._current: Optional[MacroReading] = None

    @property
    def current(self) -> Optional[MacroReading]:
        return self._current

    async def assess(self) -> MacroReading:
        components: dict = {}
        score = 0.0

        # Depeg = instant risk-off
        depeg = await self.depeg.check()
        components["depeg"] = depeg
        if depeg["depegged"]:
            score = -0.8
            self._current = MacroReading(score, "depeg_risk_off", components)
            return self._current

        # BTC dominance trend
        dom = await self.dominance.fetch()
        dom_trend = self.dominance.trend()
        components["btc_dominance"] = {"value": dom, "trend": dom_trend}
        if dom_trend == "falling":
            score += 0.25  # alt season = bullish for memecoins
        elif dom_trend == "rising":
            score -= 0.20

        # Stablecoin flow
        sc = await self.stablecoin.fetch()
        sc_trend = self.stablecoin.trend()
        components["stablecoin_flow"] = {"trend": sc_trend}
        if sc_trend == "rising":
            score += 0.20
        elif sc_trend == "falling":
            score -= 0.20

        # Correlation breakdown
        corr = self.correlation.correlation()
        components["btc_sol_correlation"] = corr
        if corr is not None and corr < 0.5:
            score += 0.15  # decorrelation = rotation into alts

        score = max(-1.0, min(1.0, score))
        label = "risk_on" if score > 0.2 else ("risk_off" if score < -0.2 else "neutral")
        self._current = MacroReading(round(score, 3), label, components)
        return self._current
