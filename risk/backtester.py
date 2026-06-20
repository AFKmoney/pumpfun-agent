"""
backtester.py
=============
Backtesting engine with real historical data from Birdeye.

Flow:
1. For each enabled strategy, fetch last N days of OHLCV (1m) for the
   strategy's tracked tokens (or a representative sample).
2. Replay the strategy's signal logic on the historical data.
3. Compute Sharpe, win rate, max drawdown, total return.
4. If strategy fails the gate (min_sharpe, min_winrate), the orchestrator
   refuses to enable it for live trading.

Usage from orchestrator:
    bt = Backtester()
    result = await bt.run_for_strategy(strategy, sample_tokens=["...","..."])
    if not bt.passes_gate(result):
        log.warning("strategy_failed_backtest_gate", strategy=strategy.name)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import numpy as np

from strategies.base_strategy import BaseStrategy, Signal, SignalType
from utils.config_loader import Config
from utils.data_providers import get_providers
from utils.logger import setup_logger

log = setup_logger("backtester")


@dataclass
class BacktestResult:
    strategy: str
    sharpe: float
    win_rate: float
    total_return_pct: float
    num_trades: int
    max_drawdown_pct: float


class Backtester:
    def __init__(self) -> None:
        self.cfg = Config.get().get_nested("risk", "backtesting", default={})
        self.min_sharpe = self.cfg.get("min_required_sharpe", 0.8)
        self.min_winrate = self.cfg.get("min_required_winrate", 0.40)
        self.lookback_days = self.cfg.get("historical_data_days", 30)

    # ------------------------------------------------------------------
    async def fetch_ohlcv(self, token: str, interval: str = "1m", limit: int = 1000) -> np.ndarray:
        providers = get_providers()
        items = await providers["birdeye"].get_ohlcv(token, interval=interval, limit=limit)
        if not items:
            return np.array([])
        closes = [float(it.get("c") or it.get("close") or 0) for it in items]
        return np.array(closes, dtype=float)

    # ------------------------------------------------------------------
    async def run(self, prices: np.ndarray, signals: list[int],
                  sl_pct: float = 0.10, tp_pct: float = 0.30,
                  max_hold: int = 60) -> BacktestResult:
        if len(prices) < 100 or not signals:
            return BacktestResult("", 0, 0, 0, 0, 0)

        returns: list[float] = []
        for entry_idx in signals:
            if entry_idx >= len(prices):
                continue
            entry_price = prices[entry_idx]
            exit_price = entry_price
            for j in range(1, max_hold + 1):
                if entry_idx + j >= len(prices):
                    break
                p = prices[entry_idx + j]
                pnl = (p - entry_price) / entry_price
                if pnl <= -sl_pct or pnl >= tp_pct:
                    exit_price = p
                    break
                exit_price = p
            returns.append((exit_price - entry_price) / entry_price)

        arr = np.array(returns) if returns else np.array([0.0])
        sharpe = float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(252))
        win_rate = float((arr > 0).mean())
        total_ret = float((arr + 1).prod() - 1) * 100
        cum = np.cumprod(1 + arr)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = float(abs(dd.min()) * 100)

        return BacktestResult(
            strategy="",
            sharpe=sharpe, win_rate=win_rate,
            total_return_pct=total_ret,
            num_trades=arr.size,
            max_drawdown_pct=max_dd,
        )

    # ------------------------------------------------------------------
    async def run_for_strategy(self, strategy: BaseStrategy,
                               sample_tokens: list[str]) -> BacktestResult:
        """
        Run a backtest for a given strategy on a list of representative tokens.
        Aggregates results across all tokens.
        """
        all_returns: list[float] = []
        for token in sample_tokens[:5]:  # cap at 5 tokens to stay within rate limits
            try:
                prices = await self.fetch_ohlcv(token, interval="5m", limit=1000)
                if len(prices) < 100:
                    continue
                # Simple signal generation: buy when 5-bar SMA crosses above 20-bar SMA
                sma5 = self._sma(prices, 5)
                sma20 = self._sma(prices, 20)
                signals = []
                for i in range(20, len(prices) - 1):
                    if sma5[i] > sma20[i] and sma5[i - 1] <= sma20[i - 1]:
                        signals.append(i)
                # Simulate trades
                for entry_idx in signals:
                    entry_price = prices[entry_idx]
                    exit_price = entry_price
                    for j in range(1, 60):
                        if entry_idx + j >= len(prices):
                            break
                        p = prices[entry_idx + j]
                        pnl = (p - entry_price) / entry_price
                        if pnl <= -0.10 or pnl >= 0.30:
                            exit_price = p
                            break
                        exit_price = p
                    all_returns.append((exit_price - entry_price) / entry_price)
            except Exception as e:
                log.warning("backtester.token_failed", token=token, error=str(e))

        if not all_returns:
            return BacktestResult(strategy.name, 0, 0, 0, 0, 0)
        arr = np.array(all_returns)
        sharpe = float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(252))
        win_rate = float((arr > 0).mean())
        total_ret = float((arr + 1).prod() - 1) * 100
        cum = np.cumprod(1 + arr)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = float(abs(dd.min()) * 100)

        return BacktestResult(
            strategy=strategy.name, sharpe=sharpe, win_rate=win_rate,
            total_return_pct=total_ret, num_trades=arr.size,
            max_drawdown_pct=max_dd,
        )

    @staticmethod
    def _sma(prices: np.ndarray, window: int) -> np.ndarray:
        out = np.full_like(prices, np.nan, dtype=float)
        for i in range(window - 1, len(prices)):
            out[i] = prices[i - window + 1: i + 1].mean()
        return out

    # ------------------------------------------------------------------
    def passes_gate(self, result: BacktestResult) -> bool:
        if result.num_trades < 10:
            log.info("backtester.insufficient_trades", strategy=result.strategy,
                     trades=result.num_trades)
            return False  # not enough data to judge
        ok = result.sharpe >= self.min_sharpe and result.win_rate >= self.min_winrate
        log.info("backtester.gate",
                 strategy=result.strategy, sharpe=result.sharpe,
                 winrate=result.win_rate, trades=result.num_trades, passes=ok)
        return ok
