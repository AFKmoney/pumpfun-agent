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
        Run a backtest that calls the REAL strategy.evaluate() on historical
        data — not a hardcoded SMA crossover. Aggregates results across tokens.

        For each token, we replay the OHLCV series bar-by-bar, feeding the
        strategy a synthetic event each bar with the price + RSI context it
        expects. We honor the BUY/SELL signals the strategy actually emits,
        and simulate trades with the strategy's own stop-loss / take-profit
        (from the returned Signal), not a fixed 10/30%.
        """
        all_returns: list[float] = []
        max_tokens = int(self.cfg.get("backtest_max_tokens", 5))
        for token in sample_tokens[:max_tokens]:
            try:
                prices = await self.fetch_ohlcv(token, interval="5m", limit=1000)
                if len(prices) < 100:
                    continue
                # Grid scalping has its own replay (no usable evaluate())
                if getattr(strategy, "name", "") == "grid_scalping":
                    all_returns.extend(self._simulate_grid(strategy, prices))
                    continue
                # Replay the series. We track one open position at a time per token
                # (matching the risk manager's per-token single-position model).
                entry_idx: Optional[int] = None
                entry_price = 0.0
                active_sl_pct = 0.10
                active_tp_pct = 0.30
                for i in range(30, len(prices) - 1):
                    price = float(prices[i])
                    # If we hold a position, check SL/TP/exit against current price
                    if entry_idx is not None:
                        pnl = (price - entry_price) / entry_price
                        # Also let the strategy opine on an exit (e.g. momentum
                        # SELL on RSI overbought). We feed it the bar context.
                        exit_signal = await self._replay_evaluate(
                            strategy, token, prices, i, price,
                        )
                        should_exit = (
                            pnl <= -active_sl_pct
                            or pnl >= active_tp_pct
                            or (exit_signal is not None
                                and exit_signal.signal_type == SignalType.SELL)
                        )
                        if should_exit:
                            all_returns.append((price - entry_price) / entry_price)
                            entry_idx = None
                        continue
                    # No open position: ask the strategy for a BUY
                    buy_signal = await self._replay_evaluate(
                        strategy, token, prices, i, price,
                    )
                    if buy_signal is not None and buy_signal.signal_type == SignalType.BUY:
                        entry_idx = i
                        entry_price = price
                        # Honor the strategy's own SL/TP if provided
                        if buy_signal.stop_loss_pct:
                            active_sl_pct = buy_signal.stop_loss_pct / 100.0
                        if buy_signal.take_profit_pct:
                            active_tp_pct = buy_signal.take_profit_pct / 100.0
                # Force-close any still-open position at the last available price
                if entry_idx is not None:
                    last_price = float(prices[-1])
                    all_returns.append((last_price - entry_price) / entry_price)
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

    async def _replay_evaluate(
        self, strategy: BaseStrategy, token: str, prices: np.ndarray,
        bar_idx: int, price: float,
    ) -> Optional[Signal]:
        """
        Build the synthetic event the strategy's evaluate() expects from a
        historical bar, and call it. Each strategy consumes a different event
        shape, so we dispatch on strategy.name.

        Grid scalping is handled separately (it has no usable evaluate(); see
        _simulate_grid below).

        Returns the Signal (BUY/SELL/HOLD/None) the strategy emits, or None if
        the strategy has no replay-compatible evaluate().
        """
        name = getattr(strategy, "name", "")
        try:
            if name == "momentum":
                # Momentum expects {token, chain, vol5m, change5m, rsi}
                window = prices[max(0, bar_idx - 14): bar_idx + 1]
                rsi = self._rsi(list(window)) if len(window) >= 15 else None
                change5m = (
                    (price - float(prices[bar_idx - 12])) / float(prices[bar_idx - 12]) * 100
                    if bar_idx >= 13 else 0.0
                )
                event = {
                    "token": token, "chain": "solana",
                    "vol5m": 0.0, "change5m": change5m, "rsi": rsi,
                }
                return await strategy.evaluate(event)
            elif name in ("sniping", "copy_trade", "grid_scalping"):
                # These either need on-chain launch events that can't be
                # reconstructed from OHLCV (sniping/copy), or have no usable
                # evaluate() (grid uses _tick). grid is handled by
                # _simulate_grid() in the caller. Skip here.
                return None
            else:
                # Unknown strategy: try a generic price event
                event = {"token": token, "chain": "solana", "price": price}
                try:
                    return await strategy.evaluate(event)
                except Exception:
                    return None
        except Exception as e:
            log.debug("backtester.replay_evaluate_failed",
                      strategy=name, token=token, error=str(e))
            return None

    def _simulate_grid(self, strategy: BaseStrategy, prices: np.ndarray) -> list[float]:
        """
        Replay the grid-scalping strategy on a historical price series.
        Replicates the production logic: build a grid around the initial price,
        emit BUY when price drops to a buy level, SELL when it rises to a sell
        level. Returns the list of per-trade returns.
        """
        levels = int(getattr(strategy, "levels", 8))
        spacing_pct = float(getattr(strategy, "spacing_pct", 1.5))
        if len(prices) < 30:
            return []
        mid = float(prices[0])
        # Build buy/sell levels
        buy_levels = [mid * (1 - (spacing_pct / 100) * i) for i in range(1, levels + 1)]
        sell_levels = [mid * (1 + (spacing_pct / 100) * i) for i in range(1, levels + 1)]
        buy_filled = [False] * levels
        returns: list[float] = []
        open_buys: list[float] = []  # entry prices of currently-held grid tranches
        for p in prices[1:]:
            # Process sells first (realize gains on existing positions)
            for si, sp in enumerate(sell_levels):
                if p >= sp and any(entry < sp for entry in open_buys):
                    # Sell the cheapest eligible tranche at this sell level
                    entry = min(e for e in open_buys if e < sp)
                    open_buys.remove(entry)
                    returns.append((sp - entry) / entry)
            # Process buys
            for bi, bp in enumerate(buy_levels):
                if p <= bp and not buy_filled[bi]:
                    buy_filled[bi] = True
                    open_buys.append(bp)
        # Force-close remaining at the last price
        last = float(prices[-1])
        for entry in open_buys:
            returns.append((last - entry) / entry)
        return returns

    @staticmethod
    def _rsi(prices: list[float], period: int = 14) -> Optional[float]:
        if len(prices) < period + 1:
            return None
        gains, losses = 0.0, 0.0
        for i in range(-period, 0):
            diff = prices[i] - prices[i - 1]
            if diff > 0:
                gains += diff
            else:
                losses -= diff
        if losses == 0:
            return 100.0
        rs = (gains / period) / (losses / period)
        return 100.0 - (100.0 / (1.0 + rs))

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
