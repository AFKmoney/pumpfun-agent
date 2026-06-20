"""
attribution.py
==============
Trade attribution + weekly AlphaSignal weight re-tuning.

Two responsibilities:

1. record_buy_features(): called by the executor at BUY time. Captures the
   per-analyzer scores so that when the trade later closes, we can correlate
   "which signal components were high on winners vs losers".

2. AutoTuner: a background task that runs weekly (Sunday 04:00 UTC). It loads
   all closed trade_features rows, fits a logistic regression
   profitable ~ score_components, and writes the normalized coefficients back
   to the alpha_weights table. The AlphaSignalGenerator picks these up on its
   next _refresh_weights_from_db() call.

This is the closed-loop learning system that distinguishes a static-heuristic
bot from one that adapts to which of its own signals actually make money.

Design choices:
- Logistic regression (not deep learning): we expect sparse data (<10k trades
  for a long time), 8 features, and want interpretable per-component weights.
- numpy-only (no sklearn dependency): a few lines of gradient descent suffice
  and keeps the dependency footprint small.
- Safety: the tuner never zeroes a component outright (floor 0.01) and never
  lets any single component exceed 0.50 (concentration cap). The priors
  (hand-tuned WEIGHTS) act as L2 regularization toward sensible defaults when
  data is scarce.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import numpy as np

from analysis.alpha_signal import AlphaSignalGenerator
from utils.config_loader import Config
from utils.logger import setup_logger
from utils.persistence import Persistence

log = setup_logger("attribution")

# The 8 component names, in a fixed column order for the regression.
COMPONENTS = [
    "order_flow_score", "smart_money_score", "token_quality_score",
    "lifecycle_score", "liquidity_score", "sentiment_score",
    "bonding_curve_score", "mev_penalty_score",
]
# Map DB column -> AlphaSignal weight key.
COMPONENT_TO_WEIGHT_KEY = {
    "order_flow_score": "order_flow",
    "smart_money_score": "smart_money",
    "token_quality_score": "token_quality",
    "lifecycle_score": "lifecycle",
    "liquidity_score": "liquidity",
    "sentiment_score": "sentiment",
    "bonding_curve_score": "bonding_curve",
    "mev_penalty_score": "mev_penalty",
}


def record_buy_features(
    chain: str, token: str, strategy: str,
    alpha_signal: Optional[AlphaSignalGenerator] = None,
    component_scores: Optional[dict[str, float]] = None,
) -> None:
    """
    Capture the per-analyzer scores for a trade at buy time.
    Either pass an AlphaSignal object's component_scores, or a pre-built dict.
    """
    try:
        scores = component_scores or {}
        # If given an alpha_signal generator, we expect the caller to have
        # already called .generate() and pass its component_scores dict.
        features = {
            "chain": chain,
            "token": token,
            "strategy": strategy,
            "entry_ts": time.time(),
            "alpha_score": scores.get("alpha_score"),
            "order_flow_score": scores.get("order_flow"),
            "smart_money_score": scores.get("smart_money"),
            "token_quality_score": scores.get("token_quality"),
            "lifecycle_score": scores.get("lifecycle"),
            "liquidity_score": scores.get("liquidity"),
            "sentiment_score": scores.get("sentiment"),
            "bonding_curve_score": scores.get("bonding_curve"),
            "mev_penalty_score": scores.get("mev_penalty"),
        }
        Persistence.get().insert_trade_features(features)
    except Exception as e:
        log.warning("attribution.record_failed", token=token, error=str(e))


def record_close_features(chain: str, token: str, entry_ts: float, pnl_pct: float) -> None:
    """Mark the trade_features row as closed with the realized pnl."""
    try:
        Persistence.get().close_trade_features(
            chain, token, entry_ts, time.time(), pnl_pct,
        )
    except Exception as e:
        log.warning("attribution.close_failed", token=token, error=str(e))


class AutoTuner:
    """Weekly AlphaSignal weight re-tuning via logistic regression."""

    def __init__(self, alpha_signal: AlphaSignalGenerator) -> None:
        self.alpha_signal = alpha_signal
        self.cfg = Config.get()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        # How many closed trades we need before trusting the fit (else keep priors).
        self.min_samples = self.cfg.get_nested("analysis", "autotuner",
                                               default={}).get("min_samples", 50)
        # L2 strength toward the prior weights (higher = more conservative).
        self.l2 = self.cfg.get_nested("analysis", "autotuner",
                                      default={}).get("l2_strength", 5.0)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())
        log.info("autotuner.started", min_samples=self.min_samples)

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        """Run the tuning job weekly, plus once at startup (after 5min warmup)."""
        # Initial warmup delay so we don't fight with startup I/O.
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=300)
            return
        except asyncio.TimeoutError:
            pass
        await self.run_once()
        while not self._stop_event.is_set():
            # Sleep until next Sunday 04:00 UTC
            delay = self._seconds_until_next_run()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                pass
            await self.run_once()

    def _seconds_until_next_run(self) -> float:
        """Seconds until the next Sunday 04:00 UTC."""
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        # weekday(): Monday=0 .. Sunday=6
        days_ahead = (6 - now.weekday()) % 7
        target = (now + timedelta(days=days_ahead)).replace(
            hour=4, minute=0, second=0, microsecond=0,
        )
        if target <= now:
            target += timedelta(days=7)
        return (target - now).total_seconds()

    async def run_once(self) -> Optional[dict[str, float]]:
        """
        Fit logistic regression profitable ~ component_scores, write normalized
        coefficients to alpha_weights. Returns the new weights or None if
        insufficient data.
        """
        try:
            db = Persistence.get()
            rows = db.load_closed_trade_features(since_ts=0.0, limit=10000)
            if len(rows) < self.min_samples:
                log.info("autotuner.skipped_low_samples", n=len(rows),
                         min=self.min_samples)
                return None

            # Build X (n x 8) and y (n,), filling missing scores with 50 (neutral).
            X = np.array(
                [[float(r.get(c) or 50.0) for c in COMPONENTS] for r in rows],
                dtype=np.float64,
            )
            y = np.array([int(r.get("profitable") or 0) for r in rows], dtype=np.float64)

            # Normalize features to 0..1 (scores are 0..100).
            X = X / 100.0

            # Prior weights as the regularization target (also normalized 0..1).
            prior = np.array([
                self.alpha_signal.WEIGHTS[COMPONENT_TO_WEIGHT_KEY[c]] for c in COMPONENTS
            ], dtype=np.float64)

            # Logistic regression via gradient descent with L2 toward `prior`.
            # We fit raw coefficients, then take softmax to get normalized weights
            # in [0,1] summing to 1 — directly usable as AlphaSignal weights.
            w = np.zeros(len(COMPONENTS), dtype=np.float64)
            lr = 0.05
            n_iters = 800
            n = len(y)
            for _ in range(n_iters):
                z = X @ w
                p = 1.0 / (1.0 + np.exp(-z))
                grad = (X.T @ (p - y)) / n + self.l2 * (w - prior * 8.0)
                w -= lr * grad

            # Softmax to normalized positive weights summing to 1.
            w_exp = np.exp(w - w.max())
            new_weights_arr = w_exp / w_exp.sum()

            # Safety: floor 0.01, cap 0.50, renormalize.
            new_weights_arr = np.maximum(new_weights_arr, 0.01)
            new_weights_arr = np.minimum(new_weights_arr, 0.50)
            new_weights_arr = new_weights_arr / new_weights_arr.sum()

            new_weights = {
                COMPONENT_TO_WEIGHT_KEY[c]: float(new_weights_arr[i])
                for i, c in enumerate(COMPONENTS)
            }
            db.upsert_alpha_weights(new_weights)
            # Apply immediately.
            self.alpha_signal._refresh_weights_from_db()
            log.info("autotuner.tuned", n_samples=n, new_weights={
                k: round(v, 3) for k, v in new_weights.items()
            })
            return new_weights
        except Exception as e:
            log.error("autotuner.run_failed", error=str(e))
            return None
