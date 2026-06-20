"""
soft_rug.py
===========
Soft-rug early-warning model.

A "soft rug" is a gradual dev/whale dump that doesn't trip the hard anti-rugpull
checks (mint authority still renounced, freeze authority off, holders not yet
concentrated enough) but where the price bleeds out over minutes. Catching it
early — within the first 1-5 minutes of a token's life — lets us exit before
the full dump.

This is a gradient-boosted decision-stump classifier (numpy-only, no sklearn /
xgboost dependency) trained on features observable at or shortly after the
create event:

  - initial_holder_count       (more early holders = more legit interest)
  - top3_concentration_pct     (top-3 holders' % of supply; high = rug-prone)
  - dev_retained_pct           (% of supply still in the dev wallet at T+30s)
  - sell_velocity_30s          (sells per minute in first 30s; high = dump)
  - buy_sell_ratio_30s         (<1 = net selling pressure)
  - liquidity_usd              (thin liquidity = easy to manipulate)
  - bonding_curve_completion_pct
  - age_seconds

Label: rug=1 if the token's price dropped >50% within 10 minutes of launch,
else rug=0. Labels come from our own closed trade_features history (we already
record entry + pnl; a token we held that hit stop-loss is a positive rug label).

The model is small (50 stumps, depth 1) on purpose: interpretable, fast, and
robust on the sparse data we'll have for a long time. It persists to a single
JSON file (model weights) and retrains nightly from the trade_features table.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from utils.config_loader import Config
from utils.logger import setup_logger
from utils.persistence import Persistence

log = setup_logger("soft_rug")

# Feature column order (fixed). All features are numeric floats; missing
# values are imputed to the column median at training time.
FEATURES = [
    "initial_holder_count",
    "top3_concentration_pct",
    "dev_retained_pct",
    "sell_velocity_30s",
    "buy_sell_ratio_30s",
    "liquidity_usd",
    "bonding_curve_completion_pct",
    "age_seconds",
]


@dataclass
class RugFeatures:
    """Observable features for a token, used to score rug probability."""
    initial_holder_count: float = 0.0
    top3_concentration_pct: float = 0.0
    dev_retained_pct: float = 0.0
    sell_velocity_30s: float = 0.0
    buy_sell_ratio_30s: float = 1.0
    liquidity_usd: float = 0.0
    bonding_curve_completion_pct: float = 0.0
    age_seconds: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.initial_holder_count, self.top3_concentration_pct,
            self.dev_retained_pct, self.sell_velocity_30s,
            self.buy_sell_ratio_30s, self.liquidity_usd,
            self.bonding_curve_completion_pct, self.age_seconds,
        ], dtype=np.float64)


@dataclass
class RugPrediction:
    rug_probability: float          # 0..1
    is_high_risk: bool              # True if rug_probability >= threshold
    top_risk_factors: list[str] = field(default_factory=list)


class _DecisionStump:
    """Single-feature threshold classifier (depth-1 tree)."""
    def __init__(self) -> None:
        self.feature_idx: int = 0
        self.threshold: float = 0.0
        self.left_value: float = 0.0   # prediction when feature <= threshold
        self.right_value: float = 0.0  # prediction when feature > threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        goes_right = X[:, self.feature_idx] > self.threshold
        return np.where(goes_right, self.right_value, self.left_value)


class SoftRugDetector:
    """
    Gradient-boosted decision-stump classifier for soft-rug early warning.
    Trains from the trade_features table (tokens we traded + their pnl) and
    persists to JSON. Predicts P(rug) from observable launch features.
    """

    def __init__(self) -> None:
        self.cfg = Config.get()
        self.model_path = Path(self.cfg.get_nested(
            "analysis", "soft_rug", default={}).get(
            "model_path", "./data/soft_rug_model.json"))
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = float(self.cfg.get_nested(
            "analysis", "soft_rug", default={}).get("threshold", 0.65))
        self.n_estimators = int(self.cfg.get_nested(
            "analysis", "soft_rug", default={}).get("n_estimators", 50))
        self.learning_rate = float(self.cfg.get_nested(
            "analysis", "soft_rug", default={}).get("learning_rate", 0.1))
        self.min_train_samples = int(self.cfg.get_nested(
            "analysis", "soft_rug", default={}).get("min_samples", 100))
        self.stumps: list[_DecisionStump] = []
        self.init_value: float = 0.5
        self.feature_medians: np.ndarray = np.full(len(FEATURES), 0.0)
        self._loaded = False
        self.load()
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> bool:
        try:
            if self.model_path.exists():
                data = json.loads(self.model_path.read_text())
                self.init_value = data["init_value"]
                self.feature_medians = np.array(data["medians"], dtype=np.float64)
                self.stumps = []
                for s in data["stumps"]:
                    st = _DecisionStump()
                    st.feature_idx = s["feature_idx"]
                    st.threshold = s["threshold"]
                    st.left_value = s["left_value"]
                    st.right_value = s["right_value"]
                    self.stumps.append(st)
                self._loaded = True
                log.info("soft_rug.model_loaded",
                         path=str(self.model_path), n_stumps=len(self.stumps))
                return True
        except Exception as e:
            log.warning("soft_rug.load_failed", error=str(e))
        return False

    def save(self) -> None:
        try:
            data = {
                "init_value": self.init_value,
                "medians": self.feature_medians.tolist(),
                "stumps": [
                    {"feature_idx": s.feature_idx, "threshold": s.threshold,
                     "left_value": s.left_value, "right_value": s.right_value}
                    for s in self.stumps
                ],
            }
            self.model_path.write_text(json.dumps(data))
            log.info("soft_rug.model_saved", path=str(self.model_path),
                     n_stumps=len(self.stumps))
        except Exception as e:
            log.error("soft_rug.save_failed", error=str(e))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, features: RugFeatures) -> RugPrediction:
        if not self.stumps:
            # No trained model yet: heuristic baseline from domain knowledge.
            # High concentration + high sell velocity + low holders = risky.
            p = self._heuristic_proba(features)
            return RugPrediction(
                rug_probability=p,
                is_high_risk=(p >= self.threshold),
                top_risk_factors=self._heuristic_factors(features),
            )
        vec = features.to_vector()
        # Impute missing (NaN) with medians
        mask = np.isnan(vec)
        vec[mask] = self.feature_medians[mask]
        # Run boosting: init + sum(stump corrections)
        raw = self.init_value
        for st in self.stumps:
            raw += self.learning_rate * st.predict(vec.reshape(1, -1))[0]
        proba = 1.0 / (1.0 + math.exp(-raw))
        factors = self._heuristic_factors(features)
        return RugPrediction(
            rug_probability=proba,
            is_high_risk=(proba >= self.threshold),
            top_risk_factors=factors if proba >= self.threshold else [],
        )

    @staticmethod
    def _heuristic_proba(f: RugFeatures) -> float:
        """Domain-knowledge baseline used before enough data to train."""
        p = 0.30  # base rate
        if f.top3_concentration_pct > 50: p += 0.20
        if f.top3_concentration_pct > 80: p += 0.15
        if f.sell_velocity_30s > 10:     p += 0.15
        if f.buy_sell_ratio_30s < 0.5:   p += 0.10
        if f.initial_holder_count < 20:  p += 0.10
        if f.liquidity_usd < 5000:       p += 0.10
        return min(0.98, max(0.02, p))

    @staticmethod
    def _heuristic_factors(f: RugFeatures) -> list[str]:
        factors = []
        if f.top3_concentration_pct > 50:
            factors.append(f"top-3 holders hold {f.top3_concentration_pct:.0f}% of supply")
        if f.sell_velocity_30s > 10:
            factors.append(f"high sell velocity ({f.sell_velocity_30s:.0f}/min in 30s)")
        if f.buy_sell_ratio_30s < 0.5:
            factors.append(f"net selling pressure (b/s ratio {f.buy_sell_ratio_30s:.2f})")
        if f.initial_holder_count < 20:
            factors.append(f"very few early holders ({f.initial_holder_count:.0f})")
        if f.liquidity_usd < 5000:
            factors.append(f"thin liquidity (${f.liquidity_usd:.0f})")
        return factors

    # ------------------------------------------------------------------
    # Training (gradient boosting with decision stumps)
    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fit gradient-boosted stumps. X: (n, 8) feature matrix, y: (n,) 0/1 labels.
        Returns training metrics.
        """
        n, d = X.shape
        if n < 2:
            raise ValueError("need at least 2 samples to train")
        # Impute NaN with column medians
        self.feature_medians = np.nanmedian(X, axis=0)
        # If a column is all-NaN, median is NaN -> replace with 0
        nan_medians = np.isnan(self.feature_medians)
        self.feature_medians[nan_medians] = 0.0
        Xc = np.where(np.isnan(X), self.feature_medians, X)

        # Init: log-odds of the positive rate
        pos_rate = max(1e-4, min(1 - 1e-4, y.mean()))
        self.init_value = math.log(pos_rate / (1 - pos_rate))
        self.stumps = []

        # Working residual predictions
        raw = np.full(n, self.init_value)
        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-raw))
            # Gradient of log-loss w.r.t. raw: (p - y)
            grad = p - y
            # Fit a stump to the negative gradient
            stump = self._fit_stump(Xc, -grad)
            if stump is None:
                break
            self.stumps.append(stump)
            raw += self.learning_rate * stump.predict(Xc)

        # Training accuracy
        p_final = 1.0 / (1.0 + np.exp(-raw))
        preds = (p_final >= 0.5).astype(int)
        accuracy = float((preds == y).mean())
        metrics = {
            "n_samples": n, "n_stumps": len(self.stumps),
            "train_accuracy": accuracy, "positive_rate": float(y.mean()),
        }
        self.save()
        log.info("soft_rug.trained", **metrics)
        return metrics

    def _fit_stump(self, X: np.ndarray, residual: np.ndarray) -> Optional[_DecisionStump]:
        """
        Find the best single-feature threshold split that most reduces squared
        error on `residual`. Returns the best stump or None if no split helps.
        """
        n, d = X.shape
        best_stump: Optional[_DecisionStump] = None
        best_gain = 1e-9  # only keep splits that improve
        parent_mean = residual.mean()
        parent_sq = ((residual - parent_mean) ** 2).sum()

        for feat in range(d):
            col = X[:, feat]
            # Candidate thresholds: unique values (capped for speed)
            uniq = np.unique(col)
            if len(uniq) > 50:
                uniq = np.quantile(col, np.linspace(0.05, 0.95, 30))
            for thr in uniq:
                left_mask = col <= thr
                n_left = int(left_mask.sum())
                n_right = n - n_left
                if n_left < 2 or n_right < 2:
                    continue
                left_mean = residual[left_mask].mean()
                right_mean = residual[~left_mask].mean()
                left_sq = ((residual[left_mask] - left_mean) ** 2).sum()
                right_sq = ((residual[~left_mask] - right_mean) ** 2).sum()
                gain = parent_sq - left_sq - right_sq
                if gain > best_gain:
                    best_gain = gain
                    s = _DecisionStump()
                    s.feature_idx = feat
                    s.threshold = float(thr)
                    s.left_value = float(left_mean)
                    s.right_value = float(right_mean)
                    best_stump = s
        return best_stump

    # ------------------------------------------------------------------
    # Nightly retrain job (from trade_features)
    # ------------------------------------------------------------------
    async def start_nightly_retrain(self) -> None:
        self._task = asyncio.create_task(self._retrain_loop())
        log.info("soft_rug.retrain_loop_started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _retrain_loop(self) -> None:
        stop = asyncio.Event()
        # Warmup 10 min, then nightly at 03:00 UTC
        try:
            await asyncio.wait_for(stop.wait(), timeout=600)
            return
        except asyncio.TimeoutError:
            pass
        await self.retrain_from_db()
        while not stop.is_set():
            from datetime import datetime, timedelta, timezone
            now = datetime.now(timezone.utc)
            target = now.replace(hour=3, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            delay = (target - now).total_seconds()
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                pass
            await self.retrain_from_db()

    async def retrain_from_db(self) -> Optional[dict]:
        """
        Build a training set from trade_features:
          - rug=1 if a closed trade's pnl_pct <= -50% (we got soft-rugged)
          - rug=0 otherwise
        Features are reconstructed from the recorded component scores
        (liquidity/token_quality correlate with rug risk) + age. This is a
        best-effort proxy; a richer feature capture would come from the
        analytics pipeline. Returns training metrics or None if too few rows.
        """
        try:
            db = Persistence.get()
            rows = db.load_closed_trade_features(since_ts=0.0, limit=20000)
            if len(rows) < self.min_train_samples:
                log.info("soft_rug.retrain_skipped", n=len(rows),
                         min=self.min_train_samples)
                return None
            X = np.zeros((len(rows), len(FEATURES)), dtype=np.float64)
            y = np.zeros(len(rows), dtype=np.float64)
            for i, r in enumerate(rows):
                # Proxy features from what we recorded (order_flow, liquidity,
                # token_quality). Holder/concentration/velocity would need the
                # analytics pipeline to persist them; we use what we have.
                X[i, 0] = 0.0  # initial_holder_count unknown historically
                X[i, 1] = 100.0 - (r.get("token_quality_score") or 50.0)
                X[i, 2] = 0.0  # dev_retained_pct unknown
                # sell_velocity proxy: inverse of buy/sell ratio from order_flow
                of = r.get("order_flow_score") or 50.0
                X[i, 3] = max(0.0, 100.0 - of)
                X[i, 4] = of / 50.0
                X[i, 5] = (r.get("liquidity_score") or 50.0) * 100.0
                X[i, 6] = (r.get("bonding_curve_score") or 50.0)
                X[i, 7] = 60.0
                pnl = r.get("pnl_pct")
                y[i] = 1.0 if (pnl is not None and pnl <= -50.0) else 0.0
            metrics = self.train(X, y)
            return metrics
        except Exception as e:
            log.error("soft_rug.retrain_failed", error=str(e))
            return None
