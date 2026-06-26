"""
crowd_weight_router.py
======================
Dynamic weight reconfiguration driven by the Crowd Positioning Engine.

WHY THIS EXISTS (the "Régime C" from the crowd engine design):
When the crowd is at an extreme (euphoric-long or panic-short), trend-following
signals become actively HARMFUL — they point the same direction the crowd is
already overcrowded, which is exactly the wrong trade. The crowd engine fades
it via ContrarianAgent, but the OTHER strategies (momentum, sniping) would
still pile in behind the trend. This router temporarily deflates their
influence on the composite AlphaSignal so the swarm doesn't fight itself.

MECHANISM:
The router exposes a method `active_multipliers()` that returns a per-component
multiplier (0..1). When a crowd extreme is active for a token:
  - trend-following components (order_flow, lifecycle, sentiment) get their
    weight multiplied by (1 - crowd_strength), down to a floor (e.g. 0.1).
  - contrarian-friendly components (mev_penalty, token_quality) are untouched.
  - the multiplier reverts to 1.0 automatically after a TTL (no permanent change).

This is deliberately token-scoped and time-bounded — it's a tactical dampener,
not a strategy change. The learned weights (from the AutoTuner) are the
strategic baseline; this is an overlay.

INTEGRATION:
- The orchestrator's _crowd_monitor calls router.apply(event) when it sees an extreme.
- AlphaSignalGenerator.generate() consults router.active_multipliers(mint) and
  scales its weights before computing the composite score.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import setup_logger

log = setup_logger("crowd_router")

# Default TTL: a crowd extreme dampener stays active for 10 minutes, then the
# weights revert to their learned values. Crowd extremes are acute; holding the
# dampener forever would make the bot structurally contrarian.
_DEFAULT_TTL_SEC = 600.0

# These components track the crowd's direction (trend-following). When the
# crowd is overcrowded, following the trend = following the crowd into the trap.
# We dampen them. The others (token_quality = rug check, mev_penalty = MEV
# defense, liquidity = exit-ability, bonding_curve = catalyst) are independent
# of crowd direction and stay at full weight.
_TREND_COMPONENTS = {"order_flow", "lifecycle", "sentiment"}

# Floor multiplier for trend components during a crowd extreme. We never zero
# them completely — a strong trend signal still warrants *some* weight, just
# not enough to override the contrarian view.
_TREND_FLOOR = 0.15


@dataclass
class _Dampening:
    """Active dampening for one token."""
    token: str
    crowd_score: float          # signed: the extreme that triggered this
    conviction: float
    started_at: float
    expires_at: float
    # Per-component multiplier (1.0 = no change, <1 = dampened)
    multipliers: dict[str, float] = field(default_factory=dict)


class CrowdWeightRouter:
    """
    Holds active crowd-driven weight dampenings and exposes per-token
    multipliers that AlphaSignalGenerator applies to its weights.
    """

    def __init__(self, ttl_sec: float = _DEFAULT_TTL_SEC) -> None:
        self.ttl_sec = ttl_sec
        # token -> _Dampening (only active/extreme tokens are present)
        self._active: dict[str, _Dampening] = {}

    def apply(self, token: str, crowd_score: float, conviction: float) -> None:
        """
        Register or refresh a crowd extreme for `token`. Computes the per-
        component dampening multipliers based on how extreme the crowd is.

        crowd_score: signed [-1,+1] from CrowdPositioningEvent.
        conviction: 0..1 from CrowdPositioningEvent.
        """
        # Strength of the dampening: 0 at the threshold, 1 at max extreme.
        strength = min(1.0, abs(crowd_score) * conviction)
        if strength < 0.15:
            # Not strong enough to warrant dampening.
            self._active.pop(token, None)
            return

        now = time.time()
        multipliers: dict[str, float] = {}
        for comp in _TREND_COMPONENTS:
            # Linear interpolation: full weight (1.0) at strength=0, floor at strength=1.
            multipliers[comp] = max(_TREND_FLOOR, 1.0 - strength * (1.0 - _TREND_FLOOR))

        self._active[token] = _Dampening(
            token=token,
            crowd_score=crowd_score,
            conviction=conviction,
            started_at=now,
            expires_at=now + self.ttl_sec,
            multipliers=multipliers,
        )
        log.info("crowd_router.dampening_applied",
                 token=token[:8], crowd_score=round(crowd_score, 2),
                 conviction=round(conviction, 2), strength=round(strength, 2),
                 multipliers={k: round(v, 2) for k, v in multipliers.items()},
                 ttl_sec=self.ttl_sec)

    def clear(self, token: str) -> None:
        """Remove the dampening for a token (e.g. position closed)."""
        if token in self._active:
            self._active.pop(token)
            log.info("crowd_router.dampening_cleared", token=token[:8])

    def _purge_expired(self) -> None:
        """Remove dampenings whose TTL has elapsed."""
        now = time.time()
        expired = [t for t, d in self._active.items() if now >= d.expires_at]
        for t in expired:
            self._active.pop(t, None)
            log.info("crowd_router.dampening_expired", token=t[:8])

    def active_multipliers(self, token: str) -> dict[str, float]:
        """
        Return the per-component weight multipliers for `token`.
        Returns an empty dict (meaning "no change") if no active dampening.
        The caller multiplies each component's weight by its multiplier.
        """
        self._purge_expired()
        damp = self._active.get(token)
        if damp is None:
            return {}
        return dict(damp.multipliers)

    def is_dampened(self, token: str) -> bool:
        """Quick check: is this token currently under crowd dampening?"""
        self._purge_expired()
        return token in self._active

    def status(self) -> list[dict]:
        """Snapshot of all active dampenings (for dashboard / debugging)."""
        self._purge_expired()
        now = time.time()
        return [
            {
                "token": d.token,
                "crowd_score": d.crowd_score,
                "conviction": d.conviction,
                "multipliers": d.multipliers,
                "remaining_sec": round(d.expires_at - now, 0),
            }
            for d in self._active.values()
        ]
