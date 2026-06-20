"""
mev_detector.py
===============
MEV / sandwich attack detector.

After each buy tx we execute, analyze:
- Was the tx sandwiched? (front-run buy + back-run sell by same attacker)
- Did the price move >X% between our submission and confirmation?

If we detect sandwiching, we:
- Flag the token as "high-MEV risk"
- Increase slippage tolerance for future buys OR
- Force Jito bundles for all subsequent buys on this token

Detection signals:
1. Same wallet bought within 1-2 slots before us AND sold within 1-2 slots after.
2. Our executed price > quote price by > slippage tolerance.
3. Multiple consecutive buys on the same token by the same wallet in same block
   (bot pattern).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from utils.config_loader import Config
from utils.data_providers import get_providers
from utils.logger import setup_logger

log = setup_logger("mev_detector")


@dataclass
class MevAttack:
    our_signature: str
    token: str
    attacker_wallet: Optional[str]
    front_run_signature: Optional[str]
    back_run_signature: Optional[str]
    price_impact_pct: float           # how much our execution deviated from quote
    sandwich_profit_sol: Optional[float]
    detected_at: float = field(default_factory=time.time)


class MevDetector:
    """Detects sandwich attacks on our executed transactions."""

    # If our execution price exceeds quote by this %, likely sandwiched
    SANDWICH_THRESHOLD_PCT = 5.0
    # Max slots to look back/forward for attacker pattern
    SLOT_WINDOW = 3

    def __init__(self) -> None:
        self.cfg = Config.get()
        self._attacks: list[MevAttack] = []
        self._high_mev_tokens: dict[str, float] = {}  # token -> first detection ts

    async def analyze_buy(
        self,
        our_signature: str,
        token: str,
        expected_price: float,
        executed_price: float,
    ) -> Optional[MevAttack]:
        """
        Check if our buy was sandwiched. Returns MevAttack if detected.
        """
        if expected_price <= 0:
            return None

        price_impact = ((executed_price - expected_price) / expected_price) * 100

        # Quick check: if price impact < threshold, probably not sandwiched
        if abs(price_impact) < self.SANDWICH_THRESHOLD_PCT:
            return None

        # Deep check: fetch tx and look for attacker pattern
        try:
            providers = get_providers()
            helius = providers["helius"]

            # Get our transaction
            our_tx_resp = await helius._rpc("getTransaction", [
                our_signature, {"maxSupportedTransactionVersion": 0, "encoding": "jsonParsed"}
            ])
            our_tx = our_tx_resp.get("result")
            if not our_tx:
                return None

            our_slot = our_tx.get("slot", 0)

            # Get transactions in slots around ours for this token
            # Look at recent signatures for the token
            sigs_resp = await helius._rpc("getSignaturesForAddress", [
                token, {"limit": 50}
            ])
            all_sigs = sigs_resp.get("result", [])

            # Find signatures in our slot window
            nearby_sigs = []
            for sig_entry in all_sigs:
                sig = sig_entry.get("signature")
                if sig == our_signature:
                    continue
                # Fetch slot for this sig
                tx_resp = await helius._rpc("getTransaction", [
                    sig, {"maxSupportedTransactionVersion": 0, "encoding": "jsonParsed"}
                ])
                tx = tx_resp.get("result")
                if not tx:
                    continue
                slot = tx.get("slot", 0)
                if abs(slot - our_slot) <= self.SLOT_WINDOW:
                    nearby_sigs.append((sig, tx, slot))

            # Look for sandwich pattern: a wallet that bought in slots before us
            # and sold in slots after us
            attacker = None
            front_sig = None
            back_sig = None
            for sig, tx, slot in nearby_sigs:
                # Get signer of this tx
                try:
                    signer = tx["transaction"]["message"]["accountKeys"][0]
                except (KeyError, IndexError):
                    continue
                # Determine if buy or sell (from logs)
                logs = (tx.get("meta") or {}).get("logMessages", []) or []
                is_buy = any("Buy" in l for l in logs)
                is_sell = any("Sell" in l for l in logs)
                if slot < our_slot and is_buy and signer not in [a for a in [attacker]]:
                    attacker = signer
                    front_sig = sig
                elif slot > our_slot and is_sell and signer == attacker:
                    back_sig = sig
                    break

            if attacker and front_sig:
                attack = MevAttack(
                    our_signature=our_signature,
                    token=token,
                    attacker_wallet=attacker,
                    front_run_signature=front_sig,
                    back_run_signature=back_sig,
                    price_impact_pct=price_impact,
                    sandwich_profit_sol=None,  # would need to compute from tx values
                )
                self._attacks.append(attack)
                self._high_mev_tokens[token] = time.time()
                log.warning("mev.SANDWICH_DETECTED",
                            token=token, attacker=attacker[:8],
                            price_impact=price_impact,
                            our_sig=our_signature[:8])
                return attack

        except Exception as e:
            log.warning("mev.analysis_failed", error=str(e))

        return None

    def is_high_mev_token(self, token: str) -> bool:
        """True if this token has had a recent sandwich attack (within 1 hour)."""
        ts = self._high_mev_tokens.get(token)
        return ts is not None and time.time() - ts < 3600

    def get_recent_attacks(self, limit: int = 20) -> list[MevAttack]:
        return self._attacks[-limit:]

    def get_stats(self) -> dict:
        return {
            "total_attacks_detected": len(self._attacks),
            "high_mev_tokens_active": sum(1 for ts in self._high_mev_tokens.values()
                                          if time.time() - ts < 3600),
        }
