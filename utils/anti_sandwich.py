"""
anti_sandwich.py
================
Dynamic slippage calculator to minimize sandwich attack losses.

A sandwich attack works as follows:
1. Attacker sees your pending buy tx in the mempool.
2. Attacker front-runs with a buy, pushing the price up.
3. Your buy executes at the inflated price.
4. Attacker sells immediately after, profiting from your slippage tolerance.

Mitigations:
1. Use Jito bundles (already implemented) — hides tx from public mempool.
2. Use tight slippage that adapts to trade size vs pool liquidity.
3. For non-Jito routes, compute the minimum slippage that still allows the
   trade to land given current pool depth.

This module computes slippage based on:
- Trade size as % of pool liquidity (smaller = tighter slippage OK)
- Recent volatility of the token (higher vol = need more headroom)
- Priority fee (high fee = tx lands faster = less exposure window)
"""
from __future__ import annotations

from dataclasses import dataclass

from utils.bonding_curve import BondingCurveAnalyzer, BondingCurveState
from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("anti_sandwich")


@dataclass
class SlippageRecommendation:
    slippage_bps: int
    reason: str
    use_jito: bool


class AntiSandwich:
    """Computes optimal slippage tolerance per trade."""

    def __init__(self) -> None:
        self.cfg = Config.get()
        self.bonding = BondingCurveAnalyzer()
        ascfg = self.cfg.get_nested("anti_sandwich", default={})
        self.base_slippage_bps = ascfg.get("base_slippage_bps", 300)        # 3% default
        self.max_slippage_bps = ascfg.get("max_slippage_bps", 1500)         # 15% cap
        self.min_slippage_bps = ascfg.get("min_slippage_bps", 50)           # 0.5% floor
        self.size_to_liquidity_threshold = ascfg.get("size_liq_threshold", 0.02)  # 2% of liq
        self.jito_threshold_sol = ascfg.get("jito_threshold_sol", 0.1)

    def compute(
        self,
        trade_size_sol: float,
        liquidity_sol: float,
        bc_state: BondingCurveState = None,
        urgent: bool = False,
    ) -> SlippageRecommendation:
        """
        Compute recommended slippage for a buy.
        - trade_size_sol: how much SOL we're spending
        - liquidity_sol: pool liquidity in SOL (use bc_state.real_sol_reserves for pump.fun)
        - bc_state: optional bonding curve state for precise math
        - urgent: True for sniping / SL exit (use Jito, tighter slippage OK)
        """
        # 1. Decide if we should use Jito
        use_jito = trade_size_sol >= self.jito_threshold_sol

        # 2. Compute slippage from trade size vs liquidity
        if liquidity_sol <= 0:
            size_ratio = 1.0
        else:
            size_ratio = trade_size_sol / liquidity_sol

        if size_ratio < 0.005:
            # Tiny trade: 0.5% slippage is enough
            slippage = self.min_slippage_bps
            reason = f"Tiny trade (size/liq={size_ratio:.3%})"
        elif size_ratio < self.size_to_liquidity_threshold:
            # Small trade: scale linearly from min to base
            slippage = int(self.min_slippage_bps + (
                self.base_slippage_bps - self.min_slippage_bps
            ) * (size_ratio / self.size_to_liquidity_threshold))
            reason = f"Small trade (size/liq={size_ratio:.3%})"
        elif size_ratio < 0.10:
            # Medium trade: scale from base to max
            slippage = int(self.base_slippage_bps + (
                self.max_slippage_bps - self.base_slippage_bps
            ) * ((size_ratio - self.size_to_liquidity_threshold) /
                 (0.10 - self.size_to_liquidity_threshold)))
            reason = f"Medium trade (size/liq={size_ratio:.3%})"
        else:
            # Large trade relative to pool — high slippage unavoidable
            slippage = self.max_slippage_bps
            reason = f"Large trade (size/liq={size_ratio:.3%})"

        # 3. Use precise bonding curve math if available
        if bc_state:
            try:
                _, _, real_slippage_pct = self.bonding.compute_real_slippage(bc_state, trade_size_sol)
                # Convert % to bps, add 50bps headroom
                precise_bps = int(real_slippage_pct * 100) + 50
                slippage = max(slippage, precise_bps)
                reason += f", real BC slippage={real_slippage_pct:.2f}%"
            except Exception:
                pass

        # 4. Urgent trades: 2x tighter (we'd rather miss than overpay)
        if urgent:
            slippage = max(self.min_slippage_bps, slippage // 2)
            reason += ", urgent (halved)"

        # 5. Clamp
        slippage = max(self.min_slippage_bps, min(self.max_slippage_bps, slippage))

        return SlippageRecommendation(
            slippage_bps=slippage,
            reason=reason,
            use_jito=use_jito,
        )
