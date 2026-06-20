"""
liquidity_depth.py
==================
Liquidity depth analyzer for AMM pools.

For pump.fun bonding curve tokens: uses BondingCurveAnalyzer math.
For migrated tokens (Raydium/Uniswap V3): queries pool reserves + computes
slippage curve for various trade sizes.

Output: LiquidityDepth object with:
- Effective slippage at 0.01, 0.05, 0.1, 0.5, 1.0 SOL trade sizes
- Max profitable trade size (largest buy that doesn't move price >2%)
- Liquidity depth ratio (liquidity_usd / market_cap_usd)
- Recommended max position size for this token
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.bonding_curve import BondingCurveAnalyzer, BondingCurveState
from utils.config_loader import Config
from utils.data_providers import get_providers
from utils.logger import setup_logger

log = setup_logger("liquidity_depth")


@dataclass
class LiquidityDepth:
    mint: str
    chain: str
    pool_type: str                    # "bonding_curve" | "raydium_v4" | "uniswap_v3"
    liquidity_usd: float
    liquidity_sol: float
    market_cap_usd: float
    liquidity_to_mcap_ratio: float    # >0.3 = healthy, <0.1 = risky
    slippage_curve: dict[float, float]   # trade_size_sol -> slippage_pct
    max_profitable_size_sol: float    # largest size with <2% slippage
    recommended_max_position_sol: float  # 1% of liquidity, capped at 5 SOL
    notes: list[str] = field(default_factory=list)


class LiquidityDepthAnalyzer:
    """Computes liquidity depth profile for any token."""

    # Trade sizes to test for slippage curve (in SOL)
    TEST_SIZES_SOL = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    MAX_ACCEPTABLE_SLIPPAGE_PCT = 2.0

    def __init__(self) -> None:
        self.cfg = Config.get()
        self.bonding = BondingCurveAnalyzer()

    async def analyze(self, mint: str, chain: str = "solana") -> Optional[LiquidityDepth]:
        """Compute liquidity depth for a token."""
        providers = get_providers()

        # Try bonding curve first (pump.fun)
        bc_state = await self.bonding.fetch_state(mint)

        if bc_state and bc_state.completion_pct < 100:
            return await self._analyze_bonding_curve(mint, chain, bc_state, providers)

        # Fall back to DEX (Raydium / Uniswap)
        return await self._analyze_dex_pool(mint, chain, providers)

    async def _analyze_bonding_curve(
        self, mint: str, chain: str, bc: BondingCurveState, providers
    ) -> LiquidityDepth:
        """Analyze pump.fun bonding curve liquidity depth."""
        slippage_curve = {}
        max_profitable = 0.001

        for size_sol in self.TEST_SIZES_SOL:
            try:
                _, _, slip_pct = self.bonding.compute_real_slippage(bc, size_sol)
                slippage_curve[size_sol] = slip_pct
                if slip_pct <= self.MAX_ACCEPTABLE_SLIPPAGE_PCT:
                    max_profitable = size_sol
            except Exception:
                slippage_curve[size_sol] = -1

        # Market cap = current price * total supply
        # For pump.fun: virtual_token_reserves + real_token_reserves ~ 1B tokens
        total_supply = bc.virtual_token_reserves + bc.real_token_reserves
        market_cap_usd = bc.current_price_usd * total_supply
        liquidity_usd = bc.real_sol_reserves * 200  # rough: 1 SOL ≈ $200
        liquidity_sol = bc.real_sol_reserves

        # Recommended max position: 1% of liquidity, capped at 5 SOL
        rec_max = min(5.0, max(0.05, liquidity_sol * 0.01))

        notes = []
        if bc.completion_pct < 30:
            notes.append("Bonding curve < 30% — very early, high risk")
        elif bc.completion_pct > 90:
            notes.append("Bonding curve > 90% — migration imminent")

        return LiquidityDepth(
            mint=mint, chain=chain, pool_type="bonding_curve",
            liquidity_usd=liquidity_usd, liquidity_sol=liquidity_sol,
            market_cap_usd=market_cap_usd,
            liquidity_to_mcap_ratio=liquidity_usd / max(market_cap_usd, 1),
            slippage_curve=slippage_curve,
            max_profitable_size_sol=max_profitable,
            recommended_max_position_sol=rec_max,
            notes=notes,
        )

    async def _analyze_dex_pool(self, mint: str, chain: str, providers) -> Optional[LiquidityDepth]:
        """Analyze Raydium/Uniswap pool depth via DexScreener."""
        try:
            pair = await providers["dexscreener"].get_pair(mint)
            if not pair:
                return None
            liq_usd = float((pair.get("liquidity") or {}).get("usd", 0))
            liq_sol = liq_usd / 200  # rough
            market_cap_usd = float(pair.get("fdv") or pair.get("marketCap") or 0)

            # For DEX pools, slippage ≈ sqrt(size / liquidity) for constant product
            # Without on-chain reserves we estimate via sqrt formula
            slippage_curve = {}
            max_profitable = 0.001
            for size_sol in self.TEST_SIZES_SOL:
                if liq_sol > 0:
                    # x*y=k model: slippage ≈ 2 * size / liq (in linear approximation)
                    slip_pct = 2 * size_sol / liq_sol * 100
                else:
                    slip_pct = 100
                slippage_curve[size_sol] = slip_pct
                if slip_pct <= self.MAX_ACCEPTABLE_SLIPPAGE_PCT:
                    max_profitable = size_sol

            rec_max = min(5.0, max(0.05, liq_sol * 0.01))

            pool_type = "raydium_v4" if chain == "solana" else "uniswap_v3"
            notes = []
            if liq_usd < 10000:
                notes.append("Very low DEX liquidity — high slippage risk")
            ratio = liq_usd / max(market_cap_usd, 1)
            if ratio < 0.05:
                notes.append("Liquidity << market cap — exit may be hard")

            return LiquidityDepth(
                mint=mint, chain=chain, pool_type=pool_type,
                liquidity_usd=liq_usd, liquidity_sol=liq_sol,
                market_cap_usd=market_cap_usd,
                liquidity_to_mcap_ratio=ratio,
                slippage_curve=slippage_curve,
                max_profitable_size_sol=max_profitable,
                recommended_max_position_sol=rec_max,
                notes=notes,
            )
        except Exception as e:
            log.warning("liquidity_depth.dex_failed", mint=mint, error=str(e))
            return None
