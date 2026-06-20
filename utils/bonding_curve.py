"""
bonding_curve.py
================
Real pump.fun bonding curve math.

pump.fun uses a virtual reserves model:
    virtual_sol_reserves  = 30 SOL
    virtual_token_reserves = 1_073_000_000 tokens (approx, depends on token decimals)
    REAL SOL RAISED at any moment = current bonding curve SOL balance
    REAL TOKENS REMAINING = current bonding curve token balance

Price formula (constant product AMM):
    price_sol_per_token = virtual_sol_reserves / virtual_token_reserves
    (NOTE: pump.fun adds a small fee on top, ~1% per trade)

Once the bonding curve accumulates ~79-85 SOL of real reserves, pump.fun
migrates the token to Raydium (burns the bonding curve, creates a real LP).
This migration is a MASSIVE catalyst — tokens typically 2-10x within minutes
of migration as real liquidity enables wider market access.

This module:
- Fetches real-time bonding curve state via Helius getTokenAccountBalance.
- Computes true price, slippage for a given trade size.
- Estimates % completion (how close to migration).
- Estimates ETA to migration (based on velocity of SOL inflow).
- Detects imminent migration (>= 95% complete) — high-value signal.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from utils.config_loader import Config
from utils.data_providers import get_providers
from utils.logger import setup_logger
from utils.pumpfun_parser import PUMP_FUN_PROGRAM

log = setup_logger("bonding_curve")

# Pump.fun constants (from on-chain program)
VIRTUAL_SOL_RESERVES = 30.0           # SOL
VIRTUAL_TOKEN_RESERVES = 1_073_000_000.0  # tokens (before any real buys)
REAL_SOL_RESERVES_AT_MIGRATION = 85.0  # SOL — pump.fun migrates around this threshold
PUMP_FEE_BPS = 100                     # 1% protocol fee
FEE_RECIPIENT_BPS = 50                 # 0.5% to fee recipient (if any)

# Cache SOL/USD price (refreshed every 60s) to avoid hammering Jupiter Price API.
_SOL_USD_CACHE: dict[str, float] = {"price": 0.0, "ts": 0.0}


async def fetch_sol_usd() -> float:
    """Fetch SOL/USD from Jupiter Price API V3. Cached for 60 seconds."""
    now = time.time()
    if _SOL_USD_CACHE["price"] > 0 and now - _SOL_USD_CACHE["ts"] < 60:
        return _SOL_USD_CACHE["price"]
    import aiohttp
    url = "https://price.jup.ag/v6/price?ids=So11111111111111111111111111111111111111112"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(url) as r:
                data = await r.json()
        price = float(data["data"]["So11111111111111111111111111111111111111112"]["price"])
        _SOL_USD_CACHE["price"] = price
        _SOL_USD_CACHE["ts"] = now
        return price
    except Exception as e:
        log.warning("bonding_curve.sol_usd_fetch_failed", error=str(e))
        # Fallback to last known price, else a conservative estimate
        return _SOL_USD_CACHE["price"] if _SOL_USD_CACHE["price"] > 0 else 150.0


@dataclass
class BondingCurveState:
    mint: str
    bonding_curve_address: str
    real_sol_reserves: float           # SOL actually deposited in bonding curve
    real_token_reserves: float         # tokens remaining for sale
    virtual_sol_reserves: float        # always 30 SOL (constant)
    virtual_token_reserves: float      # constant
    current_price_sol: float           # SOL per token (real)
    current_price_usd: float           # USD per token
    completion_pct: float              # 0..100 (% to migration)
    sol_needed_to_migrate: float       # SOL still needed
    migration_eta_sec: Optional[float]  # estimated seconds to migration (None if velocity unknown)
    last_sol_inflow_rate: Optional[float]  # SOL/sec over last 60s
    fetched_at: float                  # timestamp


class BondingCurveAnalyzer:
    """Computes real-time bonding curve state for a pump.fun token."""

    def __init__(self) -> None:
        self.cfg = Config.get()
        self._inflow_history: dict[str, list[tuple[float, float]]] = {}  # mint -> [(ts, sol_reserves), ...]

    async def fetch_state(self, mint: str, bonding_curve_address: Optional[str] = None) -> Optional[BondingCurveState]:
        """
        Fetch current bonding curve state.
        If bonding_curve_address is unknown, derive it from the mint using PDA.
        """
        providers = get_providers()
        helius = providers["helius"]
        dex = providers["dexscreener"]

        try:
            # Derive bonding curve PDA if not given
            # PDA seeds: ["bonding-curve", mint_pubkey]
            if not bonding_curve_address:
                bonding_curve_address = self._derive_bonding_curve_pda(mint)
                if not bonding_curve_address:
                    return None

            # Fetch bonding curve account data (parsed)
            # pump.fun bonding curve account layout:
            #   discriminator (8 bytes)
            #   virtual_token_reserves (u64)
            #   virtual_sol_reserves (u64)
            #   real_token_reserves (u64)
            #   real_sol_reserves (u64)
            #   token_total_supply (u64)
            #   complete (bool)
            account_info = await helius.get_account_info(bonding_curve_address)
            if not account_info:
                return None

            # Try parsed first; fall back to raw base64 decode
            real_sol_reserves, real_token_reserves, virtual_sol, virtual_token, complete = (
                self._decode_bonding_curve_account(account_info)
            )
            if real_sol_reserves is None:
                return None

            # USD price from DexScreener (if listed) + live SOL/USD from Jupiter
            price_usd = await dex.get_price_usd(mint) if complete else 0.0
            sol_usd = await fetch_sol_usd()
            current_price_sol = (virtual_sol / virtual_token) if virtual_token > 0 else 0.0
            if price_usd == 0.0:
                price_usd = current_price_sol * sol_usd

            # Completion %
            completion_pct = min(100.0, (real_sol_reserves / REAL_SOL_RESERVES_AT_MIGRATION) * 100.0)
            sol_needed = max(0.0, REAL_SOL_RESERVES_AT_MIGRATION - real_sol_reserves)

            # Track inflow history for ETA estimation
            now = time.time()
            self._inflow_history.setdefault(mint, []).append((now, real_sol_reserves))
            # Keep only last 5 minutes
            self._inflow_history[mint] = [
                (ts, r) for ts, r in self._inflow_history[mint] if now - ts < 300
            ]
            inflow_rate = self._estimate_inflow_rate(mint)
            eta_sec = (sol_needed / inflow_rate) if inflow_rate and inflow_rate > 0 else None

            return BondingCurveState(
                mint=mint,
                bonding_curve_address=bonding_curve_address,
                real_sol_reserves=real_sol_reserves,
                real_token_reserves=real_token_reserves,
                virtual_sol_reserves=virtual_sol,
                virtual_token_reserves=virtual_token,
                current_price_sol=current_price_sol,
                current_price_usd=price_usd,
                completion_pct=completion_pct,
                sol_needed_to_migrate=sol_needed,
                migration_eta_sec=eta_sec,
                last_sol_inflow_rate=inflow_rate,
                fetched_at=now,
            )

        except Exception as e:
            log.warning("bonding_curve.fetch_failed", mint=mint, error=str(e))
            return None

    # ------------------------------------------------------------------
    def _derive_bonding_curve_pda(self, mint: str) -> Optional[str]:
        """Derive the pump.fun bonding curve PDA for a given mint."""
        try:
            from solders.pubkey import Pubkey
            seeds = [b"bonding-curve", bytes(Pubkey.from_string(mint))]
            pda, _ = Pubkey.find_program_address(seeds, Pubkey.from_string(PUMP_FUN_PROGRAM))
            return str(pda)
        except Exception as e:
            log.warning("bonding_curve.pda_derive_failed", mint=mint, error=str(e))
            return None

    def _decode_bonding_curve_account(self, account_info: dict) -> tuple:
        """
        Decode the pump.fun bonding curve account.
        Layout (Anchor):
            discriminator (8 bytes)
            virtual_token_reserves (u64)
            virtual_sol_reserves (u64)
            real_token_reserves (u64)
            real_sol_reserves (u64)
            token_total_supply (u64)
            complete (bool, 1 byte)

        Handles Helius getAccountInfo response shapes for encoding=base64:
            {'value': {'data': ['<b64>', 'base64'], ...}}   <- LIST form (real)
            {'value': {'data': [b64, 'base64+zstd'], ...}}  <- zstd-compressed
        """
        try:
            import base64
            data_field = account_info.get("data")
            raw: Optional[bytes] = None
            if isinstance(data_field, list) and len(data_field) >= 2:
                b64_str, encoding = data_field[0], data_field[1]
                if encoding == "base64":
                    raw = base64.b64decode(b64_str)
                elif encoding == "base64+zstd":
                    try:
                        import zstandard  # type: ignore
                        raw = zstandard.decompress(base64.b64decode(b64_str))
                    except ImportError:
                        log.warning("bonding_curve.zstd_unavailable")
                        return (None,) * 5
            elif isinstance(data_field, dict):
                # Some RPCs return {'data': {'parsed': ...}} or [b64, enc]
                inner = data_field.get("parsed", {})
                if isinstance(inner, dict) and "info" in inner:
                    # jsonParsed SPL token form — not a bonding curve, bail
                    return (None,) * 5

            if raw is None or len(raw) < 8 + 5 * 8 + 1:
                return (None,) * 5

            offset = 8  # skip discriminator
            vtr = int.from_bytes(raw[offset:offset + 8], "little"); offset += 8
            vsr = int.from_bytes(raw[offset:offset + 8], "little"); offset += 8
            rtr = int.from_bytes(raw[offset:offset + 8], "little"); offset += 8
            rsr = int.from_bytes(raw[offset:offset + 8], "little"); offset += 8
            supply = int.from_bytes(raw[offset:offset + 8], "little"); offset += 8
            complete = bool(raw[offset])
            # SOL reserves are in lamports; token reserves are in raw token units
            # (pump.fun tokens are 6 decimals, but we return raw counts here and
            # the consumer divides by 1e9 only for SOL. Token decimals handled by
            # callers that need human amounts.)
            return (rsr / 1e9, rtr / 1e6, vsr / 1e9, vtr / 1e6, complete)
        except Exception as e:
            log.warning("bonding_curve.decode_failed", error=str(e))
            return (None,) * 5

    def _estimate_inflow_rate(self, mint: str) -> Optional[float]:
        """Estimate SOL/sec inflow rate over the last minute."""
        history = self._inflow_history.get(mint, [])
        if len(history) < 2:
            return None
        # Use last 60 seconds of data
        now = time.time()
        recent = [(ts, r) for ts, r in history if now - ts < 60]
        if len(recent) < 2:
            return None
        dt = recent[-1][0] - recent[0][0]
        dr = recent[-1][1] - recent[0][1]
        return dr / dt if dt > 0 else None

    # ------------------------------------------------------------------
    # Slippage math
    # ------------------------------------------------------------------
    def compute_real_slippage(
        self, state: BondingCurveState, buy_size_sol: float
    ) -> tuple[float, float, float]:
        """
        Compute true slippage for a buy on the bonding curve.

        Returns (effective_price, expected_tokens, slippage_pct).
        """
        # Constant-product AMM: x * y = k
        # virtual reserves are constant; real reserves evolve with trades.
        # Effective reserves for trade = virtual + real (pump.fun model).
        sol_in = buy_size_sol
        # Apply fee
        fee_pct = PUMP_FEE_BPS / 10000.0
        sol_in_after_fee = sol_in * (1 - fee_pct)

        # Effective reserves
        eff_sol = state.virtual_sol_reserves + state.real_sol_reserves
        eff_tokens = state.virtual_token_reserves + state.real_token_reserves

        # Constant product
        k = eff_sol * eff_tokens
        new_eff_sol = eff_sol + sol_in_after_fee
        new_eff_tokens = k / new_eff_sol
        tokens_out = eff_tokens - new_eff_tokens

        # Effective price = sol_in / tokens_out
        effective_price = sol_in / tokens_out if tokens_out > 0 else 0.0
        spot_price = state.current_price_sol
        slippage_pct = ((effective_price - spot_price) / spot_price * 100) if spot_price > 0 else 0.0
        return effective_price, tokens_out, slippage_pct

    def is_imminent_migration(self, state: BondingCurveState) -> bool:
        """True if bonding curve is >= 95% complete = imminent migration."""
        return state.completion_pct >= 95.0
