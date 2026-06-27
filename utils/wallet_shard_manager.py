"""
wallet_shard_manager.py
=======================
Multi-wallet sharding for anti-fingerprinting + parallel sniping.

WHY THIS EXISTS:
MEV bots on Solana fingerprint addresses. If every snipe comes from the same
wallet, competitors learn your pattern: "this address buys every new launch
within 500ms = frontrun it." By rotating across N derived sub-wallets, each
snipe appears to come from a different actor, making fingerprinting much harder.

Secondary benefit: parallelism. A single wallet can only have one pending tx
at a time (nonce/blockhash conflicts). With N shards, we can fire N concurrent
snipes without serialization.

HOW IT WORKS:
- Derives N Solana keypairs from the master seed using different derivation
  paths (m/44'/501'/0'/0', m/44'/501'/1'/0', ... m/44'/501'/(N-1)'/0').
- Each shard is a full signing keypair; funds must be distributed to them
  (a top-up helper is provided).
- `acquire()` returns the least-recently-used free shard (round-robin) and
  marks it busy; `release()` frees it after the tx lands.
- The solana_adapter can optionally use a shard keypair instead of the master
  for snipe buys.

SECURITY:
- Shards are derived deterministically from the same seed (recoverable).
- Private keys never leave the process.
- Each shard's pubkey is exposed for monitoring; the keypair itself is not.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("wallet_shard")

_DEFAULT_SHARD_COUNT = 3


@dataclass
class WalletShard:
    """One derived sub-wallet for sharded sniping."""
    index: int                       # 0-based shard index
    keypair: object                  # solders.keypair.Keypair
    pubkey: str
    busy: bool = False
    last_used_ts: float = 0.0
    snipes_fired: int = 0


class WalletShardManager:
    """
    Manages a pool of derived Solana sub-wallets for anti-fingerprinting.
    """

    def __init__(self, master_seed: bytes, count: int = _DEFAULT_SHARD_COUNT) -> None:
        """
        Args:
            master_seed: the BIP39 seed bytes (from the mnemonic).
            count: number of shards to derive.
        """
        self.master_seed = master_seed
        self.count = count
        self.shards: list[WalletShard] = []
        self._derive_all()
        log.info("wallet_shard.initialized",
                 count=len(self.shards),
                 pubkeys=[s.pubkey[:8] for s in self.shards])

    def _derive_all(self) -> None:
        """Derive all shard keypairs from the master seed."""
        try:
            from solders.keypair import Keypair
        except ImportError:
            log.error("wallet_shard.solders_unavailable")
            return
        for i in range(self.count):
            path = f"m/44'/501'/{i}'/0'"
            try:
                if hasattr(Keypair, "from_seed_and_derivation_path"):
                    kp = Keypair.from_seed_and_derivation_path(self.master_seed, path)
                else:
                    # Fallback: derive from seed + index (non-BIP44 but deterministic)
                    kp = Keypair.from_seed(self.master_seed[:32] + i.to_bytes(4, "little"))
            except Exception as e:
                log.warning("wallet_shard.derive_failed", index=i, error=str(e))
                continue
            self.shards.append(WalletShard(
                index=i, keypair=kp, pubkey=str(kp.pubkey()),
            ))

    def acquire(self) -> Optional[WalletShard]:
        """
        Get the least-recently-used free shard and mark it busy.
        Returns None if all shards are busy (caller should retry or use master).
        """
        free = [s for s in self.shards if not s.busy]
        if not free:
            return None
        # Pick the least-recently-used to spread load evenly
        shard = min(free, key=lambda s: s.last_used_ts)
        shard.busy = True
        shard.last_used_ts = time.time()
        shard.snipes_fired += 1
        return shard

    def release(self, shard: WalletShard) -> None:
        """Free a shard after its tx has landed or failed."""
        shard.busy = False

    def release_by_pubkey(self, pubkey: str) -> None:
        """Free a shard identified by its pubkey."""
        for s in self.shards:
            if s.pubkey == pubkey:
                s.busy = False
                return

    def all_pubkeys(self) -> list[str]:
        """Return all shard pubkeys (for balance monitoring / top-up)."""
        return [s.pubkey for s in self.shards]

    def stats(self) -> list[dict]:
        """Snapshot for dashboard / debugging."""
        return [
            {
                "index": s.index,
                "pubkey": s.pubkey,
                "busy": s.busy,
                "snipes_fired": s.snipes_fired,
                "last_used_ago_sec": round(time.time() - s.last_used_ts, 0) if s.last_used_ts else None,
            }
            for s in self.shards
        ]

    @property
    def available_count(self) -> int:
        return sum(1 for s in self.shards if not s.busy)


# ----------------------------------------------------------------------
# Atomic sniper+exit bundle builder
# ----------------------------------------------------------------------
"""
Atomic sniper+exit: for high-risk launches, build a single Jito bundle that
contains BOTH the buy AND a conditional sell-at-target. If the token pumps
within the same bundle's slots, the sell lands atomically — we exit in green
before any MEV bot can react. This is what MEV searchers do, not retail bots.

Implementation note: a TRUE atomic conditional sell requires a custom on-chain
program (the sell ix would need a price-check guard). Without deploying our own
program, we approximate with a 2-tx bundle: [buy, sell@market] submitted together.
The sell lands 1-2 slots after the buy; if the price moved up, we capture the
spread. If it didn't, we eat a small loss (the round-trip fee). This is only
used for HIGH-RISK launches where the edge justifies the atomicity cost.
"""
import base64


def build_atomic_snipe_exit_bundle(
    buyer_keypair,
    mint: str,
    buy_amount_lamports: int,
    sell_pct_of_buy: float,
    slippage_bps: int,
    blockhash,
) -> Optional[list[str]]:
    """
    Build a 2-tx Jito bundle: [buy, sell]. Both signed by the same keypair
    on the same fresh blockhash.

    Args:
        buyer_keypair: solders.keypair.Keypair (the shard or master).
        mint: token mint address.
        buy_amount_lamports: SOL to spend, in lamports.
        sell_pct_of_buy: fraction of bought tokens to immediately sell (0..1).
        slippage_bps: slippage tolerance.
        blockhash: fresh solders.hash.Hash.

    Returns:
        list of 2 base64-encoded signed transactions, or None on failure.
    """
    try:
        from solders.transaction import VersionedTransaction
        from solders.message import MessageV0
        from solders.compute_budget import set_compute_unit_price, set_compute_unit_limit
        from utils import pumpfun_ix
        from chains.jito_client import JitoClient

        user_pubkey = str(buyer_keypair.pubkey())

        # Tx 1: BUY
        # We don't know exact token output without bonding-curve math here, so
        # we use a generous max_sol_cost and let the buy ix compute on-chain.
        buy_ix = pumpfun_ix.build_buy_ix(
            user_pubkey, mint,
            token_amount_raw=1,  # placeholder; real impl queries bonding curve
            max_sol_cost_lamports=int(buy_amount_lamports * (1 + slippage_bps / 10_000)),
        )
        cu_limit = set_compute_unit_limit(200_000)
        cu_price = set_compute_unit_price(5_000)
        msg1 = MessageV0.try_compile(
            payer=buyer_keypair.pubkey(),
            instructions=[cu_limit, cu_price, pumpfun_ix.build_create_ata_ix(user_pubkey, mint), buy_ix],
            address_lookup_table_accounts=[],
            recent_blockhash=blockhash,
        )
        tx1 = VersionedTransaction(msg1, [buyer_keypair])

        # Tx 2: SELL (a fraction of what we just bought). This lands 1-2 slots
        # after the buy in the same bundle. If price pumped, we capture it.
        # We use a large token amount proxy; the sell will execute for whatever
        # the ATA holds (the buy just funded it).
        # NOTE: this is the approximation. A production version would compute
        # exact tokens from the bonding curve and sell that precise amount.
        sell_ix = pumpfun_ix.build_sell_ix(
            user_pubkey, mint,
            token_amount_raw=int(1_000_000 * sell_pct_of_buy),  # proxy amount
        )
        msg2 = MessageV0.try_compile(
            payer=buyer_keypair.pubkey(),
            instructions=[cu_limit, cu_price, sell_ix],
            address_lookup_table_accounts=[],
            recent_blockhash=blockhash,
        )
        tx2 = VersionedTransaction(msg2, [buyer_keypair])

        return [
            base64.b64encode(bytes(tx1)).decode("ascii"),
            base64.b64encode(bytes(tx2)).decode("ascii"),
        ]
    except Exception as e:
        log.error("atomic_bundle.build_failed", mint=mint, error=str(e))
        return None
