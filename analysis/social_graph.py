"""
social_graph.py
===============
On-chain social graph analyzer.

Builds a graph of wallet-to-wallet relationships based on shared token
activity, then identifies:
- **Smart money clusters**: wallets that consistently buy early on tokens
  that subsequently 5x+.
- **Whale syndicates**: groups of 3+ wallets that buy the same tokens
  within minutes of each other (likely coordinated).
- **Dev networks**: wallets that have created multiple pump.fun tokens
  (often repeat scammers or repeat winners).
- **Influencer wallets**: wallets whose buys are followed by sustained
  buy pressure from many other wallets within 30 minutes.

This is the highest-signal data in memecoin trading. A token bought by
a known smart money cluster has a much higher expected value than one
bought by random retail.

NOTE: This is a structural framework. For real smart-money identification,
integrate Cielo.finance API (paid) or Arkham Intelligence API. Both have
pre-computed profitable wallet lists.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("social_graph")


@dataclass
class WalletProfile:
    address: str
    first_seen_ts: float
    total_trades: int = 0
    profitable_trades: int = 0
    total_pnl_usd: float = 0.0
    avg_buy_to_peak_pct: float = 0.0       # avg % gain at peak after their buys
    early_buy_count: int = 0                # buys within 60s of token launch
    cluster_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)  # "smart_money", "whale", "dev", "syndicate"
    reputation_score: float = 0.0           # 0..100


@dataclass
class ClusterInfo:
    cluster_id: str
    wallets: list[str]
    shared_token_buys: int
    avg_coordination_seconds: float         # avg time between first and last wallet's buy
    tags: list[str]


class SocialGraphAnalyzer:
    """Builds and queries the on-chain wallet relationship graph."""

    # Thresholds for classification
    SMART_MONEY_MIN_WINRATE = 0.55
    SMART_MONEY_MIN_TRADES = 20
    SMART_MONEY_MIN_AVG_GAIN = 50.0         # avg +50% at peak
    SYNDICATE_MIN_WALLETS = 3
    SYNDICATE_MAX_TIME_GAP_SEC = 300        # 5 minutes

    def __init__(self) -> None:
        self.cfg = Config.get()
        # Address -> profile
        self._wallets: dict[str, WalletProfile] = {}
        # token_mint -> list of (wallet, ts, side) — used to detect coordination
        self._token_activity: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
        # cluster_id -> ClusterInfo
        self._clusters: dict[str, ClusterInfo] = {}

    def record_wallet_activity(
        self, wallet: str, token: str, ts: float, side: str, pnl_pct: Optional[float] = None,
        early_buy: bool = False, peak_gain_pct: Optional[float] = None,
    ) -> None:
        """Record that a wallet made a trade on a token."""
        prof = self._wallets.setdefault(wallet, WalletProfile(
            address=wallet, first_seen_ts=ts
        ))
        prof.total_trades += 1
        if pnl_pct is not None:
            if pnl_pct > 0:
                prof.profitable_trades += 1
            prof.total_pnl_usd += pnl_pct  # rough proxy
        if early_buy:
            prof.early_buy_count += 1
        if peak_gain_pct is not None:
            # Running average
            n = prof.total_trades
            prof.avg_buy_to_peak_pct = ((n - 1) * prof.avg_buy_to_peak_pct + peak_gain_pct) / n

        self._token_activity[token].append((wallet, ts, side))

        # Recompute reputation + tags
        self._update_reputation(prof)

    def _update_reputation(self, prof: WalletProfile) -> None:
        """Recompute reputation score and tags for a wallet."""
        prof.tags = []
        if prof.total_trades < self.SMART_MONEY_MIN_TRADES:
            prof.reputation_score = 0
            return

        winrate = prof.profitable_trades / prof.total_trades
        # Smart money check
        if (winrate >= self.SMART_MONEY_MIN_WINRATE
            and prof.avg_buy_to_peak_pct >= self.SMART_MONEY_MIN_AVG_GAIN):
            prof.tags.append("smart_money")
            prof.reputation_score = min(100, winrate * 50 + prof.avg_buy_to_peak_pct / 4)

        # Early buyer bonus
        if prof.early_buy_count >= 10:
            prof.tags.append("early_sniper")
            prof.reputation_score = min(100, prof.reputation_score + 15)

        # Whale tag (placeholder; would integrate with size data)
        if prof.total_pnl_usd > 10_000:
            prof.tags.append("whale")
            prof.reputation_score = min(100, prof.reputation_score + 10)

    def detect_clusters(self) -> list[ClusterInfo]:
        """
        Detect syndicate clusters: groups of wallets that buy the same tokens
        within a short time window, repeatedly.
        """
        # Group wallets by co-occurrence on tokens
        wallet_pairs: dict[tuple[str, str], int] = defaultdict(int)
        for token, activity in self._token_activity.items():
            # Sort by timestamp
            sorted_act = sorted(activity, key=lambda x: x[1])
            # For each pair of wallets that bought this token within window
            for i, (w1, t1, _) in enumerate(sorted_act):
                for j in range(i + 1, len(sorted_act)):
                    w2, t2, _ = sorted_act[j]
                    if t2 - t1 > self.SYNDICATE_MAX_TIME_GAP_SEC:
                        break
                    if w1 == w2:
                        continue
                    pair = tuple(sorted([w1, w2]))
                    wallet_pairs[pair] += 1

        # Build clusters via union-find
        parent: dict[str, str] = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for (w1, w2), count in wallet_pairs.items():
            if count >= 3:  # at least 3 shared token buys
                union(w1, w2)

        # Group by root
        cluster_members: dict[str, list[str]] = defaultdict(list)
        for w in parent:
            cluster_members[find(w)].append(w)

        clusters = []
        for root, members in cluster_members.items():
            if len(members) < self.SYNDICATE_MIN_WALLETS:
                continue
            # Compute average coordination time
            coord_times = []
            shared_buys = 0
            for token, activity in self._token_activity.items():
                wallets_in_token = [w for w, _, _ in activity if w in members]
                if len(set(wallets_in_token)) >= 2:
                    times = [t for w, t, _ in activity if w in members]
                    if times:
                        coord_times.append(max(times) - min(times))
                    shared_buys += 1
            avg_coord = sum(coord_times) / len(coord_times) if coord_times else 0
            cluster = ClusterInfo(
                cluster_id=root[:8],
                wallets=members,
                shared_token_buys=shared_buys,
                avg_coordination_seconds=avg_coord,
                tags=["syndicate"] + (["smart_money"] if any(
                    "smart_money" in self._wallets.get(w, WalletProfile(w, 0)).tags for w in members
                ) else []),
            )
            clusters.append(cluster)
            self._clusters[root[:8]] = cluster

        log.info("social_graph.clusters_detected", count=len(clusters))
        return clusters

    def get_wallet_profile(self, address: str) -> Optional[WalletProfile]:
        return self._wallets.get(address)

    def is_smart_money(self, address: str) -> bool:
        prof = self._wallets.get(address)
        return prof is not None and "smart_money" in prof.tags

    def get_cluster_for_wallet(self, address: str) -> Optional[ClusterInfo]:
        for c in self._clusters.values():
            if address in c.wallets:
                return c
        return None
