"""
persistence.py
==============
SQLite persistence layer for positions, trade history, blacklist, and metrics.
Survives restarts so the agent doesn't lose state.

Schema:
- positions:      currently open positions (synced on every open/close)
- trades:         historical trade records (for Kelly + backtesting)
- blacklist:      tokens that failed rugpull or hit stop-loss
- daily_pnl:      daily PnL snapshots (for daily loss cap recovery after restart)
- agent_state:    key/value store for misc state (last_daily_reset, etc.)
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from risk.types import Position, TradeRecord, TradeStatus
from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("persistence")


SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    chain TEXT NOT NULL,
    token TEXT NOT NULL,
    strategy TEXT NOT NULL,
    entry_price REAL NOT NULL,
    size_base REAL NOT NULL,
    size_token REAL NOT NULL,
    opened_at REAL NOT NULL,
    stop_loss_pct REAL NOT NULL,
    take_profit_pct REAL NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (chain, token)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chain TEXT NOT NULL,
    token TEXT NOT NULL,
    strategy TEXT NOT NULL,
    side TEXT NOT NULL,
    size_base REAL NOT NULL,
    price REAL NOT NULL,
    ts REAL NOT NULL,
    pnl_pct REAL
);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);

CREATE TABLE IF NOT EXISTS blacklist (
    token TEXT PRIMARY KEY,
    expiry_ts REAL NOT NULL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    day TEXT PRIMARY KEY,           -- YYYY-MM-DD
    pnl_pct REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_state (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Attribution: per-trade features at buy time + realized pnl at close.
-- Drives the weekly AlphaSignal weight re-tuning (utils/attribution.py).
CREATE TABLE IF NOT EXISTS trade_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chain TEXT NOT NULL,
    token TEXT NOT NULL,
    strategy TEXT NOT NULL,           -- signal source (sniping/momentum/copy/...)
    entry_ts REAL NOT NULL,
    -- per-analyzer scores captured at buy time
    alpha_score REAL,
    order_flow_score REAL,
    smart_money_score REAL,
    token_quality_score REAL,
    lifecycle_score REAL,
    liquidity_score REAL,
    sentiment_score REAL,
    bonding_curve_score REAL,
    mev_penalty_score REAL,
    -- realized outcome (filled on close)
    exit_ts REAL,
    pnl_pct REAL,
    profitable INTEGER                -- 0/1/NULL (NULL = still open)
);
CREATE INDEX IF NOT EXISTS idx_trade_features_token ON trade_features(token);
CREATE INDEX IF NOT EXISTS idx_trade_features_strategy ON trade_features(strategy);
CREATE INDEX IF NOT EXISTS idx_trade_features_ts ON trade_features(entry_ts);

-- Tuned AlphaSignal weights (single row, updated by the weekly job).
CREATE TABLE IF NOT EXISTS alpha_weights (
    key TEXT PRIMARY KEY,             -- component name
    weight REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class Persistence:
    _instance: Optional["Persistence"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.cfg = Config.get()
        db_path = Path(self.cfg.get_nested("persistence", "db_path", default="./data/agent.db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        # check_same_thread=False because we use it from async tasks
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        log.info("persistence.ready", db=self.db_path)

    @classmethod
    def get(cls) -> "Persistence":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @contextmanager
    def _cursor(self):
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def upsert_position(self, pos: Position) -> None:
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO positions (chain, token, strategy, entry_price, size_base,
                                       size_token, opened_at, stop_loss_pct,
                                       take_profit_pct, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chain, token) DO UPDATE SET
                    strategy=excluded.strategy,
                    entry_price=excluded.entry_price,
                    size_base=excluded.size_base,
                    size_token=excluded.size_token,
                    opened_at=excluded.opened_at,
                    stop_loss_pct=excluded.stop_loss_pct,
                    take_profit_pct=excluded.take_profit_pct,
                    status=excluded.status
            """, (pos.chain, pos.token, pos.strategy, pos.entry_price, pos.size_base,
                  pos.size_token, pos.opened_at, pos.stop_loss_pct,
                  pos.take_profit_pct, pos.status.value))

    def delete_position(self, chain: str, token: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM positions WHERE chain=? AND token=?", (chain, token))

    def load_open_positions(self) -> dict[str, dict]:
        """Returns dict keyed by f'{chain}:{token}'."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM positions WHERE status='OPEN'")
            rows = cur.fetchall()
        out = {}
        for r in rows:
            key = f"{r['chain']}:{r['token']}"
            out[key] = {
                "chain": r["chain"], "token": r["token"], "strategy": r["strategy"],
                "entry_price": r["entry_price"], "size_base": r["size_base"],
                "size_token": r["size_token"], "opened_at": r["opened_at"],
                "stop_loss_pct": r["stop_loss_pct"], "take_profit_pct": r["take_profit_pct"],
            }
        return out

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------
    def insert_trade(self, t: TradeRecord) -> None:
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO trades (chain, token, strategy, side, size_base, price, ts, pnl_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (t.chain, t.token, t.strategy, t.side, t.size_base, t.price, t.ts, t.pnl_pct))

    def load_recent_trades(self, limit: int = 100) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM trades ORDER BY ts DESC LIMIT ?", (limit,)
            )
            return [dict(r) for r in cur.fetchall()]

    def load_trades_for_strategy(self, strategy: str, limit: int = 100) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM trades WHERE strategy=? ORDER BY ts DESC LIMIT ?",
                (strategy, limit),
            )
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Blacklist
    # ------------------------------------------------------------------
    def add_blacklist(self, token: str, expiry_ts: float, reason: str = "") -> None:
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO blacklist (token, expiry_ts, reason)
                VALUES (?, ?, ?)
                ON CONFLICT(token) DO UPDATE SET expiry_ts=excluded.expiry_ts, reason=excluded.reason
            """, (token, expiry_ts, reason))

    def is_blacklisted(self, token: str) -> bool:
        with self._cursor() as cur:
            cur.execute(
                "SELECT 1 FROM blacklist WHERE token=? AND expiry_ts > ?", (token, time.time())
            )
            return cur.fetchone() is not None

    def purge_expired_blacklist(self) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM blacklist WHERE expiry_ts <= ?", (time.time(),))

    def load_blacklist(self) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM blacklist WHERE expiry_ts > ?", (time.time(),))
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Daily PnL
    # ------------------------------------------------------------------
    def upsert_daily_pnl(self, day: str, pnl_pct: float) -> None:
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO daily_pnl (day, pnl_pct, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(day) DO UPDATE SET pnl_pct=excluded.pnl_pct, updated_at=excluded.updated_at
            """, (day, pnl_pct, time.time()))

    def load_daily_pnl(self, day: str) -> Optional[float]:
        with self._cursor() as cur:
            cur.execute("SELECT pnl_pct FROM daily_pnl WHERE day=?", (day,))
            row = cur.fetchone()
            return float(row["pnl_pct"]) if row else None

    # ------------------------------------------------------------------
    # Agent state (key/value)
    # ------------------------------------------------------------------
    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._cursor() as cur:
            cur.execute("SELECT value FROM agent_state WHERE key=?", (key,))
            row = cur.fetchone()
            return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO agent_state (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (key, value))

    # ------------------------------------------------------------------
    # Trade features (attribution + weight re-tuning)
    # ------------------------------------------------------------------
    def insert_trade_features(self, features: dict) -> None:
        """Insert a trade_features row at buy time (profitable left NULL)."""
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO trade_features
                    (chain, token, strategy, entry_ts, alpha_score,
                     order_flow_score, smart_money_score, token_quality_score,
                     lifecycle_score, liquidity_score, sentiment_score,
                     bonding_curve_score, mev_penalty_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                features["chain"], features["token"], features["strategy"],
                features["entry_ts"], features.get("alpha_score"),
                features.get("order_flow_score"), features.get("smart_money_score"),
                features.get("token_quality_score"), features.get("lifecycle_score"),
                features.get("liquidity_score"), features.get("sentiment_score"),
                features.get("bonding_curve_score"), features.get("mev_penalty_score"),
            ))

    def close_trade_features(self, chain: str, token: str, entry_ts: float,
                             exit_ts: float, pnl_pct: float) -> int:
        """
        Mark the most recent open trade_features row for (chain, token, entry_ts)
        as closed with the realized pnl. Returns rows updated.
        """
        with self._cursor() as cur:
            cur.execute("""
                UPDATE trade_features
                SET exit_ts=?, pnl_pct=?, profitable=?
                WHERE chain=? AND token=? AND entry_ts=?
                  AND profitable IS NULL
                ORDER BY id DESC LIMIT 1
            """, (exit_ts, pnl_pct, 1 if pnl_pct > 0 else 0,
                  chain, token, entry_ts))
            return cur.rowcount

    def load_closed_trade_features(self, since_ts: float = 0.0, limit: int = 5000) -> list[dict]:
        """Load closed trade features for weight fitting."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT * FROM trade_features
                WHERE profitable IS NOT NULL AND entry_ts >= ?
                ORDER BY entry_ts DESC LIMIT ?
            """, (since_ts, limit))
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Alpha weights (dynamic, re-tuned weekly)
    # ------------------------------------------------------------------
    def load_alpha_weights(self) -> Optional[dict[str, float]]:
        with self._cursor() as cur:
            cur.execute("SELECT key, weight FROM alpha_weights")
            rows = cur.fetchall()
            if not rows:
                return None
            return {r["key"]: r["weight"] for r in rows}

    def upsert_alpha_weights(self, weights: dict[str, float]) -> None:
        now = time.time()
        with self._cursor() as cur:
            for key, weight in weights.items():
                cur.execute("""
                    INSERT INTO alpha_weights (key, weight, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        weight=excluded.weight, updated_at=excluded.updated_at
                """, (key, weight, now))

    def close(self) -> None:
        self._conn.close()
