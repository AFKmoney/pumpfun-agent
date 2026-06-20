"""
sentiment.py
============
Social sentiment integrator.

Aggregates mentions from:
- Twitter/X (via Twitter API v2 — requires Bearer token)
- Telegram (via Telegram Bot API — same bot token as our alerts)
- Discord (optional, via Discord Bot token)
- Pump.fun comments (via pump.fun API)

For each token, computes:
- Mention count in last 1h, 6h, 24h
- Sentiment score (-1.0 to +1.0)
- Velocity (mentions per minute, last 30 min)
- Influencer mentions (followers > 10k)
- Hype score (composite, 0..100)

The hype score is the strongest leading indicator for memecoins — tokens
with rising social mention velocity typically pump 30 min - 4h ahead of
the price move.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from utils.config_loader import Config
from utils.logger import setup_logger

log = setup_logger("sentiment")


# Positive/negative keyword lists (very simple; in production use a proper
# sentiment model like VADER or a fine-tuned BERT)
POSITIVE_KEYWORDS = {
    "moon", "pump", "bullish", "gem", "alpha", "send", "runner", "massive",
    "huge", "early", "free money", "locked", "renounced", "based", "chad",
    "wagmi", "lfg", "next 100x", "next pepe", "next bonk",
}
NEGATIVE_KEYWORDS = {
    "rug", "scam", "honeypot", "dump", "bearish", "dead", "abandoned",
    "fake", "sketchy", "avoid", "fraud", "honey", "sell", "rekt", "rip",
    "down bad", "lost", "exit liquidity",
}


@dataclass
class Mention:
    platform: str          # "twitter" | "telegram" | "discord" | "pumpfun"
    text: str
    ts: float
    author: str
    author_followers: int = 0
    sentiment: float = 0.0    # -1 to +1


@dataclass
class SentimentSnapshot:
    mint: str
    mention_count_1h: int
    mention_count_6h: int
    mention_count_24h: int
    sentiment_score: float          # -1.0 to +1.0
    mentions_per_minute: float      # last 30 min velocity
    influencer_mentions: int        # mentions by authors with >10k followers
    hype_score: float               # 0..100
    top_keywords: list[str]
    fetched_at: float = field(default_factory=time.time)


class SentimentAnalyzer:
    """Aggregates social sentiment across platforms."""

    INFLUENCER_THRESHOLD = 10_000

    def __init__(self) -> None:
        self.cfg = Config.get()
        self._mentions: dict[str, deque[Mention]] = {}  # mint -> mentions
        self._twitter_token = os.environ.get("TWITTER_BEARER_TOKEN", "")
        self._discord_token = os.environ.get("DISCORD_BOT_TOKEN", "")
        self._session: Optional[aiohttp.ClientSession] = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self._session

    # ------------------------------------------------------------------
    # Mention ingestion
    # ------------------------------------------------------------------
    def add_mention(self, mint: str, mention: Mention) -> None:
        if mint not in self._mentions:
            self._mentions[mint] = deque(maxlen=5000)
        self._mentions[mint].append(mention)

    def _compute_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment. Returns -1..1."""
        text_lower = text.lower()
        pos = sum(1 for k in POSITIVE_KEYWORDS if k in text_lower)
        neg = sum(1 for k in NEGATIVE_KEYWORDS if k in text_lower)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract $TICKER and hashtags."""
        tickers = re.findall(r"\$[A-Za-z]{2,10}", text)
        hashtags = re.findall(r"#(\w+)", text)
        return [t.lower() for t in tickers] + [h.lower() for h in hashtags]

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def snapshot(self, mint: str) -> SentimentSnapshot:
        mentions = self._mentions.get(mint, deque())
        now = time.time()
        m_1h = [m for m in mentions if now - m.ts <= 3600]
        m_6h = [m for m in mentions if now - m.ts <= 21600]
        m_24h = [m for m in mentions if now - m.ts <= 86400]
        m_30m = [m for m in mentions if now - m.ts <= 1800]

        if m_1h:
            sentiment = sum(m.sentiment for m in m_1h) / len(m_1h)
        else:
            sentiment = 0.0

        velocity = len(m_30m) / 30 if m_30m else 0
        influencer_count = sum(1 for m in m_1h if m.author_followers >= self.INFLUENCER_THRESHOLD)

        # Hype score = composite of:
        # - mention velocity (40%)
        # - sentiment (30%)
        # - influencer mentions (30%)
        vel_score = min(100, velocity * 20)                    # 5/min = 100
        sent_score = (sentiment + 1) / 2 * 100                  # -1..1 -> 0..100
        inf_score = min(100, influencer_count * 25)             # 4 influencers = 100
        hype = 0.4 * vel_score + 0.3 * sent_score + 0.3 * inf_score

        # Top keywords from last 1h
        all_kw: dict[str, int] = {}
        for m in m_1h:
            for kw in self._extract_keywords(m.text):
                all_kw[kw] = all_kw.get(kw, 0) + 1
        top_kw = sorted(all_kw.items(), key=lambda x: -x[1])[:10]
        top_kw_list = [k for k, _ in top_kw]

        return SentimentSnapshot(
            mint=mint,
            mention_count_1h=len(m_1h),
            mention_count_6h=len(m_6h),
            mention_count_24h=len(m_24h),
            sentiment_score=sentiment,
            mentions_per_minute=velocity,
            influencer_mentions=influencer_count,
            hype_score=hype,
            top_keywords=top_kw_list,
        )

    # ------------------------------------------------------------------
    # Twitter / X ingestion (requires Bearer token)
    # ------------------------------------------------------------------
    async def fetch_twitter_mentions(self, mint: str, symbol: str) -> int:
        """Search Twitter for recent mentions of $SYMBOL. Returns count ingested."""
        if not self._twitter_token:
            return 0
        try:
            s = await self.session()
            query = f"${symbol} (pump.fun OR pumpfun) -is:retweet lang:en"
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {self._twitter_token}"}
            params = {"query": query, "max_results": 100,
                      "tweet.fields": "created_at,author_id,public_metrics"}
            async with s.get(url, headers=headers, params=params) as r:
                if r.status != 200:
                    return 0
                data = await r.json()
            count = 0
            for tweet in data.get("data", []):
                ts_str = tweet.get("created_at", "")
                # Parse ISO timestamp
                try:
                    from datetime import datetime
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = time.time()
                text = tweet.get("text", "")
                self.add_mention(mint, Mention(
                    platform="twitter", text=text, ts=ts,
                    author=tweet.get("author_id", ""),
                    author_followers=tweet.get("public_metrics", {}).get("followers", 0),
                    sentiment=self._compute_sentiment(text),
                ))
                count += 1
            return count
        except Exception as e:
            log.warning("sentiment.twitter_failed", error=str(e))
            return 0

    # ------------------------------------------------------------------
    # Telegram ingestion (uses our existing bot)
    # ------------------------------------------------------------------
    async def fetch_telegram_mentions(self, mint: str, symbol: str, chat_id: str) -> int:
        """Fetch recent messages from a Telegram chat mentioning the symbol."""
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not bot_token:
            return 0
        try:
            s = await self.session()
            # Telegram getUpdates (limited; for chat history need to be a member)
            # In production: register bot in pump.fun-related chats and use
            # getChat + forwardWebhook to a webhook endpoint.
            url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
            async with s.get(url) as r:
                data = await r.json()
            count = 0
            for update in data.get("result", []):
                msg = update.get("message", {})
                text = msg.get("text", "")
                if symbol.lower() not in text.lower():
                    continue
                ts = msg.get("date", time.time())
                self.add_mention(mint, Mention(
                    platform="telegram", text=text, ts=ts,
                    author=str(msg.get("from", {}).get("id", "")),
                    sentiment=self._compute_sentiment(text),
                ))
                count += 1
            return count
        except Exception as e:
            log.warning("sentiment.telegram_failed", error=str(e))
            return 0

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
