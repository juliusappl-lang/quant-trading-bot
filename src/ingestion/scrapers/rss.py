import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import feedparser

from src.db.repository import insert_headline

TICKER_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
]

GENERAL_FEEDS = [
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://www.investing.com/rss/news.rss",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/15839135/device/rss/rss.html",
    "https://seekingalpha.com/market_currents.xml",
]

TICKER_ALIASES: dict[str, list[str]] = {
    "AAPL":    ["Apple", "AAPL"],
    "TSLA":    ["Tesla", "TSLA"],
    "GOOGL":   ["Google", "Alphabet", "GOOGL"],
    "GOOG":    ["Google", "Alphabet", "GOOG"],
    "MSFT":    ["Microsoft", "MSFT"],
    "AMZN":    ["Amazon", "AMZN"],
    "NVDA":    ["Nvidia", "NVDA"],
    "META":    ["Meta", "Facebook", "META"],
    "NFLX":    ["Netflix", "NFLX"],
    "BTC-USD": ["Bitcoin", "BTC"],
    "ETH-USD": ["Ethereum", "ETH"],
    "BNB-USD": ["Binance", "BNB"],
    "SOL-USD": ["Solana", "SOL"],
}


def _matches_ticker(title: str, ticker: str) -> bool:
    """True if any alias appears as a whole word in the title."""
    aliases = TICKER_ALIASES.get(ticker.upper(), [ticker.upper()])
    for alias in aliases:
        # \b = word boundary — "Nvidia-Style" matches, "Finvidian" does not
        if re.search(r"\b" + re.escape(alias) + r"\b", title, re.IGNORECASE):
            return True
    return False


def _get_published_at(entry) -> Optional[str]:
    """Return ISO date string from the feed entry, or None if unavailable."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        t = entry.published_parsed
        dt = datetime(t[0], t[1], t[2], t[3], t[4], t[5], tzinfo=timezone.utc)
        return dt.isoformat()
    # Try updated_parsed as fallback (some feeds use this instead)
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        t = entry.updated_parsed
        dt = datetime(t[0], t[1], t[2], t[3], t[4], t[5], tzinfo=timezone.utc)
        return dt.isoformat()
    return None  # Skip articles with no date — avoids scrape-time date contamination


def scrape_rss(conn: sqlite3.Connection, ticker: str) -> int:
    inserted = 0

    for url in [u.format(ticker=ticker) for u in TICKER_FEEDS]:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                published_at = _get_published_at(entry)
                if not title or published_at is None:
                    continue
                if not _matches_ticker(title, ticker):
                    continue
                rowid = insert_headline(
                    conn, ticker=ticker, source="rss",
                    headline=title, url=entry.get("link", ""),
                    published_at=published_at,
                )
                if rowid:
                    inserted += 1
        except Exception as e:
            print(f"RSS scrape failed for {url}: {e}")

    for url in GENERAL_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                published_at = _get_published_at(entry)
                if not title or published_at is None:
                    continue
                if not _matches_ticker(title, ticker):
                    continue
                rowid = insert_headline(
                    conn, ticker=ticker, source="rss",
                    headline=title, url=entry.get("link", ""),
                    published_at=published_at,
                )
                if rowid:
                    inserted += 1
        except Exception as e:
            print(f"RSS scrape failed for {url}: {e}")

    return inserted
