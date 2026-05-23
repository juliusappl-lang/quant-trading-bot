import sqlite3
from datetime import datetime, timezone

import feedparser

from src.db.repository import insert_headline

# Ticker-specific feeds use {ticker} placeholder.
# General feeds are filtered by ticker mention in the headline title.
TICKER_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
]

GENERAL_FEEDS = [
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",       # MarketWatch top stories
    "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",      # MarketWatch market pulse
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",                    # WSJ Markets
    "https://feeds.bloomberg.com/markets/news.rss",                     # Bloomberg Markets
    "https://www.investing.com/rss/news.rss",                           # Investing.com
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",            # CNBC Top News
    "https://www.cnbc.com/id/15839135/device/rss/rss.html",             # CNBC Markets
    "https://seekingalpha.com/market_currents.xml",                     # Seeking Alpha
]


def _normalize_date(entry) -> str:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        t = entry.published_parsed
        dt = datetime(t[0], t[1], t[2], t[3], t[4], t[5], tzinfo=timezone.utc)
        return dt.isoformat()
    return datetime.now(timezone.utc).isoformat()


def scrape_rss(conn: sqlite3.Connection, ticker: str) -> int:
    inserted = 0
    ticker_upper = ticker.upper()

    ticker_urls = [u.format(ticker=ticker) for u in TICKER_FEEDS]
    for url in ticker_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                rowid = insert_headline(
                    conn, ticker=ticker, source="rss",
                    headline=title, url=entry.get("link", ""),
                    published_at=_normalize_date(entry),
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
                if not title:
                    continue
                if ticker_upper not in title.upper():
                    continue
                rowid = insert_headline(
                    conn, ticker=ticker, source="rss",
                    headline=title, url=entry.get("link", ""),
                    published_at=_normalize_date(entry),
                )
                if rowid:
                    inserted += 1
        except Exception as e:
            print(f"RSS scrape failed for {url}: {e}")

    return inserted
