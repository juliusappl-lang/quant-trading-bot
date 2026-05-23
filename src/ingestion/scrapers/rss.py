import sqlite3
from datetime import datetime, timezone

import feedparser

from src.db.repository import insert_headline

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://feeds.reuters.com/reuters/businessNews",
]


def _normalize_date(entry) -> str:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        t = entry.published_parsed
        dt = datetime(t[0], t[1], t[2], t[3], t[4], t[5], tzinfo=timezone.utc)
        return dt.isoformat()
    return datetime.now(timezone.utc).isoformat()


def scrape_rss(conn: sqlite3.Connection, ticker: str) -> int:
    inserted = 0
    urls = [u.format(ticker=ticker) for u in RSS_FEEDS]
    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                if ticker.upper() not in title.upper():
                    continue
                link = entry.get("link", "")
                published_at = _normalize_date(entry)
                rowid = insert_headline(
                    conn, ticker=ticker, source="rss",
                    headline=title, url=link, published_at=published_at,
                )
                if rowid:
                    inserted += 1
        except Exception as e:
            print(f"RSS scrape failed for {url}: {e}")
    return inserted
