import os
import sqlite3
from datetime import datetime, timezone

import praw

from src.db.repository import insert_headline

SUBREDDITS = ["wallstreetbets", "investing", "stocks"]


def _get_reddit() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.getenv("REDDIT_USER_AGENT", "signal-engine/1.0"),
    )


def scrape_reddit(conn: sqlite3.Connection, ticker: str, limit: int = 25) -> int:
    inserted = 0
    try:
        reddit = _get_reddit()
    except KeyError as e:
        print(f"Reddit credentials not configured ({e}). Skipping Reddit scrape.")
        return 0

    for sub in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.search(ticker, sort="new", limit=limit):
                title = post.title.strip()
                published_at = datetime.fromtimestamp(
                    post.created_utc, tz=timezone.utc
                ).isoformat()
                url = f"https://reddit.com{post.permalink}"
                rowid = insert_headline(
                    conn, ticker=ticker, source="reddit",
                    headline=title, url=url, published_at=published_at,
                )
                if rowid:
                    inserted += 1
        except Exception as e:
            print(f"Reddit scrape failed for r/{sub}: {e}")
    return inserted
