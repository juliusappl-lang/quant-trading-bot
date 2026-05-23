import os
import sqlite3

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

from src.db.models import init_db
from src.db.repository import get_watchlist
from src.ingestion.market_data import fetch_and_store_market_data
from src.ingestion.scrapers.rss import scrape_rss

load_dotenv()


def run_ingestion(conn: sqlite3.Connection) -> None:
    tickers = [r["ticker"] for r in get_watchlist(conn)]
    if not tickers:
        print("Watchlist is empty, skipping ingestion.")
        return
    for ticker in tickers:
        rss_n = scrape_rss(conn, ticker)
        market_n = fetch_and_store_market_data(conn, ticker)
        print(f"{ticker}: +{rss_n} RSS headlines, +{market_n} market rows")


def main() -> None:
    db_path = os.getenv("DB_PATH", "data/trading.db")
    conn = init_db(db_path)

    scheduler = BlockingScheduler()
    scheduler.add_job(run_ingestion, "interval", minutes=15, args=[conn])
    print("Ingestion scheduler started (every 15 min). Press Ctrl+C to stop.")
    run_ingestion(conn)
    scheduler.start()


if __name__ == "__main__":
    main()
