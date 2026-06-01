import sqlite3
from datetime import timezone

import yfinance as yf

from src.db.repository import insert_headline
from src.ingestion.market_data import fetch_and_store_market_data
from src.ingestion.scrapers.rss import _matches_ticker


def _quarter(month: int) -> int:
    return (month - 1) // 3 + 1


def _scrape_yfinance_news(conn: sqlite3.Connection, ticker: str) -> int:
    inserted = 0
    try:
        news = yf.Ticker(ticker).news or []
        for item in news:
            content = item.get("content", {})
            if not isinstance(content, dict):
                continue
            title = content.get("title", "").strip()
            pub_date = content.get("pubDate", "")
            if not title or not pub_date:
                continue
            if not _matches_ticker(title, ticker):
                continue
            url = ""
            canonical = content.get("canonicalUrl", {})
            if isinstance(canonical, dict):
                url = canonical.get("url", "")
            rowid = insert_headline(
                conn, ticker=ticker, source="yfinance_news",
                headline=title, url=url,
                published_at=pub_date,
            )
            if rowid:
                inserted += 1
    except Exception as e:
        print(f"yfinance news scrape failed for {ticker}: {e}")
    return inserted


def _scrape_earnings_synthetic(conn: sqlite3.Connection, ticker: str) -> int:
    inserted = 0
    try:
        t = yf.Ticker(ticker)
        earnings = t.earnings_dates
        if earnings is None or earnings.empty:
            return 0
        for date, row in earnings.iterrows():
            try:
                eps_est = row.get("EPS Estimate")
                eps_act = row.get("Reported EPS")
                if eps_est is None or eps_act is None:
                    continue
                q = _quarter(date.month)
                year = date.year
                if eps_act > eps_est * 1.02:
                    headline = f"{ticker} Q{q} {year} earnings beat estimates"
                elif eps_act < eps_est * 0.98:
                    headline = f"{ticker} Q{q} {year} earnings missed estimates"
                else:
                    headline = f"{ticker} Q{q} {year} earnings in line with estimates"
                pub = date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date
                rowid = insert_headline(
                    conn, ticker=ticker, source="earnings_synthetic",
                    headline=headline, url=None,
                    published_at=pub.isoformat(),
                )
                if rowid:
                    inserted += 1
            except Exception:
                continue
    except Exception as e:
        print(f"Earnings synthetic scrape failed for {ticker}: {e}")
    return inserted


def run_historical_ingestion(conn: sqlite3.Connection, ticker: str) -> dict:
    """Full historical bootstrap for a newly added ticker. Safe to call multiple times."""
    print(f"[historical] Starting bootstrap for {ticker}")
    market_n = fetch_and_store_market_data(conn, ticker, period="max")
    news_n = _scrape_yfinance_news(conn, ticker)
    earnings_n = _scrape_earnings_synthetic(conn, ticker)
    result = {"market_rows": market_n, "news": news_n, "earnings_synthetic": earnings_n}
    print(f"[historical] {ticker}: {result}")
    return result
