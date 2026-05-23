import sqlite3

import yfinance as yf

from src.db.repository import insert_market_data


def fetch_and_store_market_data(conn: sqlite3.Connection, ticker: str, period: str = "2y") -> int:
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        print(f"Failed to fetch market data for {ticker}: {e}")
        return 0

    if hist.empty:
        return 0

    stored = 0
    for date, row in hist.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        open_price = float(row["Open"]) if row["Open"] else 0.0
        pct_change = float((row["Close"] - row["Open"]) / row["Open"] * 100) if open_price else 0.0
        insert_market_data(
            conn, ticker=ticker,
            date=date_str, open=open_price,
            close=float(row["Close"]), pct_change=pct_change,
        )
        stored += 1
    return stored
