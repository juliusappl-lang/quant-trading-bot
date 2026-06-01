import sqlite3
from typing import Optional


def insert_headline(conn: sqlite3.Connection, *, ticker: str, source: str,
                    headline: str, url: Optional[str], published_at: str) -> Optional[int]:
    cur = conn.execute(
        "INSERT OR IGNORE INTO headlines (ticker, source, headline, url, published_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (ticker, source, headline, url, published_at),
    )
    conn.commit()
    return cur.lastrowid


def get_pending_headlines(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM headlines WHERE status = 'pending' ORDER BY published_at ASC"
    ).fetchall()


def set_headline_embedding(conn: sqlite3.Connection, headline_id: int, embedding: bytes) -> None:
    conn.execute("UPDATE headlines SET embedding = ? WHERE id = ?", (embedding, headline_id))
    conn.commit()


def set_headline_processed(conn: sqlite3.Connection, headline_id: int) -> None:
    conn.execute("UPDATE headlines SET status = 'processed' WHERE id = ?", (headline_id,))
    conn.commit()


def get_processed_headlines_with_embeddings(
    conn: sqlite3.Connection, ticker: str, exclude_id: Optional[int] = None
) -> list[sqlite3.Row]:
    if exclude_id is not None:
        return conn.execute(
            "SELECT * FROM headlines WHERE status = 'processed' AND embedding IS NOT NULL "
            "AND ticker = ? AND id != ?",
            (ticker, exclude_id),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM headlines WHERE status = 'processed' AND embedding IS NOT NULL AND ticker = ?",
        (ticker,),
    ).fetchall()


def insert_market_data(conn: sqlite3.Connection, *, ticker: str, date: str,
                       open: float, close: float, pct_change: float) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO market_data (ticker, date, open, close, pct_change) "
        "VALUES (?, ?, ?, ?, ?)",
        (ticker, date, open, close, pct_change),
    )
    conn.commit()


def get_market_data(conn: sqlite3.Connection, ticker: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM market_data WHERE ticker = ? ORDER BY date ASC", (ticker,)
    ).fetchall()


def get_pct_change_after(conn: sqlite3.Connection, ticker: str, date: str, days: int) -> Optional[float]:
    """Returns the cumulative pct_change sum over `days` rows starting from `date`."""
    rows = conn.execute(
        "SELECT pct_change FROM market_data WHERE ticker = ? AND date >= ? "
        "ORDER BY date ASC LIMIT ?",
        (ticker, date, days),
    ).fetchall()
    if not rows:
        return None
    return float(sum(r["pct_change"] for r in rows if r["pct_change"] is not None))


def insert_signal(conn: sqlite3.Connection, *, headline_id: int, ticker: str,
                  signal: str, confidence: float, top_matches: str) -> None:
    conn.execute(
        "INSERT INTO signals (headline_id, ticker, signal, confidence, top_matches) "
        "VALUES (?, ?, ?, ?, ?)",
        (headline_id, ticker, signal, confidence, top_matches),
    )
    conn.commit()


def get_signals(conn: sqlite3.Connection, ticker: Optional[str] = None,
                limit: int = 100) -> list[sqlite3.Row]:
    if ticker:
        return conn.execute(
            "SELECT s.*, h.headline, h.published_at as headline_date FROM signals s "
            "JOIN headlines h ON s.headline_id = h.id "
            "WHERE s.ticker = ? ORDER BY s.created_at DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
    return conn.execute(
        "SELECT s.*, h.headline, h.published_at as headline_date FROM signals s "
        "JOIN headlines h ON s.headline_id = h.id "
        "ORDER BY s.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()


def add_to_watchlist(conn: sqlite3.Connection, ticker: str, asset_type: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO watchlist (ticker, asset_type, active) VALUES (?, ?, 1)",
        (ticker, asset_type),
    )
    conn.commit()


def get_watchlist(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM watchlist WHERE active = 1"
    ).fetchall()


def remove_from_watchlist(conn: sqlite3.Connection, ticker: str) -> None:
    conn.execute("UPDATE watchlist SET active = 0 WHERE ticker = ?", (ticker,))
    conn.commit()
