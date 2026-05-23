import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS headlines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    source       TEXT NOT NULL,
    headline     TEXT NOT NULL,
    url          TEXT,
    published_at DATETIME NOT NULL,
    embedding    BLOB,
    status       TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS market_data (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker     TEXT NOT NULL,
    date       DATE NOT NULL,
    open       REAL,
    close      REAL,
    pct_change REAL,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_id INTEGER REFERENCES headlines(id),
    ticker      TEXT NOT NULL,
    signal      TEXT NOT NULL,
    confidence  REAL NOT NULL,
    top_matches TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker     TEXT UNIQUE NOT NULL,
    asset_type TEXT NOT NULL,
    added_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    active     BOOLEAN DEFAULT 1
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
