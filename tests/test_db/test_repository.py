import pytest
from src.db.models import init_db
from src.db.repository import (
    insert_headline, get_pending_headlines, set_headline_processed,
    insert_market_data, get_market_data,
    insert_signal, get_signals,
    add_to_watchlist, get_watchlist, remove_from_watchlist,
)


@pytest.fixture
def db(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    yield conn
    conn.close()


def test_insert_and_get_pending_headline(db):
    insert_headline(db, ticker="AAPL", source="rss",
                    headline="Apple beats earnings", url="http://example.com",
                    published_at="2024-01-01T10:00:00")
    rows = get_pending_headlines(db)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["status"] == "pending"


def test_set_headline_processed(db):
    insert_headline(db, ticker="AAPL", source="rss",
                    headline="Apple beats earnings", url="http://example.com",
                    published_at="2024-01-01T10:00:00")
    rows = get_pending_headlines(db)
    set_headline_processed(db, rows[0]["id"])
    assert len(get_pending_headlines(db)) == 0


def test_insert_and_get_market_data(db):
    insert_market_data(db, ticker="AAPL", date="2024-01-01",
                       open=150.0, close=155.0, pct_change=3.3)
    rows = get_market_data(db, ticker="AAPL")
    assert len(rows) == 1
    assert rows[0]["pct_change"] == pytest.approx(3.3)


def test_insert_and_get_signals(db):
    insert_headline(db, ticker="AAPL", source="rss",
                    headline="Apple beats earnings", url="http://example.com",
                    published_at="2024-01-01T10:00:00")
    headline_id = get_pending_headlines(db)[0]["id"]
    insert_signal(db, headline_id=headline_id, ticker="AAPL",
                  signal="BUY", confidence=0.85, top_matches="[]")
    rows = get_signals(db, ticker="AAPL")
    assert rows[0]["signal"] == "BUY"
    assert rows[0]["confidence"] == pytest.approx(0.85)


def test_watchlist_crud(db):
    add_to_watchlist(db, ticker="TSLA", asset_type="stock")
    wl = get_watchlist(db)
    assert any(r["ticker"] == "TSLA" for r in wl)
    remove_from_watchlist(db, "TSLA")
    assert not any(r["ticker"] == "TSLA" for r in get_watchlist(db))
