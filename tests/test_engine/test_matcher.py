import pytest
from src.db.models import init_db
from src.db.repository import (
    insert_headline, set_headline_embedding, set_headline_processed, insert_market_data,
)
from src.engine.embeddings import embed_text, serialise
from src.engine.matcher import find_top_matches, Match


@pytest.fixture
def db_with_history(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    headlines = [
        ("AAPL", "Apple beats Q3 earnings by wide margin", "2023-06-01T10:00:00", "2023-06-01", 4.2),
        ("AAPL", "Apple misses revenue estimates in Q2", "2023-03-01T10:00:00", "2023-03-01", -3.1),
        ("AAPL", "Apple launches new iPhone with record pre-orders", "2023-09-01T10:00:00", "2023-09-01", 2.8),
    ]
    for ticker, hl, pub, date, pct in headlines:
        hid = insert_headline(conn, ticker=ticker, source="rss", headline=hl,
                              url=None, published_at=pub)
        set_headline_embedding(conn, hid, serialise(embed_text(hl)))
        set_headline_processed(conn, hid)
        insert_market_data(conn, ticker=ticker, date=date, open=150.0, close=156.0, pct_change=pct)
    return conn


def test_returns_top_k_matches(db_with_history):
    query_vec = embed_text("Apple reports strong quarterly results")
    matches = find_top_matches(db_with_history, query_vec, ticker="AAPL", top_k=2)
    assert len(matches) == 2


def test_match_has_required_fields(db_with_history):
    query_vec = embed_text("Apple reports strong quarterly results")
    matches = find_top_matches(db_with_history, query_vec, ticker="AAPL", top_k=1)
    m = matches[0]
    assert isinstance(m, Match)
    assert 0.0 <= m.similarity <= 1.0
    assert m.pct_change is not None
    assert isinstance(m.headline, str)


def test_most_similar_headline_ranked_first(db_with_history):
    query_vec = embed_text("Apple crushes quarterly earnings estimates")
    matches = find_top_matches(db_with_history, query_vec, ticker="AAPL", top_k=3)
    similarities = [m.similarity for m in matches]
    assert similarities == sorted(similarities, reverse=True)
