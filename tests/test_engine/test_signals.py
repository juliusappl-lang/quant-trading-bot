import json
import numpy as np
import pytest
from src.db.models import init_db
from src.db.repository import (
    insert_headline, insert_market_data, set_headline_embedding,
    set_headline_processed, get_signals, get_pending_headlines,
)
from src.engine.embeddings import embed_text, serialise
from src.engine.model import SignalModel, LABELS
from src.engine.signals import process_headline


@pytest.fixture
def db_with_model(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    for i, hl in enumerate(["Apple beats earnings", "Apple misses revenue", "Apple launches product"]):
        hid = insert_headline(conn, ticker="AAPL", source="rss",
                              headline=hl, url=None, published_at=f"2023-0{i+1}-01T10:00:00")
        set_headline_embedding(conn, hid, serialise(embed_text(hl)))
        set_headline_processed(conn, hid)
        insert_market_data(conn, ticker="AAPL", date=f"2023-0{i+1}-01",
                           open=150.0, close=155.0, pct_change=float(i * 2 - 2))
    X = np.random.rand(30, 400).astype(np.float32)
    y = np.array(["BUY", "SELL", "HOLD"] * 10)
    model = SignalModel()
    model.train(X, y)
    return conn, model


def test_process_headline_creates_signal(db_with_model):
    conn, model = db_with_model
    hid = insert_headline(conn, ticker="AAPL", source="rss",
                          headline="Apple reports record quarter",
                          url=None, published_at="2024-01-01T10:00:00")
    process_headline(conn, model, headline_id=hid)
    sigs = get_signals(conn, ticker="AAPL")
    assert len(sigs) == 1
    assert sigs[0]["signal"] in LABELS
    assert 0.0 <= sigs[0]["confidence"] <= 1.0


def test_process_headline_marks_as_processed(db_with_model):
    conn, model = db_with_model
    hid = insert_headline(conn, ticker="AAPL", source="rss",
                          headline="Apple launches new product line",
                          url=None, published_at="2024-01-02T10:00:00")
    process_headline(conn, model, headline_id=hid)
    assert len(get_pending_headlines(conn)) == 0


def test_process_headline_stores_top_matches(db_with_model):
    conn, model = db_with_model
    hid = insert_headline(conn, ticker="AAPL", source="rss",
                          headline="Apple sets revenue record",
                          url=None, published_at="2024-01-03T10:00:00")
    process_headline(conn, model, headline_id=hid)
    sig = get_signals(conn, ticker="AAPL")[0]
    matches = json.loads(sig["top_matches"])
    assert isinstance(matches, list)
