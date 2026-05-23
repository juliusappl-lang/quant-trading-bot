import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api import deps
from src.db.models import init_db
from src.db.repository import insert_headline, insert_signal


@pytest.fixture
def client(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    hid = insert_headline(conn, ticker="AAPL", source="rss",
                          headline="Apple beats earnings", url=None,
                          published_at="2024-01-01T10:00:00")
    insert_signal(conn, headline_id=hid, ticker="AAPL",
                  signal="BUY", confidence=0.87, top_matches="[]")

    app.dependency_overrides[deps.get_db] = lambda: conn
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_get_all_signals(client):
    resp = client.get("/api/signals")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["signal"] == "BUY"
    assert data[0]["ticker"] == "AAPL"


def test_get_signals_by_ticker(client):
    resp = client.get("/api/signals/AAPL")
    assert resp.status_code == 200
    assert resp.json()[0]["ticker"] == "AAPL"


def test_get_signals_for_unknown_ticker(client):
    resp = client.get("/api/signals/ZZZZ")
    assert resp.status_code == 200
    assert resp.json() == []
