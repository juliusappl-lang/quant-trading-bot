import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api import deps
from src.db.models import init_db


@pytest.fixture
def client(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    app.dependency_overrides[deps.get_db] = lambda: conn
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_get_empty_watchlist(client):
    resp = client.get("/api/watchlist")
    assert resp.status_code == 200
    assert resp.json() == []


def test_add_ticker(client):
    resp = client.post("/api/watchlist", json={"ticker": "TSLA", "asset_type": "stock"})
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "TSLA"


def test_get_watchlist_after_add(client):
    client.post("/api/watchlist", json={"ticker": "TSLA", "asset_type": "stock"})
    resp = client.get("/api/watchlist")
    tickers = [r["ticker"] for r in resp.json()]
    assert "TSLA" in tickers


def test_remove_ticker(client):
    client.post("/api/watchlist", json={"ticker": "TSLA", "asset_type": "stock"})
    resp = client.delete("/api/watchlist/TSLA")
    assert resp.status_code == 200
    tickers = [r["ticker"] for r in client.get("/api/watchlist").json()]
    assert "TSLA" not in tickers
