import sqlite3
import threading

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import get_db
from src.db.repository import add_to_watchlist, get_watchlist, remove_from_watchlist
from src.ingestion.historical import run_historical_ingestion

router = APIRouter()


class WatchlistItem(BaseModel):
    ticker: str
    asset_type: str


@router.get("/watchlist")
def list_watchlist(conn: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    return [dict(r) for r in get_watchlist(conn)]


@router.post("/watchlist")
def add_ticker(item: WatchlistItem, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    ticker = item.ticker.upper()
    add_to_watchlist(conn, ticker=ticker, asset_type=item.asset_type)
    threading.Thread(
        target=run_historical_ingestion, args=(conn, ticker), daemon=True
    ).start()
    return {"ticker": ticker, "asset_type": item.asset_type, "historical_ingestion": "started"}


@router.delete("/watchlist/{ticker}")
def delete_ticker(ticker: str, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    remove_from_watchlist(conn, ticker.upper())
    return {"removed": ticker.upper()}
