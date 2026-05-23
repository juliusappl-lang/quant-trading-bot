import json
import sqlite3

from fastapi import APIRouter, Depends

from src.api.deps import get_db
from src.db.repository import get_signals

router = APIRouter()


def _row_to_dict(row) -> dict:
    d = dict(row)
    if d.get("top_matches"):
        try:
            d["top_matches"] = json.loads(d["top_matches"])
        except Exception:
            d["top_matches"] = []
    return d


@router.get("/signals")
def list_signals(conn: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    return [_row_to_dict(r) for r in get_signals(conn)]


@router.get("/signals/{ticker}")
def signals_by_ticker(ticker: str, conn: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    return [_row_to_dict(r) for r in get_signals(conn, ticker=ticker.upper())]
