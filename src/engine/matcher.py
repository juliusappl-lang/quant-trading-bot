import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.db.repository import get_processed_headlines_with_embeddings, get_pct_change_after
from src.engine.embeddings import deserialise


@dataclass
class Match:
    headline_id: int
    headline: str
    date: str
    similarity: float
    pct_change: Optional[float]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_top_matches(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    ticker: str,
    top_k: int = 3,
    exclude_id: Optional[int] = None,
) -> list[Match]:
    rows = get_processed_headlines_with_embeddings(conn, ticker, exclude_id=exclude_id)
    if not rows:
        return []

    scored = []
    for row in rows:
        vec = deserialise(row["embedding"])
        sim = _cosine_similarity(query_vec, vec)
        pct = get_pct_change_after(conn, ticker, row["published_at"][:10], days=3)
        scored.append(Match(
            headline_id=row["id"],
            headline=row["headline"],
            date=row["published_at"][:10],
            similarity=sim,
            pct_change=pct,
        ))

    scored.sort(key=lambda m: m.similarity, reverse=True)
    return scored[:top_k]
