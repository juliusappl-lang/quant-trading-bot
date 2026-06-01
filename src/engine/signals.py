import json
import sqlite3

from src.db.repository import (
    insert_signal, set_headline_processed, set_headline_embedding,
)
from src.engine.embeddings import embed_text, serialise
from src.engine.matcher import find_top_matches
from src.engine.features import build_feature_vector
from src.engine.model import SignalModel


def process_headline(conn: sqlite3.Connection, model: SignalModel, headline_id: int) -> None:
    row = conn.execute("SELECT * FROM headlines WHERE id = ?", (headline_id,)).fetchone()
    if row is None:
        return

    embedding = embed_text(row["headline"])
    set_headline_embedding(conn, headline_id, serialise(embedding))

    matches = find_top_matches(conn, embedding, ticker=row["ticker"], top_k=3, exclude_id=headline_id)
    features = build_feature_vector(embedding, matches, headline=row["headline"])

    signal, confidence = model.predict(features)

    top_matches_json = json.dumps([
        {
            "headline": m.headline,
            "date": m.date,
            "similarity": round(m.similarity, 4),
            "pct_change": m.pct_change,
        }
        for m in matches
    ])

    insert_signal(conn, headline_id=headline_id, ticker=row["ticker"],
                  signal=signal, confidence=confidence, top_matches=top_matches_json)
    set_headline_processed(conn, headline_id)
