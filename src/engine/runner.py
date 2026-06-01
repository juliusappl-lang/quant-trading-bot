import os
import time
import sqlite3
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.db.models import init_db
from src.db.repository import get_pending_headlines
from src.engine.model import SignalModel
from src.engine.signals import process_headline

load_dotenv()


def run_once(conn: sqlite3.Connection, model: Optional[SignalModel]) -> int:
    if model is None:
        return 0
    pending = get_pending_headlines(conn)
    for row in pending:
        process_headline(conn, model, headline_id=row["id"])
    return len(pending)


def run_loop(conn: sqlite3.Connection, model: Optional[SignalModel], poll_interval: int = 30) -> None:
    print(f"Signal engine running (poll every {poll_interval}s) ...")
    while True:
        if model is None:
            model_path = os.getenv("MODEL_PATH", "models/signal_classifier.pkl")
            if Path(model_path).exists():
                model = SignalModel.load(model_path)
                print(f"Model loaded from {model_path}")
            else:
                print("Waiting for model — run 'poetry run python -m src.train' first")
        n = run_once(conn, model)
        if n:
            print(f"Processed {n} headline(s)")
        time.sleep(poll_interval)


def _load_model(model_path: str) -> Optional[SignalModel]:
    if Path(model_path).exists():
        print(f"Loaded model from {model_path}")
        return SignalModel.load(model_path)
    print(f"No model found at {model_path} — waiting for training")
    return None


if __name__ == "__main__":
    db_path = os.getenv("DB_PATH", "data/trading.db")
    model_path = os.getenv("MODEL_PATH", "models/signal_classifier.pkl")
    conn = init_db(db_path)
    model = _load_model(model_path)
    run_loop(conn, model)
