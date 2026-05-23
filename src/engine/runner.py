import time
import sqlite3

from src.db.repository import get_pending_headlines
from src.engine.model import SignalModel
from src.engine.signals import process_headline


def run_once(conn: sqlite3.Connection, model: SignalModel) -> int:
    pending = get_pending_headlines(conn)
    for row in pending:
        process_headline(conn, model, headline_id=row["id"])
    return len(pending)


def run_loop(conn: sqlite3.Connection, model: SignalModel, poll_interval: int = 30) -> None:
    print(f"Signal engine running (poll every {poll_interval}s) ...")
    while True:
        n = run_once(conn, model)
        if n:
            print(f"Processed {n} headline(s)")
        time.sleep(poll_interval)
