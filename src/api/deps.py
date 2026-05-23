import os
import sqlite3
from functools import lru_cache

from dotenv import load_dotenv

from src.db.models import init_db

load_dotenv()


@lru_cache(maxsize=1)
def get_db() -> sqlite3.Connection:
    db_path = os.getenv("DB_PATH", "data/trading.db")
    return init_db(db_path)
