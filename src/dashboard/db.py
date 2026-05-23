import os
import sqlite3

import streamlit as st
from dotenv import load_dotenv

from src.db.models import init_db

load_dotenv()


@st.cache_resource
def get_connection() -> sqlite3.Connection:
    db_path = os.getenv("DB_PATH", "data/trading.db")
    return init_db(db_path)
