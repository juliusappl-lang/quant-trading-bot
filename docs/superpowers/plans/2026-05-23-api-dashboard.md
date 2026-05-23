# API + Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI REST backend and a Streamlit dashboard with 4 pages: Live Signals, Watchlist Management, Signal History, and Model Training — all wired to the existing SQLite database.

**Architecture:** FastAPI reads/writes SQLite via `src/db/repository.py`. Streamlit connects directly to SQLite (no API hop needed for internal display). Both run as separate processes. FastAPI is the integration point for external tools and the training trigger.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, Streamlit, matplotlib, pandas, sqlite3 (stdlib), joblib

**Prerequisite:** Core Infrastructure plan must be complete (`src/db/`, `src/engine/` all implemented and tests passing).

---

## File Map

| File | Responsibility |
|---|---|
| `src/api/main.py` | FastAPI app factory, router registration, CORS |
| `src/api/routes/signals.py` | GET /api/signals, GET /api/signals/{ticker} |
| `src/api/routes/watchlist.py` | GET/POST/DELETE /api/watchlist |
| `src/api/routes/train.py` | POST /api/train (triggers training subprocess) |
| `src/api/deps.py` | Shared FastAPI dependencies (DB connection) |
| `src/dashboard/app.py` | Streamlit entrypoint, page routing |
| `src/dashboard/pages/signals.py` | Page 1: Live signals table with color coding |
| `src/dashboard/pages/watchlist.py` | Page 2: Watchlist CRUD |
| `src/dashboard/pages/history.py` | Page 3: Price chart + signal markers |
| `src/dashboard/pages/model.py` | Page 4: Training trigger + metrics + confusion matrix |
| `src/dashboard/db.py` | Shared DB connection for Streamlit |
| `tests/test_api/test_signals.py` | Signal endpoint tests |
| `tests/test_api/test_watchlist.py` | Watchlist endpoint tests |

---

## Task 1: FastAPI dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add API dependencies**

Add to `[tool.poetry.dependencies]` in `pyproject.toml`:

```toml
fastapi = "^0.111"
uvicorn = {extras = ["standard"], version = "^0.30"}
httpx = "^0.27"
```

- [ ] **Step 2: Install**

```bash
poetry install
```

Expected: resolves without errors.

- [ ] **Step 3: Create API package structure**

```bash
mkdir -p src/api/routes tests/test_api
touch src/api/__init__.py src/api/routes/__init__.py tests/test_api/__init__.py
```

- [ ] **Step 4: Commit**

```
chore(api): add FastAPI and uvicorn dependencies
```

---

## Task 2: Shared DB dependency + FastAPI app

**Files:**
- Create: `src/api/deps.py`
- Create: `src/api/main.py`

- [ ] **Step 1: Implement src/api/deps.py**

```python
# src/api/deps.py
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
```

- [ ] **Step 2: Implement src/api/main.py**

```python
# src/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import signals, watchlist, train

app = FastAPI(title="Signal Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router, prefix="/api")
app.include_router(watchlist.router, prefix="/api")
app.include_router(train.router, prefix="/api")
```

- [ ] **Step 3: Verify app imports cleanly**

```bash
poetry run python -c "from src.api.main import app; print('OK')"
```

Expected: OK.

---

## Task 3: Signals endpoint

**Files:**
- Create: `src/api/routes/signals.py`
- Create: `tests/test_api/test_signals.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api/test_signals.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sqlite3
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

    with patch("src.api.deps.get_db", return_value=conn):
        from src.api.main import app
        yield TestClient(app)

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_api/test_signals.py -v
```

Expected: ImportError or ModuleNotFoundError.

- [ ] **Step 3: Implement src/api/routes/signals.py**

```python
# src/api/routes/signals.py
import json
from fastapi import APIRouter, Depends
import sqlite3
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_api/test_signals.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```
feat(api): add signals endpoints GET /api/signals and GET /api/signals/{ticker}
```

---

## Task 4: Watchlist endpoint

**Files:**
- Create: `src/api/routes/watchlist.py`
- Create: `tests/test_api/test_watchlist.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api/test_watchlist.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.db.models import init_db

@pytest.fixture
def client(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    with patch("src.api.deps.get_db", return_value=conn):
        from src.api.main import app
        yield TestClient(app)

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_api/test_watchlist.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement src/api/routes/watchlist.py**

```python
# src/api/routes/watchlist.py
import sqlite3
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.api.deps import get_db
from src.db.repository import add_to_watchlist, get_watchlist, remove_from_watchlist

router = APIRouter()

class WatchlistItem(BaseModel):
    ticker: str
    asset_type: str

@router.get("/watchlist")
def list_watchlist(conn: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    return [dict(r) for r in get_watchlist(conn)]

@router.post("/watchlist")
def add_ticker(item: WatchlistItem, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    add_to_watchlist(conn, ticker=item.ticker.upper(), asset_type=item.asset_type)
    return {"ticker": item.ticker.upper(), "asset_type": item.asset_type}

@router.delete("/watchlist/{ticker}")
def delete_ticker(ticker: str, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    remove_from_watchlist(conn, ticker.upper())
    return {"removed": ticker.upper()}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_api/test_watchlist.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```
feat(api): add watchlist CRUD endpoints
```

---

## Task 5: Training trigger endpoint

**Files:**
- Create: `src/api/routes/train.py`

- [ ] **Step 1: Implement src/api/routes/train.py**

```python
# src/api/routes/train.py
import subprocess
import sys
from fastapi import APIRouter

router = APIRouter()

@router.post("/train")
def trigger_training() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "src.train"],
        capture_output=True, text=True, timeout=300
    )
    return {
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout,
        "error": result.stderr,
    }
```

- [ ] **Step 2: Add API server entry point to pyproject.toml**

```toml
[tool.poetry.scripts]
train = "src.train:main"
ingest = "src.ingestion.scheduler:main"
api = "uvicorn src.api.main:app --reload --port 8000"
```

Wait — `poetry scripts` doesn't support inline shell args. Instead add:

```toml
serve-api = "src.api.server:main"
```

And create `src/api/server.py`:

```python
# src/api/server.py
import uvicorn

def main():
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 3: Verify API starts**

```bash
poetry run serve-api
```

Expected: Uvicorn starts on http://0.0.0.0:8000. Open http://localhost:8000/docs — Swagger UI should show all endpoints.

- [ ] **Step 4: Commit**

```
feat(api): add training trigger endpoint and uvicorn server entry point
```

---

## Task 6: Streamlit dashboard — shared DB + page routing

**Files:**
- Create: `src/dashboard/db.py`
- Create: `src/dashboard/app.py`
- Create: `src/dashboard/pages/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add Streamlit dependency**

Add to `[tool.poetry.dependencies]`:

```toml
streamlit = "^1.35"
```

Run `poetry install`.

- [ ] **Step 2: Create dashboard package**

```bash
mkdir -p src/dashboard/pages
touch src/dashboard/__init__.py src/dashboard/pages/__init__.py
```

- [ ] **Step 3: Implement src/dashboard/db.py**

```python
# src/dashboard/db.py
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
```

- [ ] **Step 4: Implement src/dashboard/app.py**

```python
# src/dashboard/app.py
import streamlit as st

st.set_page_config(page_title="Signal Engine", layout="wide")

pages = {
    "Live Signals": "src/dashboard/pages/signals.py",
    "Watchlist": "src/dashboard/pages/watchlist.py",
    "Signal History": "src/dashboard/pages/history.py",
    "Model": "src/dashboard/pages/model.py",
}

page = st.sidebar.radio("Navigation", list(pages.keys()))

if page == "Live Signals":
    from src.dashboard.pages import signals as p
elif page == "Watchlist":
    from src.dashboard.pages import watchlist as p
elif page == "Signal History":
    from src.dashboard.pages import history as p
else:
    from src.dashboard.pages import model as p

p.render()
```

- [ ] **Step 5: Add Streamlit entry point to pyproject.toml**

```toml
[tool.poetry.scripts]
train = "src.train:main"
ingest = "src.ingestion.scheduler:main"
serve-api = "src.api.server:main"
dashboard = "src.dashboard.server:main"
```

Create `src/dashboard/server.py`:

```python
# src/dashboard/server.py
import subprocess
import sys

def main():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])
```

- [ ] **Step 6: Commit**

```
feat(dashboard): add Streamlit app shell and shared DB connection
```

---

## Task 7: Page 1 — Live Signals

**Files:**
- Create: `src/dashboard/pages/signals.py`

- [ ] **Step 1: Implement src/dashboard/pages/signals.py**

```python
# src/dashboard/pages/signals.py
import pandas as pd
import streamlit as st
from src.dashboard.db import get_connection
from src.db.repository import get_signals, get_watchlist

def render():
    st.title("Live Signals")

    conn = get_connection()
    tickers = ["All"] + [r["ticker"] for r in get_watchlist(conn)]
    selected = st.selectbox("Filter by Ticker", tickers)
    signal_filter = st.multiselect("Signal Type", ["BUY", "SELL", "HOLD"],
                                   default=["BUY", "SELL", "HOLD"])

    rows = get_signals(conn, ticker=None if selected == "All" else selected, limit=200)

    if not rows:
        st.info("No signals yet. Add tickers to watchlist and run ingestion.")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    df = df[df["signal"].isin(signal_filter)]

    def color_signal(val):
        colors = {"BUY": "background-color: #d4edda", "SELL": "background-color: #f8d7da",
                  "HOLD": "background-color: #fff3cd"}
        return colors.get(val, "")

    display_cols = ["ticker", "signal", "confidence", "headline", "created_at"]
    existing = [c for c in display_cols if c in df.columns]
    styled = df[existing].style.applymap(color_signal, subset=["signal"])
    st.dataframe(styled, use_container_width=True)
```

- [ ] **Step 2: Smoke test**

```bash
poetry run dashboard
```

Open http://localhost:8501. Expected: "No signals yet." message on Live Signals page — correct for empty DB.

- [ ] **Step 3: Commit**

```
feat(dashboard): add Live Signals page with color-coded table
```

---

## Task 8: Page 2 — Watchlist Management

**Files:**
- Create: `src/dashboard/pages/watchlist.py`

- [ ] **Step 1: Implement src/dashboard/pages/watchlist.py**

```python
# src/dashboard/pages/watchlist.py
import streamlit as st
from src.dashboard.db import get_connection
from src.db.repository import get_watchlist, add_to_watchlist, remove_from_watchlist

def render():
    st.title("Watchlist")
    conn = get_connection()

    st.subheader("Add Ticker")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        new_ticker = st.text_input("Ticker Symbol", placeholder="AAPL").upper()
    with col2:
        asset_type = st.selectbox("Asset Type", ["stock", "crypto"])
    with col3:
        st.write("")
        st.write("")
        if st.button("Add") and new_ticker:
            add_to_watchlist(conn, ticker=new_ticker, asset_type=asset_type)
            st.success(f"Added {new_ticker}")
            st.rerun()

    st.subheader("Active Tickers")
    watchlist = get_watchlist(conn)
    if not watchlist:
        st.info("No tickers in watchlist.")
        return

    for row in watchlist:
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.write(row["ticker"])
        c2.write(row["asset_type"])
        if c3.button("Remove", key=f"rm_{row['ticker']}"):
            remove_from_watchlist(conn, row["ticker"])
            st.rerun()
```

- [ ] **Step 2: Smoke test**

Open http://localhost:8501 → Watchlist. Add "AAPL" as stock. Expected: appears in list, Remove button works.

- [ ] **Step 3: Commit**

```
feat(dashboard): add Watchlist management page
```

---

## Task 9: Page 3 — Signal History

**Files:**
- Create: `src/dashboard/pages/history.py`

- [ ] **Step 1: Implement src/dashboard/pages/history.py**

```python
# src/dashboard/pages/history.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from src.dashboard.db import get_connection
from src.db.repository import get_watchlist, get_signals, get_market_data

def render():
    st.title("Signal History")
    conn = get_connection()

    tickers = [r["ticker"] for r in get_watchlist(conn)]
    if not tickers:
        st.info("No tickers in watchlist.")
        return

    ticker = st.selectbox("Select Ticker", tickers)

    market_rows = get_market_data(conn, ticker)
    signal_rows = get_signals(conn, ticker=ticker, limit=500)

    if not market_rows:
        st.warning(f"No market data for {ticker}. Run ingestion first.")
        return

    price_df = pd.DataFrame([dict(r) for r in market_rows])
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df.sort_values("date", inplace=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(price_df["date"], price_df["close"], color="#4C72B0", linewidth=1.5, label="Close Price")

    if signal_rows:
        sig_df = pd.DataFrame([dict(r) for r in signal_rows])
        sig_df["date"] = pd.to_datetime(sig_df["created_at"]).dt.normalize()
        buy = sig_df[sig_df["signal"] == "BUY"]
        sell = sig_df[sig_df["signal"] == "SELL"]

        if not buy.empty:
            buy_prices = price_df.set_index("date")["close"].reindex(buy["date"]).values
            ax.scatter(buy["date"], buy_prices, marker="^", color="green", s=80, label="BUY", zorder=5)
        if not sell.empty:
            sell_prices = price_df.set_index("date")["close"].reindex(sell["date"]).values
            ax.scatter(sell["date"], sell_prices, marker="v", color="red", s=80, label="SELL", zorder=5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title(f"{ticker} — Price with Signal Markers")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("All Signals")
    if signal_rows:
        df = pd.DataFrame([dict(r) for r in signal_rows])
        st.dataframe(df[["signal", "confidence", "headline", "created_at"]],
                     use_container_width=True)
    else:
        st.info("No signals generated yet for this ticker.")
```

- [ ] **Step 2: Smoke test**

Open http://localhost:8501 → Signal History. Select AAPL. Expected: price chart shown (if market data exists), or warning message if not.

- [ ] **Step 3: Commit**

```
feat(dashboard): add Signal History page with price chart and signal markers
```

---

## Task 10: Page 4 — Model Training

**Files:**
- Create: `src/dashboard/pages/model.py`

- [ ] **Step 1: Implement src/dashboard/pages/model.py**

```python
# src/dashboard/pages/model.py
import os
import subprocess
import sys
from pathlib import Path
import streamlit as st
from src.dashboard.db import get_connection

def render():
    st.title("Model")
    conn = get_connection()

    model_path = os.getenv("MODEL_PATH", "models/signal_classifier.pkl")
    confusion_path = "models/confusion_matrix.png"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Status")
        if Path(model_path).exists():
            mtime = Path(model_path).stat().st_mtime
            import datetime
            last_trained = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            st.success(f"Model trained — last updated: {last_trained}")
        else:
            st.warning("No trained model found.")

        n_processed = conn.execute(
            "SELECT COUNT(*) as n FROM headlines WHERE status = 'processed'"
        ).fetchone()["n"]
        st.metric("Processed Headlines (training data)", n_processed)

    with col2:
        st.subheader("Train Model")
        st.write("Trains on all processed headlines with known 3-day market outcomes.")
        if st.button("Train Now", type="primary"):
            with st.spinner("Training..."):
                result = subprocess.run(
                    [sys.executable, "-m", "src.train"],
                    capture_output=True, text=True, timeout=300
                )
            if result.returncode == 0:
                st.success("Training complete!")
                st.text(result.stdout)
            else:
                st.error("Training failed.")
                st.text(result.stderr)

    if Path(confusion_path).exists():
        st.subheader("Confusion Matrix")
        st.image(confusion_path)
```

- [ ] **Step 2: Smoke test**

Open http://localhost:8501 → Model. Expected: Status panel shows model state, metric shows 0 processed headlines, "Train Now" button runs training (which exits with "Not enough training samples" message).

- [ ] **Step 3: Commit**

```
feat(dashboard): add Model page with training trigger and confusion matrix
```

---

## Task 11: Full integration test + README update

- [ ] **Step 1: Run all tests**

```bash
poetry run pytest tests/ -v --tb=short
```

Expected: All tests PASS.

- [ ] **Step 2: Update README.md with usage instructions**

Write the following to `README.md`:

```markdown
# Quant Trading Bot — News-Driven Signal Engine

A trading signal engine that scrapes financial headlines, matches them against historical market-moving events, and uses a trained neural network to generate BUY/SELL/HOLD signals.

## Architecture

Three independent services coordinated via SQLite:

- **Ingestion** — scrapes RSS feeds + Reddit every 15 min
- **Signal Engine** — generates embeddings, matches history, runs MLPClassifier
- **Dashboard** — Streamlit UI + FastAPI REST API

## Setup

```bash
# 1. Install dependencies
poetry install

# 2. Configure environment
cp .env.example .env
# Fill in Reddit API credentials (free: https://www.reddit.com/prefs/apps)

# 3. Create data + model directories
mkdir -p data models
```

## Running

```bash
# Start ingestion (runs every 15 min)
poetry run ingest

# In a separate terminal: start API
poetry run serve-api

# In a separate terminal: start dashboard
poetry run dashboard

# Train model (after collecting some headlines)
poetry run train
```

## Dashboard

Open http://localhost:8501

- **Live Signals** — current BUY/SELL/HOLD signals with confidence scores
- **Watchlist** — add/remove tickers (stocks and crypto)
- **Signal History** — price chart with signal markers
- **Model** — trigger training, view accuracy and confusion matrix

## API

Open http://localhost:8000/docs for Swagger UI.

| Endpoint | Description |
|---|---|
| GET /api/signals | All recent signals |
| GET /api/signals/{ticker} | Signals for one ticker |
| GET /api/watchlist | Active watchlist |
| POST /api/watchlist | Add ticker |
| DELETE /api/watchlist/{ticker} | Remove ticker |
| POST /api/train | Trigger model training |

## ML Pipeline

1. Headline → `sentence-transformers` (all-MiniLM-L6-v2) → 384-dim embedding
2. Cosine similarity → Top-3 historical matches with known market outcomes
3. VADER sentiment → 4-dim sentiment vector
4. Combined ~400-dim feature vector → `sklearn.MLPClassifier`
5. Output: BUY / SELL / HOLD + confidence score
```

- [ ] **Step 3: Run linter and type check**

```bash
poetry run ruff check src/ tests/
poetry run mypy src/ --ignore-missing-imports
```

Expected: No errors.

- [ ] **Step 4: Final commit**

```
docs: update README with full setup and usage instructions
```
