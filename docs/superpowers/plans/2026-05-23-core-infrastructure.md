# Core Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete data pipeline: DB schema, RSS/Reddit ingestion, feature engineering (embeddings + similarity + VADER), scikit-learn MLPClassifier training and inference — all wired together and tested.

**Architecture:** Three Python modules (db, ingestion, engine) coordinate via SQLite. Ingestion writes `headlines` with `status=pending`. Signal engine polls pending headlines, computes a ~400-dim feature vector, runs MLPClassifier inference, writes to `signals`. Training is a separate CLI entry point.

**Tech Stack:** Python 3.11, Poetry, SQLite (stdlib), pandas, numpy, scikit-learn, sentence-transformers, vaderSentiment, feedparser, praw, yfinance, APScheduler, joblib, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `src/db/models.py` | Schema definitions + `init_db()` |
| `src/db/repository.py` | All DB reads/writes (no raw SQL outside this file) |
| `src/ingestion/scrapers/rss.py` | Fetch + parse RSS feeds per ticker |
| `src/ingestion/scrapers/reddit.py` | Fetch Reddit posts mentioning ticker |
| `src/ingestion/scheduler.py` | APScheduler wiring — runs scrapers every 15 min |
| `src/engine/embeddings.py` | Generate + deserialise sentence-transformer vectors |
| `src/engine/sentiment.py` | VADER compound/pos/neg/neu scores |
| `src/engine/matcher.py` | Cosine similarity, return top-K matches with market data |
| `src/engine/features.py` | Assemble final ~400-dim feature vector as numpy array |
| `src/engine/model.py` | MLPClassifier train/save/load/predict |
| `src/engine/signals.py` | Orchestrate pipeline for one pending headline |
| `src/engine/runner.py` | Poll loop — process all pending headlines |
| `src/train.py` | CLI entry point: `poetry run train` |
| `tests/test_db/test_repository.py` | DB read/write tests |
| `tests/test_engine/test_embeddings.py` | Embedding shape + serialisation |
| `tests/test_engine/test_sentiment.py` | VADER output range |
| `tests/test_engine/test_matcher.py` | Cosine similarity correctness |
| `tests/test_engine/test_features.py` | Feature vector shape + content |
| `tests/test_engine/test_model.py` | Train + predict smoke test |
| `tests/test_engine/test_signals.py` | Full pipeline integration test |

---

## Task 1: Project dependencies + directory structure

**Files:**
- Modify: `pyproject.toml`
- Create: `src/db/__init__.py`, `src/ingestion/__init__.py`, `src/ingestion/scrapers/__init__.py`, `src/engine/__init__.py`, `tests/__init__.py`, `tests/test_db/__init__.py`, `tests/test_engine/__init__.py`
- Create: `.env.example`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Replace the `[tool.poetry.dependencies]` section with:

```toml
[tool.poetry.dependencies]
python = "^3.11"
yfinance = "^0.2"
pandas = "^2.2"
numpy = "^1.26"
matplotlib = "^3.9"
scikit-learn = "^1.5"
sentence-transformers = "^3.0"
vaderSentiment = "^3.3"
feedparser = "^6.0"
praw = "^7.7"
apscheduler = "^3.10"
joblib = "^1.4"
python-dotenv = "^1.0"
```

- [ ] **Step 2: Install dependencies**

```bash
poetry install
```

Expected: resolves and installs all packages, no errors.

- [ ] **Step 3: Create package init files**

```bash
mkdir -p src/db src/ingestion/scrapers src/engine tests/test_db tests/test_engine
touch src/db/__init__.py src/ingestion/__init__.py src/ingestion/scrapers/__init__.py src/engine/__init__.py
touch tests/__init__.py tests/test_db/__init__.py tests/test_engine/__init__.py
```

- [ ] **Step 4: Create .env.example**

```bash
cat > .env.example << 'EOF'
# Reddit API credentials (free: https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=signal-engine/1.0

# SQLite DB path
DB_PATH=data/trading.db

# Model path
MODEL_PATH=models/signal_classifier.pkl
EOF
mkdir -p data models
echo "data/*.db" >> .gitignore
echo "models/*.pkl" >> .gitignore
echo ".env" >> .gitignore
```

- [ ] **Step 5: Verify pytest runs**

```bash
poetry run pytest --collect-only
```

Expected: "no tests ran" — collection works, no import errors.

---

## Task 2: Database schema + repository

**Files:**
- Create: `src/db/models.py`
- Create: `src/db/repository.py`
- Create: `tests/test_db/test_repository.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_db/test_repository.py
import os
import pytest
from src.db.models import init_db
from src.db.repository import (
    insert_headline, get_pending_headlines, set_headline_processed,
    insert_market_data, get_market_data,
    insert_signal, get_signals,
    add_to_watchlist, get_watchlist, remove_from_watchlist,
)

@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    yield conn, db_path
    conn.close()

def test_insert_and_get_pending_headline(db):
    conn, _ = db
    insert_headline(conn, ticker="AAPL", source="rss",
                    headline="Apple beats earnings", url="http://example.com",
                    published_at="2024-01-01T10:00:00")
    rows = get_pending_headlines(conn)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["status"] == "pending"

def test_set_headline_processed(db):
    conn, _ = db
    insert_headline(conn, ticker="AAPL", source="rss",
                    headline="Apple beats earnings", url="http://example.com",
                    published_at="2024-01-01T10:00:00")
    rows = get_pending_headlines(conn)
    set_headline_processed(conn, rows[0]["id"])
    assert len(get_pending_headlines(conn)) == 0

def test_insert_and_get_market_data(db):
    conn, _ = db
    insert_market_data(conn, ticker="AAPL", date="2024-01-01",
                       open=150.0, close=155.0, pct_change=3.3)
    rows = get_market_data(conn, ticker="AAPL")
    assert len(rows) == 1
    assert rows[0]["pct_change"] == pytest.approx(3.3)

def test_insert_and_get_signals(db):
    conn, _ = db
    insert_headline(conn, ticker="AAPL", source="rss",
                    headline="Apple beats earnings", url="http://example.com",
                    published_at="2024-01-01T10:00:00")
    headline_id = get_pending_headlines(conn)[0]["id"]
    insert_signal(conn, headline_id=headline_id, ticker="AAPL",
                  signal="BUY", confidence=0.85, top_matches="[]")
    rows = get_signals(conn, ticker="AAPL")
    assert rows[0]["signal"] == "BUY"
    assert rows[0]["confidence"] == pytest.approx(0.85)

def test_watchlist_crud(db):
    conn, _ = db
    add_to_watchlist(conn, ticker="TSLA", asset_type="stock")
    wl = get_watchlist(conn)
    assert any(r["ticker"] == "TSLA" for r in wl)
    remove_from_watchlist(conn, "TSLA")
    assert not any(r["ticker"] == "TSLA" for r in get_watchlist(conn))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_db/test_repository.py -v
```

Expected: ImportError or ModuleNotFoundError.

- [ ] **Step 3: Implement src/db/models.py**

```python
# src/db/models.py
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS headlines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    source       TEXT NOT NULL,
    headline     TEXT NOT NULL,
    url          TEXT,
    published_at DATETIME NOT NULL,
    embedding    BLOB,
    status       TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS market_data (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker     TEXT NOT NULL,
    date       DATE NOT NULL,
    open       REAL,
    close      REAL,
    pct_change REAL,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_id INTEGER REFERENCES headlines(id),
    ticker      TEXT NOT NULL,
    signal      TEXT NOT NULL,
    confidence  REAL NOT NULL,
    top_matches TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker     TEXT UNIQUE NOT NULL,
    asset_type TEXT NOT NULL,
    added_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    active     BOOLEAN DEFAULT 1
);
"""

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
```

- [ ] **Step 4: Implement src/db/repository.py**

```python
# src/db/repository.py
import sqlite3
from typing import Optional

def insert_headline(conn: sqlite3.Connection, *, ticker: str, source: str,
                    headline: str, url: Optional[str], published_at: str) -> int:
    cur = conn.execute(
        "INSERT OR IGNORE INTO headlines (ticker, source, headline, url, published_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (ticker, source, headline, url, published_at),
    )
    conn.commit()
    return cur.lastrowid

def get_pending_headlines(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM headlines WHERE status = 'pending' ORDER BY published_at ASC"
    ).fetchall()

def set_headline_embedding(conn: sqlite3.Connection, headline_id: int, embedding: bytes) -> None:
    conn.execute("UPDATE headlines SET embedding = ? WHERE id = ?", (embedding, headline_id))
    conn.commit()

def set_headline_processed(conn: sqlite3.Connection, headline_id: int) -> None:
    conn.execute("UPDATE headlines SET status = 'processed' WHERE id = ?", (headline_id,))
    conn.commit()

def get_processed_headlines_with_embeddings(conn: sqlite3.Connection, ticker: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM headlines WHERE status = 'processed' AND embedding IS NOT NULL AND ticker = ?",
        (ticker,),
    ).fetchall()

def insert_market_data(conn: sqlite3.Connection, *, ticker: str, date: str,
                       open: float, close: float, pct_change: float) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO market_data (ticker, date, open, close, pct_change) "
        "VALUES (?, ?, ?, ?, ?)",
        (ticker, date, open, close, pct_change),
    )
    conn.commit()

def get_market_data(conn: sqlite3.Connection, ticker: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM market_data WHERE ticker = ? ORDER BY date ASC", (ticker,)
    ).fetchall()

def get_pct_change_after(conn: sqlite3.Connection, ticker: str, date: str, days: int) -> Optional[float]:
    row = conn.execute(
        "SELECT pct_change FROM market_data WHERE ticker = ? AND date >= ? "
        "ORDER BY date ASC LIMIT 1 OFFSET ?",
        (ticker, date, days - 1),
    ).fetchone()
    return row["pct_change"] if row else None

def insert_signal(conn: sqlite3.Connection, *, headline_id: int, ticker: str,
                  signal: str, confidence: float, top_matches: str) -> None:
    conn.execute(
        "INSERT INTO signals (headline_id, ticker, signal, confidence, top_matches) "
        "VALUES (?, ?, ?, ?, ?)",
        (headline_id, ticker, signal, confidence, top_matches),
    )
    conn.commit()

def get_signals(conn: sqlite3.Connection, ticker: Optional[str] = None,
                limit: int = 100) -> list[sqlite3.Row]:
    if ticker:
        return conn.execute(
            "SELECT s.*, h.headline, h.published_at as headline_date FROM signals s "
            "JOIN headlines h ON s.headline_id = h.id "
            "WHERE s.ticker = ? ORDER BY s.created_at DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
    return conn.execute(
        "SELECT s.*, h.headline, h.published_at as headline_date FROM signals s "
        "JOIN headlines h ON s.headline_id = h.id "
        "ORDER BY s.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()

def add_to_watchlist(conn: sqlite3.Connection, ticker: str, asset_type: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO watchlist (ticker, asset_type, active) VALUES (?, ?, 1)",
        (ticker, asset_type),
    )
    conn.commit()

def get_watchlist(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM watchlist WHERE active = 1"
    ).fetchall()

def remove_from_watchlist(conn: sqlite3.Connection, ticker: str) -> None:
    conn.execute("UPDATE watchlist SET active = 0 WHERE ticker = ?", (ticker,))
    conn.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
poetry run pytest tests/test_db/test_repository.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```
feat(db): add SQLite schema and repository layer
```

---

## Task 3: Embeddings module

**Files:**
- Create: `src/engine/embeddings.py`
- Create: `tests/test_engine/test_embeddings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine/test_embeddings.py
import numpy as np
from src.engine.embeddings import embed_text, serialise, deserialise

def test_embed_returns_correct_shape():
    vec = embed_text("Apple beats Q3 earnings expectations")
    assert vec.shape == (384,)
    assert vec.dtype == np.float32

def test_serialise_deserialise_roundtrip():
    vec = embed_text("Fed raises interest rates by 50 basis points")
    blob = serialise(vec)
    assert isinstance(blob, bytes)
    restored = deserialise(blob)
    np.testing.assert_array_almost_equal(vec, restored)

def test_different_texts_produce_different_vectors():
    v1 = embed_text("Apple beats earnings")
    v2 = embed_text("Tesla recalls 10000 vehicles")
    assert not np.allclose(v1, v2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_engine/test_embeddings.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement src/engine/embeddings.py**

```python
# src/engine/embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_text(text: str) -> np.ndarray:
    vec = _get_model().encode(text, convert_to_numpy=True)
    return vec.astype(np.float32)

def serialise(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()

def deserialise(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_engine/test_embeddings.py -v
```

Expected: 3 tests PASS. First run downloads ~90MB model — normal.

- [ ] **Step 5: Commit**

```
feat(engine): add sentence-transformer embeddings module
```

---

## Task 4: Sentiment module

**Files:**
- Create: `src/engine/sentiment.py`
- Create: `tests/test_engine/test_sentiment.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine/test_sentiment.py
import numpy as np
from src.engine.sentiment import score_sentiment

def test_returns_four_floats():
    scores = score_sentiment("Apple crushes earnings, stock surges 10%")
    assert scores.shape == (4,)

def test_positive_headline_has_positive_compound():
    scores = score_sentiment("Record profits send stock to all-time high")
    compound = scores[0]
    assert compound > 0.0

def test_negative_headline_has_negative_compound():
    scores = score_sentiment("Company faces massive fraud scandal and bankruptcy")
    compound = scores[0]
    assert compound < 0.0

def test_scores_in_valid_range():
    scores = score_sentiment("Market opens flat ahead of Fed decision")
    assert all(-1.0 <= s <= 1.0 for s in scores)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_engine/test_sentiment.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement src/engine/sentiment.py**

```python
# src/engine/sentiment.py
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

def score_sentiment(text: str) -> np.ndarray:
    """Returns [compound, pos, neg, neu] as float32 array."""
    s = _analyzer.polarity_scores(text)
    return np.array([s["compound"], s["pos"], s["neg"], s["neu"]], dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_engine/test_sentiment.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```
feat(engine): add VADER sentiment scoring module
```

---

## Task 5: Similarity matcher

**Files:**
- Create: `src/engine/matcher.py`
- Create: `tests/test_engine/test_matcher.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine/test_matcher.py
import numpy as np
import sqlite3
import pytest
from src.db.models import init_db
from src.db.repository import insert_headline, set_headline_embedding, set_headline_processed, insert_market_data
from src.engine.embeddings import embed_text, serialise
from src.engine.matcher import find_top_matches, Match

@pytest.fixture
def db_with_history(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    headlines = [
        ("AAPL", "Apple beats Q3 earnings by wide margin", "2023-06-01T10:00:00", "2023-06-01", 4.2),
        ("AAPL", "Apple misses revenue estimates in Q2", "2023-03-01T10:00:00", "2023-03-01", -3.1),
        ("AAPL", "Apple launches new iPhone with record pre-orders", "2023-09-01T10:00:00", "2023-09-01", 2.8),
    ]
    for ticker, hl, pub, date, pct in headlines:
        hid = insert_headline(conn, ticker=ticker, source="rss", headline=hl, url=None, published_at=pub)
        set_headline_embedding(conn, hid, serialise(embed_text(hl)))
        set_headline_processed(conn, hid)
        insert_market_data(conn, ticker=ticker, date=date, open=150.0, close=156.0, pct_change=pct)
    return conn

def test_returns_top_k_matches(db_with_history):
    query_vec = embed_text("Apple reports strong quarterly results")
    matches = find_top_matches(db_with_history, query_vec, ticker="AAPL", top_k=2)
    assert len(matches) == 2

def test_match_has_required_fields(db_with_history):
    query_vec = embed_text("Apple reports strong quarterly results")
    matches = find_top_matches(db_with_history, query_vec, ticker="AAPL", top_k=1)
    m = matches[0]
    assert isinstance(m, Match)
    assert 0.0 <= m.similarity <= 1.0
    assert m.pct_change is not None
    assert isinstance(m.headline, str)

def test_most_similar_headline_ranked_first(db_with_history):
    query_vec = embed_text("Apple crushes quarterly earnings estimates")
    matches = find_top_matches(db_with_history, query_vec, ticker="AAPL", top_k=3)
    similarities = [m.similarity for m in matches]
    assert similarities == sorted(similarities, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_engine/test_matcher.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement src/engine/matcher.py**

```python
# src/engine/matcher.py
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

def find_top_matches(conn: sqlite3.Connection, query_vec: np.ndarray,
                     ticker: str, top_k: int = 3) -> list[Match]:
    rows = get_processed_headlines_with_embeddings(conn, ticker)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_engine/test_matcher.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```
feat(engine): add cosine similarity matcher
```

---

## Task 6: Feature vector assembly

**Files:**
- Create: `src/engine/features.py`
- Create: `tests/test_engine/test_features.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine/test_features.py
import numpy as np
import pytest
from src.engine.matcher import Match
from src.engine.features import build_feature_vector, FEATURE_DIM

def _make_matches(n: int) -> list[Match]:
    return [
        Match(headline_id=i, headline=f"Headline {i}", date="2024-01-01",
              similarity=0.9 - i * 0.1, pct_change=float(i))
        for i in range(n)
    ]

def test_feature_vector_has_correct_dimension():
    embedding = np.random.rand(384).astype(np.float32)
    matches = _make_matches(3)
    vec = build_feature_vector(embedding, matches)
    assert vec.shape == (FEATURE_DIM,)

def test_feature_vector_with_no_matches():
    embedding = np.random.rand(384).astype(np.float32)
    vec = build_feature_vector(embedding, [])
    assert vec.shape == (FEATURE_DIM,)
    # similarity and pct_change features should be zero-padded
    assert not np.any(np.isnan(vec))

def test_feature_vector_with_fewer_than_three_matches():
    embedding = np.random.rand(384).astype(np.float32)
    matches = _make_matches(1)
    vec = build_feature_vector(embedding, matches)
    assert vec.shape == (FEATURE_DIM,)
    assert not np.any(np.isnan(vec))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_engine/test_features.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement src/engine/features.py**

```python
# src/engine/features.py
import numpy as np
from src.engine.matcher import Match
from src.engine.sentiment import score_sentiment

# 384 (embedding) + 3 (similarity scores) + 9 (pct_change 1d/3d/7d per match) + 4 (sentiment) = 400
FEATURE_DIM = 400
_TOP_K = 3

def build_feature_vector(embedding: np.ndarray, matches: list[Match],
                          headline: str = "") -> np.ndarray:
    similarity_feats = np.zeros(_TOP_K, dtype=np.float32)
    pct_feats = np.zeros(_TOP_K * 3, dtype=np.float32)  # placeholder: 1d/3d/7d per match

    for i, m in enumerate(matches[:_TOP_K]):
        similarity_feats[i] = m.similarity
        pct_val = m.pct_change if m.pct_change is not None else 0.0
        # all three windows use pct_change (3d); extend when more windows added to DB
        pct_feats[i * 3 : i * 3 + 3] = pct_val

    sentiment_feats = score_sentiment(headline) if headline else np.zeros(4, dtype=np.float32)

    return np.concatenate([embedding, similarity_feats, pct_feats, sentiment_feats])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_engine/test_features.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```
feat(engine): add feature vector assembly
```

---

## Task 7: MLPClassifier model (train + infer)

**Files:**
- Create: `src/engine/model.py`
- Create: `tests/test_engine/test_model.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine/test_model.py
import numpy as np
import pytest
from pathlib import Path
from src.engine.model import SignalModel, LABELS

def _make_dataset(n: int = 60):
    X = np.random.rand(n, 400).astype(np.float32)
    y = np.array([LABELS[i % 3] for i in range(n)])
    return X, y

def test_labels_are_correct():
    assert set(LABELS) == {"BUY", "SELL", "HOLD"}

def test_train_and_predict(tmp_path):
    model = SignalModel()
    X, y = _make_dataset()
    model.train(X, y)
    signal, confidence = model.predict(X[0])
    assert signal in LABELS
    assert 0.0 <= confidence <= 1.0

def test_save_and_load(tmp_path):
    model_path = str(tmp_path / "model.pkl")
    model = SignalModel()
    X, y = _make_dataset()
    model.train(X, y)
    model.save(model_path)

    loaded = SignalModel.load(model_path)
    signal, confidence = loaded.predict(X[0])
    assert signal in LABELS

def test_predict_raises_if_not_trained():
    model = SignalModel()
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict(np.zeros(400, dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_engine/test_model.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement src/engine/model.py**

```python
# src/engine/model.py
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

LABELS = ["BUY", "SELL", "HOLD"]

class SignalModel:
    def __init__(self) -> None:
        self._clf: MLPClassifier | None = None
        self._le = LabelEncoder().fit(LABELS)

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        self._clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
        )
        self._clf.fit(X, y)
        preds = self._clf.predict(X)
        accuracy = float((preds == y).mean())
        return {"accuracy": accuracy, "n_samples": len(y)}

    def predict(self, x: np.ndarray) -> tuple[str, float]:
        if self._clf is None:
            raise RuntimeError("Model not trained — call train() or load() first")
        proba = self._clf.predict_proba(x.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        classes = list(self._clf.classes_)
        signal = classes[idx]
        confidence = float(proba[idx])
        return signal, confidence

    def save(self, path: str) -> None:
        joblib.dump(self._clf, path)

    @classmethod
    def load(cls, path: str) -> "SignalModel":
        instance = cls()
        instance._clf = joblib.load(path)
        return instance
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_engine/test_model.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```
feat(engine): add MLPClassifier signal model
```

---

## Task 8: Signal orchestration pipeline

**Files:**
- Create: `src/engine/signals.py`
- Create: `src/engine/runner.py`
- Create: `tests/test_engine/test_signals.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine/test_signals.py
import json
import pytest
import numpy as np
from src.db.models import init_db
from src.db.repository import (
    insert_headline, insert_market_data, set_headline_embedding,
    set_headline_processed, get_signals, get_pending_headlines
)
from src.engine.embeddings import embed_text, serialise
from src.engine.model import SignalModel, LABELS
from src.engine.signals import process_headline

@pytest.fixture
def db_with_model(tmp_path):
    conn = init_db(str(tmp_path / "test.db"))
    # seed 3 processed headlines with market data
    for i, hl in enumerate(["Apple beats earnings", "Apple misses revenue", "Apple launches product"]):
        hid = insert_headline(conn, ticker="AAPL", source="rss",
                              headline=hl, url=None, published_at=f"2023-0{i+1}-01T10:00:00")
        set_headline_embedding(conn, hid, serialise(embed_text(hl)))
        set_headline_processed(conn, hid)
        insert_market_data(conn, ticker="AAPL", date=f"2023-0{i+1}-01",
                           open=150.0, close=155.0, pct_change=float(i * 2 - 2))
    # untrained model (predict will use random-init, but structure is valid after train)
    X = np.random.rand(30, 400).astype(np.float32)
    y = np.array(["BUY", "SELL", "HOLD"] * 10)
    model = SignalModel()
    model.train(X, y)
    return conn, model

def test_process_headline_creates_signal(db_with_model):
    conn, model = db_with_model
    hid = insert_headline(conn, ticker="AAPL", source="rss",
                          headline="Apple reports record quarter",
                          url=None, published_at="2024-01-01T10:00:00")
    process_headline(conn, model, headline_id=hid)
    sigs = get_signals(conn, ticker="AAPL")
    assert len(sigs) == 1
    assert sigs[0]["signal"] in LABELS
    assert 0.0 <= sigs[0]["confidence"] <= 1.0

def test_process_headline_marks_as_processed(db_with_model):
    conn, model = db_with_model
    hid = insert_headline(conn, ticker="AAPL", source="rss",
                          headline="Apple launches new product line",
                          url=None, published_at="2024-01-02T10:00:00")
    process_headline(conn, model, headline_id=hid)
    assert len(get_pending_headlines(conn)) == 0

def test_process_headline_stores_top_matches(db_with_model):
    conn, model = db_with_model
    hid = insert_headline(conn, ticker="AAPL", source="rss",
                          headline="Apple sets revenue record",
                          url=None, published_at="2024-01-03T10:00:00")
    process_headline(conn, model, headline_id=hid)
    sig = get_signals(conn, ticker="AAPL")[0]
    matches = json.loads(sig["top_matches"])
    assert isinstance(matches, list)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_engine/test_signals.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement src/engine/signals.py**

```python
# src/engine/signals.py
import json
import sqlite3
from src.db.repository import (
    insert_signal, set_headline_processed, set_headline_embedding,
    get_pending_headlines,
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

    matches = find_top_matches(conn, embedding, ticker=row["ticker"], top_k=3)
    features = build_feature_vector(embedding, matches, headline=row["headline"])

    signal, confidence = model.predict(features)

    top_matches_json = json.dumps([
        {"headline": m.headline, "date": m.date,
         "similarity": round(m.similarity, 4),
         "pct_change": m.pct_change}
        for m in matches
    ])

    insert_signal(conn, headline_id=headline_id, ticker=row["ticker"],
                  signal=signal, confidence=confidence, top_matches=top_matches_json)
    set_headline_processed(conn, headline_id)
```

- [ ] **Step 4: Implement src/engine/runner.py**

```python
# src/engine/runner.py
import os
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
poetry run pytest tests/test_engine/test_signals.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```
feat(engine): add signal orchestration pipeline
```

---

## Task 9: Training CLI

**Files:**
- Create: `src/train.py`
- Modify: `pyproject.toml` (add script entry point)

- [ ] **Step 1: Implement src/train.py**

```python
# src/train.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from src.db.models import init_db
from src.db.repository import get_processed_headlines_with_embeddings, get_pct_change_after
from src.engine.embeddings import embed_text, deserialise
from src.engine.matcher import find_top_matches
from src.engine.features import build_feature_vector
from src.engine.model import SignalModel

load_dotenv()

BUY_THRESHOLD = 2.0
SELL_THRESHOLD = -2.0

def _label(pct: float | None) -> str | None:
    if pct is None:
        return None
    if pct >= BUY_THRESHOLD:
        return "BUY"
    if pct <= SELL_THRESHOLD:
        return "SELL"
    return "HOLD"

def build_training_data(conn) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        "SELECT * FROM headlines WHERE status = 'processed' AND embedding IS NOT NULL"
    ).fetchall()

    X, y = [], []
    for row in rows:
        pct = get_pct_change_after(conn, row["ticker"], row["published_at"][:10], days=3)
        label = _label(pct)
        if label is None:
            continue
        embedding = deserialise(row["embedding"])
        matches = find_top_matches(conn, embedding, ticker=row["ticker"], top_k=3)
        features = build_feature_vector(embedding, matches, headline=row["headline"])
        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)

def main():
    db_path = os.getenv("DB_PATH", "data/trading.db")
    model_path = os.getenv("MODEL_PATH", "models/signal_classifier.pkl")

    print(f"Loading data from {db_path} ...")
    conn = init_db(db_path)
    X, y = build_training_data(conn)

    if len(X) < 10:
        print(f"Not enough training samples ({len(X)}). Need at least 10.")
        return

    print(f"Training on {len(X)} samples ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = SignalModel()
    metrics = model.train(X_train, y_train)
    print(f"Train accuracy: {metrics['accuracy']:.3f}")

    preds = [model.predict(x)[0] for x in X_test]
    print(classification_report(y_test, preds, target_names=["BUY", "SELL", "HOLD"]))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
    fig.savefig("models/confusion_matrix.png", bbox_inches="tight")
    print("Confusion matrix saved to models/confusion_matrix.png")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add script entry point to pyproject.toml**

Add under `[tool.poetry]`:

```toml
[tool.poetry.scripts]
train = "src.train:main"
```

- [ ] **Step 3: Verify entry point works**

```bash
poetry run train
```

Expected: "Not enough training samples (0). Need at least 10." — correct, DB is empty.

- [ ] **Step 4: Commit**

```
feat(engine): add training CLI with confusion matrix output
```

---

## Task 10: RSS ingestion scraper

**Files:**
- Create: `src/ingestion/scrapers/rss.py`

- [ ] **Step 1: Implement src/ingestion/scrapers/rss.py**

```python
# src/ingestion/scrapers/rss.py
import hashlib
import sqlite3
from datetime import datetime, timezone
import feedparser
from src.db.repository import insert_headline

RSS_FEEDS = {
    "default": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        "https://feeds.reuters.com/reuters/businessNews",
    ]
}

def _normalize_date(entry) -> str:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    return datetime.now(timezone.utc).isoformat()

def scrape_rss(conn: sqlite3.Connection, ticker: str) -> int:
    inserted = 0
    urls = [u.format(ticker=ticker) for u in RSS_FEEDS["default"]]
    for url in urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            if not title:
                continue
            if ticker.upper() not in title.upper() and ticker.lower() not in title.lower():
                continue
            link = entry.get("link", "")
            published_at = _normalize_date(entry)
            rowid = insert_headline(
                conn, ticker=ticker, source="rss",
                headline=title, url=link, published_at=published_at
            )
            if rowid:
                inserted += 1
    return inserted
```

- [ ] **Step 2: Smoke test manually**

```bash
poetry run python -c "
from src.db.models import init_db
from src.ingestion.scrapers.rss import scrape_rss
conn = init_db('data/test_rss.db')
n = scrape_rss(conn, 'AAPL')
print(f'Inserted {n} headlines')
"
```

Expected: Inserted N headlines (>0 if network available, 0 is also fine if feeds are empty).

- [ ] **Step 3: Commit**

```
feat(ingestion): add RSS scraper for Yahoo Finance and Reuters feeds
```

---

## Task 11: Reddit ingestion scraper

**Files:**
- Create: `src/ingestion/scrapers/reddit.py`
- Create: `.env` (local only, gitignored)

- [ ] **Step 1: Create .env from example**

```bash
cp .env.example .env
# Fill in REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET
# Get free credentials at: https://www.reddit.com/prefs/apps (script type)
```

- [ ] **Step 2: Implement src/ingestion/scrapers/reddit.py**

```python
# src/ingestion/scrapers/reddit.py
import os
import sqlite3
from datetime import datetime, timezone
import praw
from src.db.repository import insert_headline

SUBREDDITS = ["wallstreetbets", "investing", "stocks"]

def _get_reddit() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.getenv("REDDIT_USER_AGENT", "signal-engine/1.0"),
    )

def scrape_reddit(conn: sqlite3.Connection, ticker: str, limit: int = 25) -> int:
    reddit = _get_reddit()
    inserted = 0
    for sub in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.search(ticker, sort="new", limit=limit):
                title = post.title.strip()
                published_at = datetime.fromtimestamp(
                    post.created_utc, tz=timezone.utc
                ).isoformat()
                url = f"https://reddit.com{post.permalink}"
                rowid = insert_headline(
                    conn, ticker=ticker, source="reddit",
                    headline=title, url=url, published_at=published_at
                )
                if rowid:
                    inserted += 1
        except Exception as e:
            print(f"Reddit scrape failed for r/{sub}: {e}")
    return inserted
```

- [ ] **Step 3: Smoke test manually**

```bash
poetry run python -c "
from dotenv import load_dotenv; load_dotenv()
from src.db.models import init_db
from src.ingestion.scrapers.reddit import scrape_reddit
conn = init_db('data/test_reddit.db')
n = scrape_reddit(conn, 'AAPL', limit=5)
print(f'Inserted {n} headlines')
"
```

Expected: Inserted N headlines.

- [ ] **Step 4: Commit**

```
feat(ingestion): add Reddit scraper for WSB and investing subreddits
```

---

## Task 12: Market data fetcher + ingestion scheduler

**Files:**
- Create: `src/ingestion/market_data.py`
- Create: `src/ingestion/scheduler.py`

- [ ] **Step 1: Implement src/ingestion/market_data.py**

```python
# src/ingestion/market_data.py
import sqlite3
import yfinance as yf
import pandas as pd
from src.db.repository import insert_market_data

def fetch_and_store_market_data(conn: sqlite3.Connection, ticker: str, period: str = "2y") -> int:
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        return 0
    stored = 0
    for date, row in hist.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        pct_change = float((row["Close"] - row["Open"]) / row["Open"] * 100) if row["Open"] else 0.0
        insert_market_data(
            conn, ticker=ticker,
            date=date_str, open=float(row["Open"]),
            close=float(row["Close"]), pct_change=pct_change,
        )
        stored += 1
    return stored
```

- [ ] **Step 2: Implement src/ingestion/scheduler.py**

```python
# src/ingestion/scheduler.py
import os
import sqlite3
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from src.db.models import init_db
from src.db.repository import get_watchlist
from src.ingestion.scrapers.rss import scrape_rss
from src.ingestion.scrapers.reddit import scrape_reddit
from src.ingestion.market_data import fetch_and_store_market_data

load_dotenv()

def run_ingestion(conn: sqlite3.Connection) -> None:
    tickers = [r["ticker"] for r in get_watchlist(conn)]
    if not tickers:
        print("Watchlist is empty, skipping ingestion.")
        return
    for ticker in tickers:
        rss_n = scrape_rss(conn, ticker)
        reddit_n = scrape_reddit(conn, ticker)
        market_n = fetch_and_store_market_data(conn, ticker)
        print(f"{ticker}: +{rss_n} RSS, +{reddit_n} Reddit, +{market_n} market rows")

def main() -> None:
    db_path = os.getenv("DB_PATH", "data/trading.db")
    conn = init_db(db_path)

    scheduler = BlockingScheduler()
    scheduler.add_job(run_ingestion, "interval", minutes=15, args=[conn])
    print("Ingestion scheduler started (every 15 min). Press Ctrl+C to stop.")
    run_ingestion(conn)  # run immediately on start
    scheduler.start()

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add script entry point to pyproject.toml**

```toml
[tool.poetry.scripts]
train = "src.train:main"
ingest = "src.ingestion.scheduler:main"
```

- [ ] **Step 4: Smoke test**

```bash
poetry run ingest
```

Expected: "Watchlist is empty, skipping ingestion." (correct — no tickers added yet).

- [ ] **Step 5: Commit**

```
feat(ingestion): add market data fetcher and APScheduler ingestion loop
```

---

## Task 13: Full test suite run

- [ ] **Step 1: Run all tests**

```bash
poetry run pytest tests/ -v --tb=short
```

Expected: All tests PASS. (Ingestion scrapers have no unit tests — they use external APIs; smoke tests covered manually in previous tasks.)

- [ ] **Step 2: Run type check**

```bash
poetry run mypy src/ --ignore-missing-imports
```

Expected: No errors, or only minor missing stub warnings (acceptable).

- [ ] **Step 3: Run linter**

```bash
poetry run ruff check src/ tests/
```

Expected: No errors.

- [ ] **Step 4: Final commit**

```
chore: all tests passing, linting clean
```
