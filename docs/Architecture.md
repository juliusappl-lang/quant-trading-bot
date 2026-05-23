# News-Driven Trading Signal Engine — Design Spec


---

## Overview

A trading signal engine that scrapes financial news headlines and social media, matches them against a historical database of headlines and market movements, and uses a locally trained neural network (scikit-learn MLPClassifier) to generate BUY/SELL/HOLD signals for stocks and crypto.

The system is fully local, requires no paid APIs, and is designed as a professional showcase project.

---

## Architecture

Three independently runnable services coordinated via a shared SQLite database:

```
┌─────────────────┐     SQLite      ┌─────────────────┐     SQLite      ┌─────────────────┐
│  Ingestion      │ ─── headlines ─▶│  Signal Engine  │ ─── signals ──▶│  API + Dashboard│
│  Service        │                 │                 │                 │                 │
│  (APScheduler)  │                 │  (polling loop) │                 │  FastAPI        │
│  RSS + Reddit   │                 │  Embeddings     │                 │  Streamlit      │
└─────────────────┘                 │  NN Inference   │                 └─────────────────┘
                                    │  Training CLI   │
                                    └─────────────────┘
```

**Coordination pattern:** SQLite is the shared state store. Ingestion writes `headlines` with `status=pending`. Signal Engine polls for pending headlines, processes them, writes to `signals`, sets `status=processed`. No external broker or queue needed.

---

## Project Structure

```
quant-trading-bot/
├── src/
│   ├── ingestion/
│   │   ├── scrapers/
│   │   │   ├── rss.py          # Reuters, Yahoo Finance RSS feeds
│   │   │   └── reddit.py       # r/wallstreetbets, r/investing
│   │   └── scheduler.py        # APScheduler, runs scrapers every 15min
│   │
│   ├── engine/
│   │   ├── embeddings.py       # sentence-transformers all-MiniLM-L6-v2
│   │   ├── matcher.py          # Cosine similarity search (numpy)
│   │   ├── sentiment.py        # VADER sentiment scoring
│   │   ├── features.py         # Combines embedding + similarity + sentiment → feature vector
│   │   ├── model.py            # MLPClassifier train/load/predict
│   │   └── signals.py          # Orchestrates pipeline, writes to DB
│   │
│   ├── api/
│   │   ├── routes/
│   │   │   ├── signals.py      # GET /api/signals, GET /api/signals/{ticker}
│   │   │   ├── watchlist.py    # GET/POST/DELETE /api/watchlist
│   │   │   └── train.py        # POST /api/train
│   │   └── main.py
│   │
│   ├── db/
│   │   ├── models.py           # SQLite schema definitions
│   │   ├── repository.py       # DB access layer (no raw SQL in business logic)
│   │   └── migrations/         # Versioned .sql migration files
│   │
│   └── dashboard/
│       └── app.py              # Streamlit app (4 pages)
│
├── models/
│   └── signal_classifier.pkl   # Trained MLPClassifier (joblib)
│
├── data/                       # SQLite DB files (gitignored)
├── tests/
│   ├── test_ingestion/
│   ├── test_engine/
│   └── test_api/
├── docs/
│   └── superpowers/specs/
├── notebooks/
└── pyproject.toml
```

---

## Database Schema

```sql
-- All scraped headlines
CREATE TABLE headlines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    source       TEXT NOT NULL,        -- 'rss' | 'reddit'
    headline     TEXT NOT NULL,
    url          TEXT,
    published_at DATETIME NOT NULL,
    embedding    BLOB,                 -- serialised numpy float32 array (384 dim)
    status       TEXT DEFAULT 'pending' -- 'pending' | 'processed' | 'skipped'
);

-- Historical market data fetched via yfinance
CREATE TABLE market_data (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker     TEXT NOT NULL,
    date       DATE NOT NULL,
    open       REAL,
    close      REAL,
    pct_change REAL,                   -- daily % change
    UNIQUE(ticker, date)
);

-- Generated trading signals
CREATE TABLE signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_id INTEGER REFERENCES headlines(id),
    ticker      TEXT NOT NULL,
    signal      TEXT NOT NULL,         -- 'BUY' | 'SELL' | 'HOLD'
    confidence  REAL NOT NULL,         -- 0.0 – 1.0 (softmax max probability)
    top_matches TEXT,                  -- JSON: [{headline, date, pct_change, similarity}]
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Dynamic watchlist managed via dashboard
CREATE TABLE watchlist (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker     TEXT UNIQUE NOT NULL,
    asset_type TEXT NOT NULL,          -- 'stock' | 'crypto'
    added_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    active     BOOLEAN DEFAULT 1
);
```

---

## Signal Pipeline

### Step 1 — Ingestion (every 15 min)
- RSS scrapers fetch headlines for all active watchlist tickers
- Reddit scrapers fetch top posts from r/wallstreetbets, r/investing, filtered by ticker mention
- Deduplication via URL hash before insert
- yfinance fetches/updates market_data for active tickers (2 years history)

### Step 2 — Feature Engineering (per pending headline)

| Feature Group | Dimensions | Source |
|---|---|---|
| Headline embedding | 384 | sentence-transformers all-MiniLM-L6-v2 |
| Top-3 similarity scores | 3 | Cosine similarity vs processed headlines |
| Top-3 historical pct_change (1d, 3d, 7d) | 9 | market_data join |
| VADER sentiment (compound, pos, neg, neu) | 4 | vaderSentiment |
| **Total** | **~400** | |

### Step 3 — NN Inference
```
Input(400) → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dense(3, Softmax)
```
- Implemented as `sklearn.neural_network.MLPClassifier`
- Output: class probabilities → signal = argmax, confidence = max probability
- Model loaded from `models/signal_classifier.pkl` at startup

### Step 4 — Training Pipeline
- Triggered via `poetry run train` or POST /api/train
- Labels derived from market_data: pct_change 3d after headline
  - `> +2%` → BUY, `< -2%` → SELL, else → HOLD
- Train/test split 80/20, stratified
- Metrics logged: accuracy, precision, recall per class, confusion matrix saved as PNG
- Model serialised via `joblib.dump()` → `models/signal_classifier.pkl`

---

## Dashboard (Streamlit)

**Page 1 — Live Signals**
- Table: Ticker | Signal | Confidence | Headline | Time
- Color coding: BUY=green, SELL=red, HOLD=yellow
- Filters: ticker, signal type, time range

**Page 2 — Watchlist**
- View active tickers
- Add ticker (text input + asset_type selector)
- Deactivate / delete ticker

**Page 3 — Signal History**
- Matplotlib chart: price line + BUY/SELL marker overlay
- Table: all historical signals with confidence + headline text

**Page 4 — Model**
- Trigger training (button)
- Display: accuracy, last trained timestamp, training data size
- Confusion matrix (matplotlib)

---

## REST API (FastAPI)

| Method | Endpoint | Description |
|---|---|---|
| GET | /api/signals | All recent signals |
| GET | /api/signals/{ticker} | Signals for one ticker |
| GET | /api/watchlist | Active watchlist |
| POST | /api/watchlist | Add ticker `{ticker, asset_type}` |
| DELETE | /api/watchlist/{ticker} | Remove ticker |
| POST | /api/train | Trigger model training |

---

## ML Stack

| Library | Purpose |
|---|---|
| `sentence-transformers` | Headline embeddings (all-MiniLM-L6-v2, local) |
| `vaderSentiment` | Sentiment scoring (local, no API) |
| `scikit-learn` | MLPClassifier, train/eval, metrics |
| `pandas` | Feature engineering, data pipelines |
| `numpy` | Cosine similarity computation |
| `matplotlib` | Confusion matrix, price charts |
| `joblib` | Model serialisation |

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, merges via PR only |
| `develop` | Integration branch |
| `feature/db-schema` | DB models + migrations |
| `feature/ingestion` | RSS + Reddit scrapers + scheduler |
| `feature/engine` | Embeddings, matcher, sentiment, features, model |
| `feature/api` | FastAPI routes |
| `feature/dashboard` | Streamlit app |

Each feature branch is merged into `develop` via PR. `develop` → `main` after full integration test.

---

## Dependencies to Add

```toml
sentence-transformers = "^3.0"
vaderSentiment = "^3.3"
scikit-learn = "^1.5"
streamlit = "^1.35"
fastapi = "^0.111"
uvicorn = "^0.30"
apscheduler = "^3.10"
praw = "^7.7"          # Reddit API (free credentials: reddit.com/prefs/apps)
feedparser = "^6.0"    # RSS parsing
joblib = "^1.4"
httpx = "^0.27"
```
