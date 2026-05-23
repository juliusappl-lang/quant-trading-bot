# Quant Trading Bot — News-Driven Signal Engine

A trading signal engine that scrapes financial headlines from RSS feeds and Reddit, matches them against a historical database of market-moving events, and uses a locally trained neural network (scikit-learn MLPClassifier) to generate **BUY / SELL / HOLD** signals for stocks and crypto.

Fully local. No paid APIs. No cloud dependencies.

---

## Architecture

Three independent services coordinated via a shared SQLite database:

```
┌─────────────────┐     SQLite      ┌─────────────────┐     SQLite     ┌──────────────────┐
│   Ingestion     │ ── headlines ──▶│  Signal Engine  │ ── signals ──▶│  API + Dashboard │
│                 │                 │                 │                │                  │
│  RSS scraper    │                 │  Embeddings     │                │  FastAPI REST    │
│  Reddit scraper │                 │  Similarity     │                │  Streamlit UI    │
│  yfinance data  │                 │  MLPClassifier  │                │                  │
└─────────────────┘                 └─────────────────┘                └──────────────────┘
```

## ML Pipeline

```
Headline text
    │
    ├── sentence-transformers (all-MiniLM-L6-v2) → 384-dim embedding
    ├── Cosine similarity → Top-3 historical matches with market outcomes
    └── VADER sentiment → 4-dim score
          │
          ▼
    ~400-dim feature vector → MLPClassifier → BUY / SELL / HOLD + confidence
```

---

## Setup

### 1. Install dependencies

```bash
poetry install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in your Reddit API credentials (free at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) — choose **script** type).

### 3. Create required directories

```bash
mkdir -p data models
```

---

## Running

Open four terminal tabs:

```bash
# Tab 1 — Ingestion (scrapes RSS + Reddit every 15 min)
poetry run python -m src.ingestion.scheduler

# Tab 2 — Signal Engine (processes pending headlines)
poetry run python -m src.engine.runner

# Tab 3 — API
poetry run python -m src.api.server

# Tab 4 — Dashboard
poetry run python -m streamlit run src/dashboard/app.py
```

### Train the model

After collecting headlines (needs ≥10 processed samples with market data):

```bash
poetry run python -m src.train
```

---

## Dashboard

Open **http://localhost:8501**

| Page | Description |
|---|---|
| Live Signals | BUY/SELL/HOLD signals with confidence scores, color-coded |
| Watchlist | Add/remove tickers (stocks and crypto) dynamically |
| Signal History | Price chart with signal markers + historical signal table |
| Model | Trigger training, view accuracy and confusion matrix |

## REST API

Open **http://localhost:8000/docs** for interactive Swagger UI.

| Method | Endpoint | Description |
|---|---|---|
| GET | /api/signals | All recent signals |
| GET | /api/signals/{ticker} | Signals for one ticker |
| GET | /api/watchlist | Active watchlist |
| POST | /api/watchlist | Add ticker `{"ticker": "AAPL", "asset_type": "stock"}` |
| DELETE | /api/watchlist/{ticker} | Remove ticker |
| POST | /api/train | Trigger model training |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, local) |
| Sentiment | VADER (local, no API) |
| ML Model | scikit-learn MLPClassifier |
| Feature Engineering | pandas + numpy |
| Charts | matplotlib |
| Database | SQLite (stdlib) |
| News Sources | RSS (Yahoo Finance, Reuters) + Reddit (PRAW) |
| Market Data | yfinance |
| API | FastAPI + uvicorn |
| Dashboard | Streamlit |
| Scheduler | APScheduler |

## Testing

```bash
poetry run pytest tests/ -v
```

## Project Structure

```
src/
├── db/              # SQLite schema + repository layer
├── engine/          # Embeddings, matcher, sentiment, features, model, signals
├── ingestion/       # RSS + Reddit scrapers, market data fetcher, scheduler
├── api/             # FastAPI routes + server
├── dashboard/       # Streamlit pages
└── train.py         # Training CLI

tests/
├── test_db/
├── test_engine/
└── test_api/

docs/superpowers/
├── specs/           # Design documents
└── plans/           # Implementation plans
```
