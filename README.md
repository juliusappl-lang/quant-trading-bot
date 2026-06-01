# Quant Trading Bot — News-Driven Signal Engine

A trading signal engine that scrapes financial headlines from RSS feeds and yfinance, matches them against a historical database of market-moving events, and uses a locally trained gradient boosting classifier to generate **BUY / SELL / HOLD** signals for stocks and crypto.

Fully local. No paid APIs. No cloud dependencies.

---

## Architecture

Four services coordinated via a shared SQLite database:

```
┌─────────────────┐     SQLite      ┌─────────────────┐     SQLite     ┌──────────────────┐
│   Ingestion     │ ── headlines ──▶│  Signal Engine  │ ── signals ──▶│  API + Dashboard │
│                 │                 │                 │                │                  │
│  RSS scraper    │                 │  Embeddings     │                │  FastAPI REST    │
│  yfinance news  │                 │  Similarity     │                │  Streamlit UI    │
│  Earnings data  │                 │  GBM Classifier │                │                  │
└─────────────────┘                 └─────────────────┘                └──────────────────┘
```

When a ticker is added to the watchlist, a one-time **historical bootstrap** runs automatically:
- Full price history fetched via yfinance (`period="max"`)
- Recent news headlines scraped from yfinance
- Synthetic earnings event headlines generated from quarterly EPS data

---

## ML Pipeline

```
Headline text
    │
    ├── sentence-transformers (all-MiniLM-L6-v2) → 384-dim embedding
    ├── Cosine similarity → Top-3 historical matches (self excluded) + 3-day price outcome
    └── VADER sentiment → 4-dim score
          │
          ▼
    394-dim feature vector → GradientBoostingClassifier → BUY / SELL / HOLD + confidence
```

**Label strategy:** Thresholds are computed per ticker using the 25th/75th percentile of 3-day forward returns, producing a balanced ~25/50/25 BUY/HOLD/SELL split. Sample weights balance classes during training. Model is evaluated with 5-fold stratified cross-validation.

**Relevance filter:** Every headline is matched against the ticker's name aliases (e.g. "Nvidia", "NVDA") using word-boundary regex before insertion — generic market articles that merely mention a ticker in passing are rejected.

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

Edit `DB_PATH` and `MODEL_PATH` if needed — defaults work out of the box.

### 3. Create required directories

```bash
mkdir -p data models
```

---

## Running

### One command (recommended)

```bash
poetry run python main.py
```

Starts all four services in parallel, frees ports 8000/8501 if occupied, and auto-restarts any service that crashes. Press Ctrl+C to stop everything.

### Manual (four terminal tabs)

```bash
# Tab 1 — Ingestion (scrapes RSS feeds every 15 min)
poetry run python -m src.ingestion.scheduler

# Tab 2 — Signal Engine (processes pending headlines)
poetry run python -m src.engine.runner

# Tab 3 — API
poetry run python -m src.api.server

# Tab 4 — Dashboard
poetry run streamlit run src/dashboard/app.py
```

### Train the model

After adding tickers and collecting headlines (needs ≥10 processed samples):

```bash
poetry run python -m src.train
```

Outputs 5-fold CV accuracy, per-class precision/recall, saves `models/signal_classifier.pkl` and `models/thresholds.json`.

The signal engine runner will automatically load the model if it exists, and wait gracefully if it doesn't.

---

## Watchlist

Adding a ticker triggers a full historical bootstrap automatically:

| Step | What happens |
|---|---|
| 1 | Full price history fetched via yfinance (up to 45 years for major stocks) |
| 2 | Recent news headlines scraped and relevance-filtered |
| 3 | Synthetic earnings event headlines generated (quarterly EPS beats/misses) |

This ensures newly added tickers have training data immediately available.

---

## Dashboard

Open **http://localhost:8501**

| Page | Description |
|---|---|
| Live Signals | BUY/SELL/HOLD signals with confidence scores, color-coded, filterable by ticker and signal type |
| Watchlist | Add/remove tickers — triggers historical bootstrap on add |
| Signal History | Price chart with BUY/SELL markers at actual headline publish dates |
| Model | Train the model, view 5-fold CV results and confusion matrix |

---

## REST API

Open **http://localhost:8000/docs** for interactive Swagger UI.

| Method | Endpoint | Description |
|---|---|---|
| GET | /api/signals | All recent signals |
| GET | /api/signals/{ticker} | Signals for one ticker |
| GET | /api/watchlist | Active watchlist |
| POST | /api/watchlist | Add ticker — triggers historical ingestion in background |
| DELETE | /api/watchlist/{ticker} | Remove ticker |
| POST | /api/train | Trigger model training |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, local) |
| Sentiment | VADER (local, no API) |
| ML Model | scikit-learn GradientBoostingClassifier (max_depth=3, subsample=0.8) |
| Feature Engineering | pandas + numpy (394-dim vectors) |
| Charts | matplotlib |
| Database | SQLite (stdlib) |
| News Sources | RSS — Yahoo Finance, MarketWatch (×2), WSJ Markets, Bloomberg, Investing.com, CNBC (×2), Seeking Alpha |
| Historical News | yfinance news + synthetic earnings events (lxml) |
| Market Data | yfinance |
| API | FastAPI + uvicorn |
| Dashboard | Streamlit |
| Scheduler | APScheduler |

---

## Testing

```bash
poetry run pytest tests/ -v
```

---

## Project Structure

```
src/
├── db/              # SQLite schema + repository layer
├── engine/          # Embeddings, matcher, sentiment, features, model, signals, runner
├── ingestion/       # RSS scraper, yfinance news, historical bootstrap, market data, scheduler
├── api/             # FastAPI routes + server
├── dashboard/       # Streamlit pages
└── train.py         # Training CLI

models/
├── signal_classifier.pkl   # Trained GBM model
└── thresholds.json         # Per-ticker BUY/SELL percentile thresholds

data/
└── trading.db              # SQLite database

tests/
├── test_db/
├── test_engine/
└── test_api/

docs/superpowers/
├── specs/           # Design documents
└── plans/           # Implementation plans
```
