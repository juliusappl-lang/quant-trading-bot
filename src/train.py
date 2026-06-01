import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.db.models import init_db
from src.db.repository import get_pct_change_after
from src.engine.embeddings import deserialise
from src.engine.matcher import find_top_matches
from src.engine.features import build_feature_vector
from src.engine.model import SignalModel

load_dotenv()


def _compute_thresholds(conn, tickers: list[str]) -> dict[str, tuple[float, float]]:
    """Returns {ticker: (sell_threshold, buy_threshold)} using 25th/75th percentiles."""
    thresholds = {}
    for ticker in tickers:
        rows = conn.execute(
            "SELECT * FROM headlines WHERE status = 'processed' AND embedding IS NOT NULL AND ticker = ?",
            (ticker,),
        ).fetchall()
        pcts = []
        for row in rows:
            pct = get_pct_change_after(conn, ticker, row["published_at"][:10], days=3)
            if pct is not None:
                pcts.append(pct)
        if len(pcts) < 4:
            thresholds[ticker] = (-1.0, 1.0)
        else:
            arr = np.array(pcts)
            thresholds[ticker] = (float(np.percentile(arr, 25)), float(np.percentile(arr, 75)))
    return thresholds


def _label(pct: float, sell_thresh: float, buy_thresh: float) -> str:
    if pct >= buy_thresh:
        return "BUY"
    if pct <= sell_thresh:
        return "SELL"
    return "HOLD"


def build_training_data(conn) -> tuple[np.ndarray, np.ndarray, dict]:
    tickers = [
        r["ticker"] for r in conn.execute(
            "SELECT DISTINCT ticker FROM headlines WHERE status='processed'"
        ).fetchall()
    ]
    thresholds = _compute_thresholds(conn, tickers)

    rows = conn.execute(
        "SELECT * FROM headlines WHERE status = 'processed' AND embedding IS NOT NULL"
    ).fetchall()

    X, y = [], []
    for row in rows:
        pct = get_pct_change_after(conn, row["ticker"], row["published_at"][:10], days=3)
        if pct is None:
            continue
        sell_t, buy_t = thresholds.get(row["ticker"], (-1.0, 1.0))
        label = _label(pct, sell_t, buy_t)
        embedding = deserialise(row["embedding"])
        matches = find_top_matches(conn, embedding, ticker=row["ticker"], top_k=3, exclude_id=row["id"])
        features = build_feature_vector(embedding, matches, headline=row["headline"])
        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y), thresholds


def main():
    db_path = os.getenv("DB_PATH", "data/trading.db")
    model_path = os.getenv("MODEL_PATH", "models/signal_classifier.pkl")
    thresholds_path = "models/thresholds.json"

    print(f"Loading data from {db_path} ...")
    conn = init_db(db_path)
    X, y, thresholds = build_training_data(conn)

    if len(X) < 10:
        print(f"Not enough training samples ({len(X)}). Need at least 10.")
        return

    print(f"Training on {len(X)} samples. Label distribution: {dict(Counter(y))}")

    # 5-fold cross-validated evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = np.empty(len(y), dtype=object)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        m = SignalModel()
        sw = compute_sample_weight("balanced", y[train_idx])
        m._clf = m._build_clf()
        m._clf.fit(X[train_idx], y[train_idx], sample_weight=sw)
        fold_preds = m._clf.predict(X[val_idx])
        for i, idx in enumerate(val_idx):
            cv_preds[idx] = fold_preds[i]
        fold_acc = float((fold_preds == y[val_idx]).mean())
        print(f"  Fold {fold}: val accuracy = {fold_acc:.3f}")

    print("\n--- 5-Fold Cross-Validation Results ---")
    print(classification_report(y, cv_preds, labels=["BUY", "SELL", "HOLD"],
                                target_names=["BUY", "SELL", "HOLD"], zero_division=0))

    # Final model trained on all data
    print("Training final model on all data ...")
    sw_full = compute_sample_weight("balanced", y)
    model = SignalModel()
    model._clf = model._build_clf()
    model._clf.fit(X, y, sample_weight=sw_full)
    train_preds = model._clf.predict(X)
    print(f"Train accuracy: {float((train_preds == y).mean()):.3f}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Thresholds saved to {thresholds_path}")

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y, train_preds, ax=ax, labels=["BUY", "SELL", "HOLD"]
    )
    fig.savefig("models/confusion_matrix.png", bbox_inches="tight")
    print("Confusion matrix saved to models/confusion_matrix.png")


if __name__ == "__main__":
    main()
