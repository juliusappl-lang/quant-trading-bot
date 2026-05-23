import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.db.models import init_db
from src.db.repository import get_pct_change_after
from src.engine.embeddings import deserialise
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
    print(classification_report(y_test, preds, target_names=["BUY", "SELL", "HOLD"],
                                zero_division=0))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
    fig.savefig("models/confusion_matrix.png", bbox_inches="tight")
    print("Confusion matrix saved to models/confusion_matrix.png")


if __name__ == "__main__":
    main()
