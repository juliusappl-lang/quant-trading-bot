import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier

LABELS = ["BUY", "SELL", "HOLD"]


class SignalModel:
    def __init__(self) -> None:
        self._clf: GradientBoostingClassifier | None = None

    def _build_clf(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            subsample=0.8,
            learning_rate=0.05,
            min_samples_leaf=5,
            random_state=42,
        )

    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> dict:
        self._clf = self._build_clf()
        self._clf.fit(X, y, sample_weight=sample_weight)
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
