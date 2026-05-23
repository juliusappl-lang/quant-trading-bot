import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier

LABELS = ["BUY", "SELL", "HOLD"]


class SignalModel:
    def __init__(self) -> None:
        self._clf: MLPClassifier | None = None

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
