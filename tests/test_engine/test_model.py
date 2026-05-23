import numpy as np
import pytest
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
