import numpy as np
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
    assert not np.any(np.isnan(vec))


def test_feature_vector_with_fewer_than_three_matches():
    embedding = np.random.rand(384).astype(np.float32)
    matches = _make_matches(1)
    vec = build_feature_vector(embedding, matches)
    assert vec.shape == (FEATURE_DIM,)
    assert not np.any(np.isnan(vec))
