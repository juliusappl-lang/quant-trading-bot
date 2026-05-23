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
