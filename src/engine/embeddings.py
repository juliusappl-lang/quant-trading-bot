import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> np.ndarray:
    vec = _get_model().encode(text, convert_to_numpy=True)
    return vec.astype(np.float32)


def serialise(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def deserialise(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
