import numpy as np

from src.engine.matcher import Match
from src.engine.sentiment import score_sentiment

# 384 (embedding) + 3 (similarity scores) + 9 (pct_change per match x3) + 4 (sentiment) = 400
FEATURE_DIM = 400
_TOP_K = 3


def build_feature_vector(embedding: np.ndarray, matches: list[Match],
                         headline: str = "") -> np.ndarray:
    similarity_feats = np.zeros(_TOP_K, dtype=np.float32)
    pct_feats = np.zeros(_TOP_K * 3, dtype=np.float32)

    for i, m in enumerate(matches[:_TOP_K]):
        similarity_feats[i] = m.similarity
        pct_val = m.pct_change if m.pct_change is not None else 0.0
        pct_feats[i * 3: i * 3 + 3] = pct_val

    sentiment_feats = score_sentiment(headline) if headline else np.zeros(4, dtype=np.float32)

    return np.concatenate([embedding, similarity_feats, pct_feats, sentiment_feats])
