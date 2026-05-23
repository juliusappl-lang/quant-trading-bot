import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_sentiment(text: str) -> np.ndarray:
    """Returns [compound, pos, neg, neu] as float32 array."""
    s = _analyzer.polarity_scores(text)
    return np.array([s["compound"], s["pos"], s["neg"], s["neu"]], dtype=np.float32)
