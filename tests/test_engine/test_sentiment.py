from src.engine.sentiment import score_sentiment


def test_returns_four_floats():
    scores = score_sentiment("Apple crushes earnings, stock surges 10%")
    assert scores.shape == (4,)


def test_positive_headline_has_positive_compound():
    scores = score_sentiment("Record profits send stock to all-time high")
    compound = scores[0]
    assert compound > 0.0


def test_negative_headline_has_negative_compound():
    scores = score_sentiment("Company faces massive fraud scandal and bankruptcy")
    compound = scores[0]
    assert compound < 0.0


def test_scores_in_valid_range():
    scores = score_sentiment("Market opens flat ahead of Fed decision")
    assert all(-1.0 <= s <= 1.0 for s in scores)
