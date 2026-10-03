import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.predict import ReviewPredictor


@pytest.fixture
def predictor():
    vectorizer = TfidfVectorizer()
    texts = ["great battery life", "poor battery life", "excellent screen", "bad camera"]
    labels = [1, 0, 1, 0]
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)
    return ReviewPredictor(model=model, vectorizer=vectorizer)


def test_predict_single_returns_sentiment_and_confidence(predictor):
    result = predictor.predict_single("great battery life")
    assert "sentiment" in result
    assert result["sentiment"] in {"Positive", "Negative"}
    assert "confidence" in result


def test_predict_single_rejects_empty_review(predictor):
    with pytest.raises(ValueError):
        predictor.predict_single("   ")
