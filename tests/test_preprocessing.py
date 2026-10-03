import pandas as pd

from src.preprocessing import build_training_frame, clean_text, normalize_whitespace


def test_clean_text_removes_punctuation_and_lowercases():
    result = clean_text("Great! Battery life is excellent.")
    assert result == "great battery life is excellent"


def test_normalize_whitespace_collapses_spaces():
    result = normalize_whitespace("  Battery   life  is   great   ")
    assert result == "Battery life is great"


def test_build_training_frame_handles_missing_text():
    rows = [{"label": 1, "content": "The phone is durable and fast."}, {"label": 0, "content": None}]
    df = build_training_frame(rows)
    assert list(df.columns) == ["review_text", "label"]
    assert df["review_text"].str.len().gt(0).all()
    assert df["label"].isin([0, 1]).all()
