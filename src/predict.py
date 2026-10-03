from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import clean_text, normalize_whitespace


class ReviewPredictor:
    def __init__(self, model, vectorizer, label_map=None):
        self.model = model
        self.vectorizer = vectorizer
        self.label_map = label_map or {0: "Negative", 1: "Positive"}

    def predict_single(self, review: str):
        if review is None or not isinstance(review, str):
            raise ValueError("Review text must be a non-empty string.")

        cleaned = normalize_whitespace(clean_text(review))
        if not cleaned:
            raise ValueError("Review text cannot be empty after cleaning.")

        features = self.vectorizer.transform([cleaned])
        pred = int(self.model.predict(features)[0])
        result = {
            "sentiment": self.label_map.get(pred, str(pred)),
            "predicted_label": pred,
        }

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            pred_idx = int(np.argmax(proba))
            conf = float(proba[pred_idx])
            result["confidence"] = conf
            result["probabilities"] = {self.label_map.get(i, str(i)): float(p) for i, p in enumerate(proba)}
        else:
            decision = float(self.model.decision_function(features)[0])
            result["decision_score"] = decision
            result["confidence"] = abs(decision)
            result["confidence_label"] = "Decision score from LinearSVC"
        return result

    def predict_batch(self, reviews: list[str]):
        records = []
        for review in reviews:
            records.append(self.predict_single(review))
        return records


def build_prediction_dataframe(review_texts: list[str], predictions: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame({"review_text": review_texts, "sentiment": [p["sentiment"] for p in predictions]})
    if "confidence" in predictions[0]:
        df["confidence"] = [p.get("confidence", 0.0) for p in predictions]
    if "decision_score" in predictions[0]:
        df["decision_score"] = [p.get("decision_score", 0.0) for p in predictions]
    return df
