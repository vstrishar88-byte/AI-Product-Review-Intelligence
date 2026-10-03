from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_amazon_polarity_subset
from src.evaluate import compute_metrics, plot_confusion_matrix, save_metrics
from src.preprocessing import build_training_frame

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def train_and_evaluate_model():
    dataset = load_amazon_polarity_subset(sample_size=20000, seed=RANDOM_STATE)
    df = build_training_frame(dataset)

    if df.empty or len(df) < 10:
        raise ValueError("Dataset is too small for training.")

    X = df["review_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    logistic_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")
    logistic_model.fit(X_train_tfidf, y_train)

    linear_svm = LinearSVC(random_state=RANDOM_STATE, class_weight="balanced")
    linear_svm.fit(X_train_tfidf, y_train)

    logistic_pred = logistic_model.predict(X_test_tfidf)
    svm_pred = linear_svm.predict(X_test_tfidf)

    metrics = {
        "logistic_regression": compute_metrics(y_test, logistic_pred, labels=[0, 1]),
        "linear_svm": compute_metrics(y_test, svm_pred, labels=[0, 1]),
    }

    best_model_name = "logistic_regression"
    if metrics["linear_svm"]["macro_f1"] > metrics["logistic_regression"]["macro_f1"]:
        best_model_name = "linear_svm"

    metadata = {
        "dataset_name": "amazon_polarity",
        "source": "Hugging Face datasets",
        "sample_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "label_meaning": {0: "negative", 1: "positive"},
        "random_state": RANDOM_STATE,
        "test_size_fraction": TEST_SIZE,
        "selected_model": best_model_name,
        "model_types": {"logistic_regression": "TF-IDF + Logistic Regression", "linear_svm": "TF-IDF + Linear SVM"},
    }

    model_artifacts = {
        "logistic_regression": logistic_model,
        "linear_svm": linear_svm,
        "tfidf_vectorizer": tfidf,
        "metadata": metadata,
    }

    joblib.dump(logistic_model, MODEL_DIR / "logistic_regression.joblib")
    joblib.dump(linear_svm, MODEL_DIR / "linear_svm.joblib")
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(metadata, MODEL_DIR / "model_metadata.joblib")
    joblib.dump(metrics[best_model_name], MODEL_DIR / "selected_model_metrics.joblib")
    joblib.dump(model_artifacts[best_model_name], MODEL_DIR / "selected_model.joblib")

    save_metrics(metrics, MODEL_DIR / "metrics.json")

    plot_confusion_matrix(confusion_matrix(y_test, logistic_pred, labels=[0, 1]), [0, 1], MODEL_DIR / "logistic_confusion_matrix.png")
    plot_confusion_matrix(confusion_matrix(y_test, svm_pred, labels=[0, 1]), [0, 1], MODEL_DIR / "svm_confusion_matrix.png")

    return model_artifacts, metrics


if __name__ == "__main__":
    _, metrics = train_and_evaluate_model()
    print(json.dumps({
        "logistic_regression": metrics["logistic_regression"],
        "linear_svm": metrics["linear_svm"],
    }, indent=2))
