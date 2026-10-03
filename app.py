from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.genai import GenAIInsightsClient
from src.predict import ReviewPredictor

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="AI Product Review Intelligence", page_icon="🤖", layout="wide")


def load_artifacts():
    required = [MODEL_DIR / "tfidf_vectorizer.joblib", MODEL_DIR / "model_metadata.joblib"]
    if not all(path.exists() for path in required):
        st.warning("Model files are not available yet. Please run the training pipeline first.")
        return None, None, None

    meta = joblib.load(MODEL_DIR / "model_metadata.joblib")
    log_model = joblib.load(MODEL_DIR / "logistic_regression.joblib")
    svm_model = joblib.load(MODEL_DIR / "linear_svm.joblib")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    selected_name = meta.get("selected_model", "logistic_regression")
    selected_model = joblib.load(MODEL_DIR / "selected_model.joblib") if (MODEL_DIR / "selected_model.joblib").exists() else (log_model if selected_name == "logistic_regression" else svm_model)
    return meta, {"logistic_regression": log_model, "linear_svm": svm_model, "selected_model": selected_model}, {"vectorizer": vectorizer, "metrics": metrics}


def detect_review_column(df: pd.DataFrame):
    candidates = ["review", "review_text", "text", "reviewText", "content"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def summarize_reviews(df: pd.DataFrame, predictor: ReviewPredictor):
    records = []
    for review in df["review_text"].fillna("").tolist():
        try:
            result = predictor.predict_single(review)
            records.append({"review_text": review, "sentiment": result["sentiment"], "confidence": result.get("confidence", 0.0)})
        except ValueError:
            continue

    analyzed = pd.DataFrame(records)
    if analyzed.empty:
        return analyzed, {"total_reviews": 0, "positive_pct": 0.0, "negative_pct": 0.0}

    pos_pct = (analyzed["sentiment"] == "Positive").mean() * 100
    neg_pct = (analyzed["sentiment"] == "Negative").mean() * 100
    summary = {
        "total_reviews": len(analyzed),
        "positive_pct": round(pos_pct, 2),
        "negative_pct": round(neg_pct, 2),
        "sentiment_counts": analyzed["sentiment"].value_counts().to_dict(),
    }
    return analyzed, summary


def build_genai_summary(analyzed_df: pd.DataFrame):
    if analyzed_df.empty:
        return {"error": "No data available for AI-generated insights."}

    sentiment_summary = analyzed_df["sentiment"].value_counts().to_dict()
    positive_reviews = analyzed_df[analyzed_df["sentiment"] == "Positive"]["review_text"].tolist()
    negative_reviews = analyzed_df[analyzed_df["sentiment"] == "Negative"]["review_text"].tolist()

    # Simple keyword aggregation to keep the GenAI payload lightweight and interpretable.
    positive_words = {}
    negative_words = {}
    for text in positive_reviews:
        for word in str(text).lower().split():
            if len(word) < 4:
                continue
            positive_words[word] = positive_words.get(word, 0) + 1
    for text in negative_reviews:
        for word in str(text).lower().split():
            if len(word) < 4:
                continue
            negative_words[word] = negative_words.get(word, 0) + 1

    top_positive = sorted(positive_words.items(), key=lambda x: x[1], reverse=True)[:8]
    top_negative = sorted(negative_words.items(), key=lambda x: x[1], reverse=True)[:8]
    aspects = ["battery", "performance", "price", "camera", "quality", "design", "screen", "service"]

    return {
        "total_reviews": len(analyzed_df),
        "sentiment_summary": sentiment_summary,
        "positive_themes": [item[0] for item in top_positive],
        "negative_themes": [item[0] for item in top_negative],
        "product_aspects": aspects,
        "sample_reviews": {
            "positive": positive_reviews[:3],
            "negative": negative_reviews[:3],
        },
    }


st.title("AI Product Review Intelligence")
st.caption("ML-powered sentiment analysis with GenAI customer insights")

metadata, models, artifacts = load_artifacts()
selected_model_name = metadata.get("selected_model", "logistic_regression") if metadata else "logistic_regression"
if models and "selected_model" in models:
    selected_model_name = metadata.get("selected_model", "logistic_regression")
metrics_data = artifacts["metrics"] if artifacts else {}

if metadata is None:
    st.info("Train the model first by running the training script or use the project setup instructions in the README.")
else:
    st.subheader("Model status")
    st.write(f"Selected model: {selected_model_name}")

    if not os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        st.info("GenAI insights require an API key. Set LLM_API_KEY or OPENAI_API_KEY in your environment or .env file to enable the AI summary feature.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Review Analyzer", "Batch Analysis", "Customer Insights", "Model Performance"])

    with tab1:
        st.header("Dashboard")
        # Use metadata source if available; otherwise dataset summary is not built here.
        total_reviews = metadata.get("sample_size", 0)
        st.metric("Total reviews", total_reviews)

        if metrics_data:
            st.metric("Model accuracy", round(metrics_data[selected_model_name]["accuracy"], 4))
            st.metric("Model Macro F1", round(metrics_data[selected_model_name]["macro_f1"], 4))

        if total_reviews:
            sample = pd.DataFrame({"sentiment": ["Positive", "Negative"], "count": [50, 50]})
            st.bar_chart(sample.set_index("sentiment"))

    with tab2:
        st.header("Single Review Analysis")
        review_text = st.text_area("Enter a product review", value="The battery life is excellent but the camera quality is disappointing.")
        if st.button("Analyze Review"):
            active_model = models.get(selected_model_name, models.get("selected_model", models["logistic_regression"]))
            predictor = ReviewPredictor(active_model, artifacts["vectorizer"])
            try:
                result = predictor.predict_single(review_text)
                st.success(f"Predicted sentiment: {result['sentiment']}")
                if "confidence" in result:
                    st.write(f"Confidence: {result['confidence']:.4f}")
                else:
                    st.write(f"Decision score: {result['decision_score']:.4f}")
                st.write("The model evaluates the text using the trained TF-IDF + classifier pipeline.")
            except ValueError as exc:
                st.error(str(exc))

    with tab3:
        st.header("Batch Analysis")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            review_col = detect_review_column(df)
            if review_col is None:
                st.error("No suitable review column found. Please use one of: review, review_text, text, reviewText, content.")
            else:
                batch_df = df[[review_col]].rename(columns={review_col: "review_text"})
                active_model = models.get(selected_model_name, models.get("selected_model", models["logistic_regression"]))
                predictor = ReviewPredictor(active_model, artifacts["vectorizer"])
                analyzed_df, summary = summarize_reviews(batch_df, predictor)
                if analyzed_df.empty:
                    st.warning("No valid review text was available after cleaning.")
                else:
                    st.write(summary)
                    st.dataframe(analyzed_df.head(50), use_container_width=True)
                    csv = analyzed_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download analyzed CSV", csv, file_name="analyzed_reviews.csv", mime="text/csv")

    with tab4:
        st.header("Customer Insights")
        if not os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            st.warning("Add LLM_API_KEY or OPENAI_API_KEY to enable the AI-generated insights section.")
        else:
            if st.button("Generate AI Insights"):
                try:
                    data = pd.DataFrame({"review_text": ["battery life is great", "camera is poor", "performance is excellent", "price is too high"]})
                    active_model = models.get(selected_model_name, models.get("selected_model", models["logistic_regression"]))
                    predictor = ReviewPredictor(active_model, artifacts["vectorizer"])
                    analyzed_df, _ = summarize_reviews(data, predictor)
                    summary = build_genai_summary(analyzed_df)
                    client = GenAIInsightsClient()
                    response = client.generate_customer_insights(summary)
                    st.markdown("### AI-Generated Insights")
                    st.write(response["content"])
                except Exception as exc:
                    st.error(f"AI insights generation failed: {exc}")

    with tab5:
        st.header("Model Performance")
        if metrics_data:
            for model_name, metrics in metrics_data.items():
                st.subheader(model_name.replace("_", " ").title())
                st.write({
                    "Accuracy": round(metrics["accuracy"], 4),
                    "Precision": round(metrics["precision"], 4),
                    "Recall": round(metrics["recall"], 4),
                    "Macro F1": round(metrics["macro_f1"], 4),
                })
                st.text(metrics["classification_report"])
        else:
            st.warning("No metrics are available yet. Run the training pipeline first.")
