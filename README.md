# AI Product Review Intelligence

## Project Overview
AI Product Review Intelligence is a portfolio-ready machine learning and natural language processing project that analyzes product reviews, predicts sentiment, and converts aggregated review patterns into business-friendly AI-generated insights.

## Problem Statement
Online product reviews are often noisy, large in volume, and difficult to summarize quickly. Businesses need an efficient way to understand customer sentiment, identify recurring strengths and weaknesses, and turn review data into action.

## Objectives
- Train a real sentiment classification model on a real product-review dataset.
- Use reproducible data preparation and evaluation.
- Compare two TF-IDF-based classifiers.
- Save the trained models and vectorizer.
- Build a local Streamlit dashboard for analysis.
- Add a GenAI layer that interprets aggregated results, not individual review predictions.

## Key Features
- Real Amazon Polarity dataset subset
- Preprocessing pipeline with missing-value handling, duplicate removal, and text cleaning
- TF-IDF feature extraction
- Logistic Regression and Linear SVM comparison
- Held-out evaluation with real metrics
- Joblib model persistence
- Single-review and CSV batch prediction
- AI-generated customer insight summaries
- Local Streamlit app

## System Architecture
The project is intentionally simple and interview-friendly:

1. Data loading from Hugging Face Datasets
2. Preprocessing and train/test splitting
3. TF-IDF transformation
4. Model training and evaluation
5. Model artifact saving
6. Streamlit dashboard for prediction and business summaries
7. Optional LLM integration for aggregated customer insight generation

## ML Methodology
The workflow follows a standard ML pipeline:

- Load a reproducible subset of Amazon Polarity
- Clean and normalize text
- Split into train/test with stratification
- Fit TF-IDF on training data only
- Train Logistic Regression and Linear SVM
- Evaluate on the held-out test set
- Select the best model using Macro F1-score

## Dataset
- Dataset name: `amazon_polarity`
- Source: Hugging Face Datasets
- Records used: a reproducible subset, currently set to 20,000 rows for a laptop-friendly workflow
- Labels: 0 = negative, 1 = positive
- Train/test split: 80/20 with stratification and random_state=42

Note: This project intentionally does not claim to train on the full multi-million-record Amazon dataset.

## Preprocessing
The preprocessing pipeline includes:
- missing-value handling
- duplicate removal
- lowercase conversion
- punctuation removal
- whitespace normalization
- filtering of empty cleaned text

This is done only after the dataset is loaded so that the model does not leak information from the test set.

## TF-IDF Explanation
TF-IDF converts review text into numerical features by weighting terms according to their frequency in a document and rarity across the corpus. It is a strong baseline for text classification when combined with linear models.

## Models
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM

The selected model is based primarily on Macro F1-score, which is the most appropriate summary metric for balanced binary classification.

## Evaluation Metrics
The project evaluates on the held-out test set using:
- Accuracy
- Precision
- Recall
- Macro F1-score
- Classification report
- Confusion matrix

## GenAI Integration
The GenAI module does not perform sentiment classification. Instead, it works on summarized review data and produces:
- executive customer summary
- positive themes
- negative themes
- product aspects
- recommendations
- brief business conclusion

This is clearly labeled as AI-generated output and uses a configurable API provider through environment variables.

## Streamlit Application
The app includes five tabs:
- Dashboard
- Review Analyzer
- Batch Analysis
- Customer Insights
- Model Performance

The dashboard runs locally and is designed for portfolio demos.

## Project Structure

```text
ai-product-review-intelligence/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── data/
│   └── README.md
├── models/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── genai.py
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── reports/
│   └── README.md
├── screenshots/
│   └── README.md
└── .env.example
```

## Installation
1. Create a virtual environment.
2. Install dependencies using requirements.txt.
3. Set environment variables using .env.example.

## Training Instructions
Run:

```bash
python src/train.py
```

This script loads the Amazon Polarity subset, trains both models, evaluates them on a held-out test set, and saves the trained artifacts in the `models` directory.

## Running the Streamlit Application
Run:

```bash
streamlit run app.py
```

## Environment Variable Setup
Create a local environment file based on `.env.example`:

```bash
copy .env.example .env
```

Then set:
- `LLM_API_KEY` or `OPENAI_API_KEY`
- `LLM_BASE_URL` if needed
- `LLM_MODEL` if needed
- `LLM_TIMEOUT` if needed

The app runs without an LLM key, and the GenAI panel clearly explains this requirement.

## Screenshots
Add screenshots here after running the app locally.

## Results
Model results will appear after the training pipeline runs. Do not add placeholder performance values.

## Limitations
- The training subset is intentionally smaller than the full Amazon Polarity dataset.
- GenAI insights depend on an external API key and network access.
- The project focuses on a practical, laptop-friendly workflow rather than large-scale production deployment.

## Future Improvements
- Add rating-aware sentiment modeling
- Add explainability with SHAP or feature importances
- Add asynchronous job queue for large upload processing
- Add a stronger retrieval or document layer in a future project

## License
This project is provided for educational and portfolio use.
