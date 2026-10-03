from __future__ import annotations

import re
import string
from typing import Iterable

import pandas as pd


def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_review_dataframe(raw_records: Iterable[dict], text_column: str = "content") -> pd.DataFrame:
    df = pd.DataFrame(list(raw_records))
    if df.empty:
        return df

    if text_column not in df.columns:
        raise ValueError(f"Missing required text column: {text_column}")

    df = df.rename(columns={text_column: "review_text"})
    df["review_text"] = df["review_text"].fillna("")
    df = df.drop_duplicates(subset=["review_text"]).copy()
    df["review_text"] = df["review_text"].apply(clean_text)
    df["review_text"] = df["review_text"].apply(normalize_whitespace)
    df = df[df["review_text"].str.len() > 0].copy()
    return df


def build_training_frame(dataset, label_column: str = "label", text_column: str = "text") -> pd.DataFrame:
    records = []
    for row in dataset:
        item = {"label": row.get(label_column, row.get("label", 0)), "content": row.get(text_column, "")}
        records.append(item)
    df = prepare_review_dataframe(records, text_column="content")
    df["label"] = df["label"].astype(int)
    return df[["review_text", "label"]]
