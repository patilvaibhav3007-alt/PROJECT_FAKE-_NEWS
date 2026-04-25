#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a Fake News Detection model on Indian news/social media-style text.

Workflow:
1) Load data from data/true.csv and data/fake.csv
2) Build a TF-IDF + Logistic Regression pipeline
3) Train/test split for quick evaluation
4) Persist the trained pipeline to model/fake_news_model.pkl
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle as sk_shuffle


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"


def load_dataset(true_path: Path, fake_path: Path) -> pd.DataFrame:
    """
    Load TRUE and FAKE datasets and return a combined DataFrame with labels.
    Handles inconsistent CSV quoting and extra columns.
    """
    def clean_csv(path):
        # Read the file as a single column and strip quotes
        try:
            # First try standard CSV reading
            df = pd.read_csv(path, sep=',', quoting=0, on_bad_lines='skip')
            if 'text' not in df.columns:
                # Fallback: read line by line if standard CSV fails or 'text' col is missing
                with open(path, 'r', encoding='utf-8') as f:
                    lines = [line.strip().strip('"').strip("'") for line in f if line.strip()]
                    # Skip the 'text' header if present
                    if lines and lines[0].lower() == 'text':
                        lines = lines[1:]
                    df = pd.DataFrame({'text': lines})
            else:
                # Clean the text column if it exists
                df['text'] = df['text'].astype(str).str.strip().str.strip('"').str.strip("'")
            return df
        except Exception:
            # Final fallback
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip().strip('"').strip("'") for line in f if line.strip()]
                if lines and lines[0].lower() == 'text':
                    lines = lines[1:]
                return pd.DataFrame({'text': lines})

    true_df = clean_csv(true_path).assign(label="TRUE")
    fake_df = clean_csv(fake_path).assign(label="FAKE")

    df = pd.concat([true_df, fake_df], axis=0, ignore_index=True)
    # Final cleanup
    df["text"] = df["text"].astype(str).str.strip()
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() > 10]
    df = sk_shuffle(df, random_state=42).reset_index(drop=True)
    return df


def build_pipeline() -> Pipeline:
    """
    Build a text classification pipeline with TF-IDF + Logistic Regression.
    Config tuned for Hinglish/Indian news style without language-specific stopwords.
    """
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),           # capture unigrams + bigrams to pick up phrases like "free petrol"
        max_df=0.95,                  # ignore extremely common tokens
        min_df=1,                     # keep rare tokens; adjust higher for larger datasets
        max_features=50000,           # cap vocabulary size
        strip_accents="unicode",
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        solver="liblinear",           # good for small/medium data; supports predict_proba
        C=1.0,
        class_weight="balanced",      # handle potential imbalance between TRUE/FAKE
        random_state=42,
        max_iter=200,
    )
    return Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("clf", clf),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> Tuple[Pipeline, float, float]:
    """
    Train the pipeline and return (pipeline, accuracy, f1_macro).
    """
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("=== Evaluation on Hold-out Set ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print("\nDetailed report:")
    print(classification_report(y_test, y_pred, digits=4))
    return pipeline, acc, f1


def save_model(pipeline: Pipeline, path: Path) -> None:
    """
    Persist the trained pipeline to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved to: {path}")


def main() -> None:
    """
    Entry point for training from local CSVs.
    """
    print("Loading dataset...")
    df = load_dataset(DATA_DIR / "true.csv", DATA_DIR / "fake.csv")
    print(f"Loaded {len(df)} samples (TRUE={sum(df.label=='TRUE')}, FAKE={sum(df.label=='FAKE')})")

    print("Training pipeline (TF-IDF + Logistic Regression)...")
    pipeline, acc, f1 = train_and_evaluate(df)

    print("Saving model...")
    save_model(pipeline, MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()

