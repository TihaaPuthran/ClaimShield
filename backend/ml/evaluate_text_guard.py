"""Evaluate saved Text Guard artifacts on the official test split.

Usage: python -m ml.evaluate_text_guard
"""
import json

import joblib
from datasets import load_dataset

from ml.train_text_guard import (ARTIFACT_DIR, DATASET_CONFIG, DATASET_NAME,
                                 _metrics, _split_data)


def evaluate():
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    texts, labels = _split_data(dataset, "test")
    model = joblib.load(ARTIFACT_DIR / "model.pkl")
    vectorizer = joblib.load(ARTIFACT_DIR / "vectorizer.pkl")
    metrics = _metrics(model, vectorizer.transform(texts), labels)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate()
