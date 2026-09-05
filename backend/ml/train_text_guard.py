"""Train classical ML Text Guard candidates on the official Hugging Face splits.

Usage: python -m ml.train_text_guard
"""
import json
from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATASET_NAME = "neuralchemy/Prompt-injection-dataset"
DATASET_CONFIG = "core"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models" / "text_guard_model"


def _split_data(dataset, split):
    if split not in dataset:
        raise ValueError(f"Dataset must provide an official '{split}' split")
    records = dataset[split]
    if not {"text", "label"}.issubset(records.column_names):
        raise ValueError("The core configuration must contain 'text' and 'label' fields")
    rows = {(str(text).strip(), int(label)) for text, label in zip(records["text"], records["label"]) if text is not None and str(text).strip() and label in (0, 1)}
    if not rows or {label for _, label in rows} != {0, 1}:
        raise ValueError(f"Official '{split}' split must contain both labels 0 and 1")
    texts, labels = zip(*rows)
    return list(texts), list(labels)


def _metrics(model, features, labels):
    predictions = model.predict(features)
    result = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1_score": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
    if hasattr(model, "decision_function"):
        result["roc_auc"] = roc_auc_score(labels, model.decision_function(features))
    elif hasattr(model, "predict_proba"):
        result["roc_auc"] = roc_auc_score(labels, model.predict_proba(features)[:, 1])
    return result


def train():
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    train_texts, train_labels = _split_data(dataset, "train")
    validation_texts, validation_labels = _split_data(dataset, "validation")
    test_texts, test_labels = _split_data(dataset, "test")
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1)
    train_features = vectorizer.fit_transform(train_texts)
    validation_features = vectorizer.transform(validation_texts)
    test_features = vectorizer.transform(test_texts)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    }
    validation_metrics = {}
    trained = {}
    for name, model in candidates.items():
        model.fit(train_features, train_labels)
        trained[name] = model
        validation_metrics[name] = _metrics(model, validation_features, validation_labels)
    best_name = max(validation_metrics, key=lambda name: validation_metrics[name]["f1_score"])
    best_model = trained[best_name]
    test_metrics = _metrics(best_model, test_features, test_labels)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACT_DIR / "model.pkl")
    joblib.dump(vectorizer, ARTIFACT_DIR / "vectorizer.pkl")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps({"dataset": DATASET_NAME, "configuration": DATASET_CONFIG, "best_model": best_name, "validation": validation_metrics, "test": test_metrics}, indent=2), encoding="utf-8")
    return validation_metrics, test_metrics


if __name__ == "__main__":
    train()
