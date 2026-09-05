"""Train/evaluate supervised FastText and compare with the preserved SVM baseline."""
import csv
import json
import shutil
import tempfile
from pathlib import Path

import fasttext
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "claimshield_text"
ARTIFACT_DIR = BASE / "models" / "text_guard_fasttext"


def read_split(name):
    with open(DATA_DIR / name, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_baseline(rows):
    model = joblib.load(BASE / "models" / "text_guard_model" / "model.pkl")
    vectorizer = joblib.load(BASE / "models" / "text_guard_model" / "vectorizer.pkl")
    predictions = model.predict(vectorizer.transform([row["text"] for row in rows]))
    return metrics(predictions, [int(row["label"]) for row in rows])


def metrics(predictions, labels):
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"accuracy": accuracy_score(labels, predictions), "precision": precision_score(labels, predictions, zero_division=0), "recall": recall_score(labels, predictions, zero_division=0), "f1_score": f1_score(labels, predictions, zero_division=0), "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0, "confusion_matrix": matrix.tolist()}


def train():
    train_rows, validation_rows, test_rows = (read_split(name) for name in ("train.csv", "validation.csv", "test.csv"))
    with open(BASE / "data" / "insurance_eval.csv", encoding="utf-8", newline="") as file:
        insurance_rows = list(csv.DictReader(file))
    configurations = [{"wordNgrams": n, "epoch": epoch, "lr": lr, "dim": dim, "minn": 2, "maxn": 5} for n in (1, 2) for epoch, lr, dim in ((10, 0.1, 100), (20, 0.3, 200), (25, 0.5, 200))]
    with tempfile.TemporaryDirectory() as temporary:
        train_file = Path(temporary) / "train.txt"
        train_file.write_text("\n".join(f"__label__{row['label']} {row['text']}" for row in train_rows), encoding="utf-8")
        results = []
        for config in configurations:
            model = fasttext.train_supervised(input=str(train_file), loss="ova", **config)
            predictions = [int(model.predict(row["text"])[0][0].rsplit("__label__", 1)[1]) for row in validation_rows]
            results.append({"configuration": config, "validation": metrics(predictions, [int(row["label"]) for row in validation_rows]), "model": model})
        selected = max(results, key=lambda result: result["validation"]["f1_score"])
        model = selected["model"]
        test_predictions = [int(model.predict(row["text"])[0][0].rsplit("__label__", 1)[1]) for row in test_rows]
        insurance_predictions = [int(model.predict(row["text"])[0][0].rsplit("__label__", 1)[1]) for row in insurance_rows]
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(ARTIFACT_DIR / "model.bin"))
        summary = {"model_comparison": [{"configuration": result["configuration"], "validation": result["validation"]} for result in results], "selected_configuration": selected["configuration"], "official_test": metrics(test_predictions, [int(row["label"]) for row in test_rows]), "insurance_domain_test": metrics(insurance_predictions, [int(row["label"]) for row in insurance_rows]), "linear_svm_baseline": {"official_test": read_baseline(test_rows), "insurance_domain_test": read_baseline(insurance_rows)}}
        (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    train()
