"""Train improved TF-IDF Text Guard v2 without overwriting the baseline."""
import csv
import json
from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import LinearSVC

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "models" / "text_guard_model_v2"
DATA = BASE / "data" / "claimshield_text"


def metrics(model, features, labels):
    prediction = model.predict(features)
    matrix = confusion_matrix(labels, prediction, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(features)[:, 1]
        auc = roc_auc_score(labels, score)
    else:
        score = model.decision_function(features)
        auc = roc_auc_score(labels, score)
    return {"accuracy": accuracy_score(labels, prediction), "precision": precision_score(labels, prediction, zero_division=0), "recall": recall_score(labels, prediction, zero_division=0), "f1_score": f1_score(labels, prediction, zero_division=0), "roc_auc": auc, "confusion_matrix": matrix.tolist(), "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0}


def insurance_training_rows():
    rows = []
    for split in ("train.csv", "validation.csv", "test.csv"):
        with open(DATA / split, encoding="utf-8", newline="") as file:
            rows.extend(row for row in csv.DictReader(file) if row["label"] == "0")
    return rows


def insurance_eval_rows():
    with open(BASE / "data" / "insurance_eval.csv", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def vectorizer(name):
    if name == "A":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True, max_features=50000, strip_accents="unicode")
    if name == "B":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, max_df=0.98, sublinear_tf=True, max_features=75000, strip_accents="unicode")
    return FeatureUnion([("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000, strip_accents="unicode")), ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True))])


def train():
    dataset = load_dataset("neuralchemy/Prompt-injection-dataset", "core")
    official_train = [(str(row["text"]).strip(), int(row["label"])) for row in dataset["train"] if row["text"] and int(row["label"]) in (0, 1)]
    official_validation = [(str(row["text"]).strip(), int(row["label"])) for row in dataset["validation"] if row["text"] and int(row["label"]) in (0, 1)]
    official_test = [(str(row["text"]).strip(), int(row["label"])) for row in dataset["test"] if row["text"] and int(row["label"]) in (0, 1)]
    insurance_train = [(row["text"].strip(), 0) for row in insurance_training_rows() if row["text"].strip()]
    train_rows = list(dict.fromkeys(official_train + insurance_train))
    x_train, y_train = zip(*train_rows)
    x_validation, y_validation = zip(*official_validation)
    x_test, y_test = zip(*official_test)
    x_insurance, y_insurance = zip(*[(row["text"], int(row["label"])) for row in insurance_eval_rows()])
    comparison = []
    for config_name in ("A", "B", "C"):
        fitted_vectorizer = vectorizer(config_name)
        train_features = fitted_vectorizer.fit_transform(x_train)
        validation_features = fitted_vectorizer.transform(x_validation)
        for classifier_name, factory in [("logistic_regression", LogisticRegression), ("linear_svm", LinearSVC), ("calibrated_linear_svm", lambda **kwargs: CalibratedClassifierCV(LinearSVC(**kwargs), method="sigmoid", cv=5, n_jobs=-1))]:
            values = [0.1, 1.0, 2.0, 5.0] if classifier_name == "logistic_regression" else [0.1, 0.5, 1.0, 2.0]
            for c_value in values:
                for weight_name, weight in (("default", None), ("balanced", "balanced")):
                    kwargs = {"C": c_value, "max_iter": 1000, "class_weight": weight}
                    model = factory(**kwargs)
                    model.fit(train_features, y_train)
                    result = metrics(model, validation_features, y_validation)
                    comparison.append({"configuration": config_name, "classifier": classifier_name, "C": c_value, "class_weight": weight_name, "validation": result, "model": model, "vectorizer": fitted_vectorizer})
                    print(json.dumps({"configuration": config_name, "classifier": classifier_name, "C": c_value, "class_weight": weight_name, "validation": result}))
    best = max(comparison, key=lambda item: (item["validation"]["f1_score"], item["validation"]["recall"], -item["validation"]["false_positive_rate"]))
    test_features = best["vectorizer"].transform(x_test)
    insurance_features = best["vectorizer"].transform(x_insurance)
    official_test = metrics(best["model"], test_features, y_test)
    insurance_test = metrics(best["model"], insurance_features, y_insurance)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], OUT / "model.pkl")
    joblib.dump(best["vectorizer"], OUT / "vectorizer.pkl")
    serializable = [{key: value for key, value in item.items() if key not in {"model", "vectorizer"}} for item in comparison]
    (OUT / "comparison.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps({"best_model": {key: best[key] for key in ("configuration", "classifier", "C", "class_weight")}, "official_test": official_test, "insurance_domain_test": insurance_test}, indent=2), encoding="utf-8")
    print(json.dumps({"best_model": {key: best[key] for key in ("configuration", "classifier", "C", "class_weight")}, "official_test": official_test, "insurance_domain_test": insurance_test}, indent=2))


if __name__ == "__main__":
    train()
