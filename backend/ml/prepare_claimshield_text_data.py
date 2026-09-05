"""Build ClaimShield CSV splits from ShieldLM and GNOTHEIA.

This script uses only unambiguous ShieldLM binary labels and extracts claim,
context-document, and summary text from GNOTHEIA as benign insurance text.
"""
import csv
import re
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "claimshield_text"
SHIELD = "Abdennebi/shieldlm-prompt-injection"
INSURANCE = "gratex/GNOTHEIA-synthetic-insurance-dataset"


def _clean(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _fingerprint(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _insurance_text(data):
    claim = data.get("claim", {})
    summary = data.get("summary", {})
    parts = [claim.get("damagedDescription"), claim.get("lossEvent"), claim.get("lossSummary"), summary.get("summary"), summary.get("missingInformation")]
    for document in data.get("contextDocuments", []) or []:
        parts.extend([document.get("documentType"), document.get("extractedDocumentContent")])
    return _clean(" ".join(str(part) for part in parts if part))


def _unique(rows):
    seen = set()
    output = []
    for row in rows:
        key = _fingerprint(row["text"])
        if key and key not in seen:
            seen.add(key)
            output.append(row)
    return output


def build():
    shield = load_dataset(SHIELD)
    insurance = load_dataset(INSURANCE)
    malicious = []
    for split in ("train", "validation", "test"):
        for row in shield[split]:
            if int(row["label_binary"]) == 1 and _clean(row["text"]):
                malicious.append({"text": _clean(row["text"]), "label": 1, "source": SHIELD, "category": row.get("label_category") or "prompt_injection"})
    benign = []
    for row in insurance["train"]:
        text = _insurance_text(row["data"])
        if text:
            benign.append({"text": text, "label": 0, "source": INSURANCE, "category": "insurance_claim_document"})
    malicious, benign = _unique(malicious), _unique(benign)
    eval_malicious, eval_benign = malicious[-100:], benign[-100:]
    eval_rows = eval_benign + eval_malicious
    eval_keys = {_fingerprint(row["text"]) for row in eval_rows}
    malicious = [row for row in malicious[:-100] if _fingerprint(row["text"]) not in eval_keys]
    benign = [row for row in benign[:-100] if _fingerprint(row["text"]) not in eval_keys]
    target = min(len(malicious), len(benign))
    rows = malicious[:target] + benign[:target]
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    train_rows, holdout = train_test_split(rows, test_size=0.30, random_state=42, stratify=labels)
    holdout_labels = [row["label"] for row in holdout]
    validation_rows, test_rows = train_test_split(holdout, test_size=0.50, random_state=42, stratify=holdout_labels)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train.csv", train_rows), ("validation.csv", validation_rows), ("test.csv", test_rows)):
        with open(OUT_DIR / name, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["text", "label", "source", "category"])
            writer.writeheader()
            writer.writerows(split)
    with open(OUT_DIR.parent / "insurance_eval.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label", "source", "category"])
        writer.writeheader()
        writer.writerows(eval_rows)
    print({"shield_malicious_rows": len(malicious), "insurance_benign_rows": len(benign), "class_distribution": {"0": target, "1": target}, "train": len(train_rows), "validation": len(validation_rows), "test": len(test_rows), "insurance_eval": len(eval_rows)})


if __name__ == "__main__":
    build()
