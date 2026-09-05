"""Inference interface for the trained ML-based Text Guard."""
import logging
from functools import lru_cache
from pathlib import Path

import joblib

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "text_guard_model"
logger = logging.getLogger("claimshield.text_guard")


@lru_cache(maxsize=1)
def _load_artifacts():
    model_path = MODEL_DIR / "model.pkl"
    vectorizer_path = MODEL_DIR / "vectorizer.pkl"
    if not model_path.exists() or not vectorizer_path.exists():
        raise RuntimeError("Text Guard model artifacts are missing")
    return joblib.load(model_path), joblib.load(vectorizer_path)


def analyze_text(text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must not be empty")
    model, vectorizer = _load_artifacts()
    features = vectorizer.transform([text])
    prediction = int(model.predict(features)[0])
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(features)[0][prediction])
    else:
        confidence = 1.0
    malicious = prediction == 1
    classification = "PROMPT_INJECTION" if malicious else "BENIGN"
    logger.info("[TextGuard] prediction=%s classification=%s confidence=%.4f", prediction, classification, confidence)
    return {"source": "text", "provided": True, "prediction": prediction, "classification": classification, "confidence": confidence, "evidence_text": text, "detector": "TF-IDF + Linear SVM", "is_prompt_injection": malicious, "explanation": "The trained Text Guard detected prompt-injection-like language in the claim narrative." if malicious else "The trained Text Guard classified the claim narrative as benign."}
