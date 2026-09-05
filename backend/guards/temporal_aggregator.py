"""Reconstruct prompt-injection text spread across chronological video frames."""
from guards.text_guard import analyze_text


def analyze_temporal(frames):
    combined_text = " ".join(frame.get("ocr_text", "").strip() for frame in sorted(frames, key=lambda item: item.get("timestamp", 0)) if frame.get("ocr_text", "").strip())
    if not combined_text:
        return {"combined_text": "", "prediction": 0, "classification": "BENIGN", "confidence": 0.0, "is_temporal_attack": False}
    result = analyze_text(combined_text)
    return {"combined_text": combined_text, "prediction": result["prediction"], "classification": "TEMPORAL_PROMPT_INJECTION" if result["prediction"] else "BENIGN", "confidence": result.get("confidence", result.get("risk_score", 0.0)), "is_temporal_attack": result["prediction"] == 1}
