"""OCR-only image guard; it reuses the trained text guard and no image model."""
import os
import time
from pathlib import Path

from guards.text_guard import analyze_text
from services.ocr_service import get_tesseract_path


def _configure_tesseract(pytesseract):
    configured = os.getenv("TESSERACT_CMD")
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
    elif (system_tesseract := get_tesseract_path()):
        pytesseract.pytesseract.tesseract_cmd = system_tesseract


def _ocr(path):
    try:
        import pytesseract
        from PIL import Image
        _configure_tesseract(pytesseract)
        return pytesseract.image_to_string(Image.open(path)).strip(), ""
    except Exception as error:
        return "", f"OCR unavailable: {error}"


def _ocr_frame(frame):
    try:
        import cv2
        import pytesseract
        _configure_tesseract(pytesseract)
        started = time.perf_counter()
        height, width = frame.shape[:2]
        if width > 1280:
            frame = cv2.resize(frame, (1280, int(height * 1280 / width)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        _, processed = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(processed, config="--oem 3 --psm 6", timeout=8).strip()
        return text, "", time.perf_counter() - started
    except Exception as error:
        return "", f"OCR unavailable: {error}", time.perf_counter() - started if 'started' in locals() else 0.0


def analyze_image(path):
    ocr_text, ocr_error = _ocr(Path(path))
    if not ocr_text and not ocr_error:
        return {"source": "image", "provided": True, "ocr_status": "SUCCESS", "ocr_text": "", "ocr_error": "", "prediction": None, "classification": "NO_TEXT_DETECTED", "confidence": None, "is_prompt_injection": False, "detector": "Tesseract OCR + Text Guard"}
    if ocr_error:
        return {"source": "image", "provided": True, "ocr_status": "UNAVAILABLE", "ocr_text": "", "ocr_error": ocr_error, "prediction": None, "classification": "UNDETERMINED", "confidence": None, "is_prompt_injection": None, "detector": "Tesseract OCR + Text Guard"}
    result = analyze_text(ocr_text)
    return {"source": "image", "provided": True, "ocr_status": "SUCCESS", "ocr_text": ocr_text, "ocr_error": "", "prediction": result["prediction"], "classification": result["classification"], "decision_score": result.get("decision_score"), "confidence": None, "is_prompt_injection": result["prediction"] == 1, "detector": "Tesseract OCR + Text Guard"}
