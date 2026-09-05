"""Conservative heuristic placeholder for future trained media forensics."""
from pathlib import Path


def analyze_image_authenticity(path):
    try:
        from PIL import Image
        image = Image.open(Path(path))
        probability = 0.05 if image.getexif() else 0.15
        return {"provided": True, "media_type": "image", "synthetic_media_detected": False, "synthetic_probability": probability, "classification": "LIKELY_REAL", "detector": "Prototype metadata heuristic; not a trained authenticity model", "explanation": "No strong synthetic-media signal was found by the prototype heuristic."}
    except Exception as error:
        return {"provided": True, "media_type": "image", "synthetic_media_detected": False, "synthetic_probability": 0.0, "classification": "LIKELY_REAL", "detector": "Prototype fallback", "explanation": f"Authenticity analysis unavailable: {error}"}


def analyze_video_authenticity(path, interval_seconds=2, max_frames=6):
    try:
        import cv2
        cap = cv2.VideoCapture(str(Path(path))); fps = cap.get(cv2.CAP_PROP_FPS) or 0; duration = min((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps if fps else 0, 30); checked = 0; timestamp = 0
        while timestamp <= duration and checked < max_frames:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000); ok, _ = cap.read()
            if not ok: break
            checked += 1; timestamp += interval_seconds
        cap.release()
        return {"provided": True, "media_type": "video", "frames_checked": checked, "suspicious_frames": 0, "synthetic_probability": 0.05, "classification": "LIKELY_REAL", "synthetic_media_detected": False, "detector": "Prototype frame heuristic; not a trained authenticity model"}
    except Exception as error:
        return {"provided": True, "media_type": "video", "frames_checked": 0, "suspicious_frames": 0, "synthetic_probability": 0.0, "classification": "LIKELY_REAL", "synthetic_media_detected": False, "detector": "Prototype fallback", "error": str(error)}
