"""Sparse OpenCV video sampling with bounded OCR work."""
from pathlib import Path
import logging
import time

from guards.image_guard import _ocr_frame
from guards.text_guard import analyze_text

logger = logging.getLogger("claimshield")


def analyze_video(path, interval_seconds=2, max_frames=6):
    capture = None
    started = time.perf_counter()
    try:
        import cv2
        capture = cv2.VideoCapture(str(Path(path)))
        if not capture.isOpened():
            return {"source": "video", "provided": True, "frames_analyzed": 0, "frame_results": [], "suspicious_frames": 0, "error": "OpenCV could not open or decode this video format. Try MP4 (H.264) or AVI."}
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = min(frame_count / fps if fps else 0, 30)
        logger.info("[ClaimShield] Video opened: %.2fs (duration %.2fs)", time.perf_counter() - started, duration)
        frames = []
        timestamp = 0.0
        while timestamp <= duration and len(frames) < max_frames:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                break
            ocr_started = time.perf_counter()
            text, ocr_error, ocr_elapsed = _ocr_frame(frame)
            logger.info("[ClaimShield] Frame %d OCR: %.2fs", len(frames), ocr_elapsed)
            result = analyze_text(text) if text else {"prediction": 0, "confidence": 0.0, "is_prompt_injection": False}
            frames.append({"timestamp": timestamp, "ocr_text": text, "ocr_error": ocr_error, "prediction": result["prediction"], "confidence": result.get("confidence", result.get("risk_score", 0.0)), "is_prompt_injection": result["prediction"] == 1})
            timestamp += interval_seconds
        if not frames:
            return {"source": "video", "provided": True, "frames_analyzed": 0, "frame_results": [], "suspicious_frames": 0, "error": "Video opened but no decodable frames were found."}
        logger.info("[ClaimShield] Frame extraction/OCR complete: %.2fs", time.perf_counter() - started)
        return {"source": "video", "provided": True, "frames_analyzed": len(frames), "frame_results": frames, "suspicious_frames": sum(frame["is_prompt_injection"] for frame in frames), "error": ""}
    except Exception as error:
        return {"source": "video", "provided": True, "frames_analyzed": 0, "frame_results": [], "suspicious_frames": 0, "error": f"Video analysis failed: {error}"}
    finally:
        if capture is not None:
            capture.release()
