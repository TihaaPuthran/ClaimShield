import os
import shutil


def get_tesseract_path():
    configured = os.getenv("TESSERACT_CMD")
    if configured and os.path.isfile(configured):
        return configured
    system_path = shutil.which("tesseract")
    if system_path:
        return system_path
    if os.name == "nt":
        local_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(local_path):
            return local_path
    return None


def is_ocr_available():
    return get_tesseract_path() is not None
