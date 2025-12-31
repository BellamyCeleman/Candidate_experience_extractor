import os
import shutil
from typing import Tuple, Optional


def init_tesseract(verbose: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Find and initialize Tesseract OCR.

    Returns:
        (is_available, path_to_tesseract)
    """
    # 1. Проверяем pytesseract
    try:
        import pytesseract
    except ImportError:
        if verbose:
            print("⚠️  pytesseract not installed. Run: pip install pytesseract Pillow")
        return False, None

    # 2. Ищем tesseract
    path = shutil.which('tesseract')

    if not path and os.name == 'nt':  # Windows fallback
        for p in [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]:
            if os.path.exists(p):
                path = p
                break

    if not path:
        if verbose:
            print("⚠️  Tesseract not found. Install: apt install tesseract-ocr / brew install tesseract")
        return False, None

    # 3. Проверяем работоспособность
    pytesseract.pytesseract.tesseract_cmd = path

    try:
        pytesseract.get_tesseract_version()
        if verbose:
            print(f"✅ Tesseract ready: {path}")
        return True, path
    except Exception as e:
        if verbose:
            print(f"⚠️  Tesseract broken: {e}")
        return False, path

if __name__ == "__main__":
    OCR_AVAILABLE, TESSERACT_PATH = init_tesseract()