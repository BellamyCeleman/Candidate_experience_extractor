import os

# ================================
# LOOKING FOR TESSERACT ON WINDOWS / LINUX
# ================================

def find_tesseract():
    """
    Automatically finds tesseract.exe on Windows or Linux.

    Returns:
        str: Path to tesseract or None if not found
    """
    if os.name == 'nt':  # Windows
        # Possible paths on Windows
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Tesseract-OCR\tesseract.exe',
            os.path.expandvars(r'%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe'),
            os.path.expandvars(r'%PROGRAMFILES%\Tesseract-OCR\tesseract.exe'),
        ]

        # Also check PATH
        try:
            import shutil
            path_tesseract = shutil.which('tesseract')
            if path_tesseract:
                possible_paths.insert(0, path_tesseract)
        except:
            pass

        for path in possible_paths:
            if os.path.exists(path):
                return path
    else:  # Linux/Mac
        try:
            import shutil
            return shutil.which('tesseract')
        except:
            pass

    return None

# Check OCR availability
OCR_AVAILABLE = False
TESSERACT_PATH = None

try:
    import pytesseract

    # Automatically find tesseract
    TESSERACT_PATH = find_tesseract()

    if TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

        # Verify that it works
        try:
            version = pytesseract.get_tesseract_version()
            OCR_AVAILABLE = True
            print(f"✅ Tesseract found: {TESSERACT_PATH}")
            print(f"✅ Version: {version}")
        except Exception as e:
            print(f"⚠️  Tesseract found at {TESSERACT_PATH}, but not working: {e}")
            OCR_AVAILABLE = False
    else:
        print("⚠️  Tesseract not found in standard paths")
        print("📝 Install Tesseract:")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Linux: sudo apt-get install tesseract-ocr")
        print("   Mac: brew install tesseract")
        OCR_AVAILABLE = False

except ImportError:
    print("⚠️  WARNING: pytesseract not installed. OCR will be unavailable.")
    print("   Install with: pip install pytesseract Pillow")
    OCR_AVAILABLE = False