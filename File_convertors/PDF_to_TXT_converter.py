import fitz
import io
import pdfplumber
from PIL import Image
from .Tesseract_searcher import init_tesseract
import pytesseract

class PDFToTextConverter:
    """Конвертер PDF в текст с поддержкой OCR."""

    def __init__(self):
        # Looking for tessearc on PC
        self.OCR_AVAILABLE, self.TESSERACT_PATH = init_tesseract()

    def convert(self, pdf_bytes: bytes, use_fallback: bool = True, use_ocr: bool = True) -> str | None:
        """
        Конвертирует PDF в текст.
        """
        # Попытка 1: pdfplumber
        text = self._extract_with_pdfplumber(pdf_bytes)
        if text:
            return text

        # Попытка 2: PyMuPDF
        if use_fallback:
            text = self._extract_with_pymupdf(pdf_bytes)
            if text:
                return text

        # Попытка 3: OCR
        if use_ocr:
            text = self._extract_with_ocr(pdf_bytes)
            if text:
                return text

        return None

    def _extract_with_pdfplumber(self, pdf_bytes: bytes) -> str | None:
        """Извлечение текста через pdfplumber."""
        try:
            pdf_file = io.BytesIO(pdf_bytes)

            with pdfplumber.open(pdf_file) as pdf:
                pages_text = []

                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3, layout=False)
                    if page_text and page_text.strip():
                        pages_text.append(page_text)

                return "\n".join(pages_text) if pages_text else None

        except Exception:
            return None

    def _extract_with_pymupdf(self, pdf_bytes: bytes) -> str | None:
        """Извлечение текста через PyMuPDF (fitz)."""
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_file, filetype="pdf")

            pages_text = []

            for page in doc:
                page_text = page.get_text(option="text", sort=True)
                if page_text and page_text.strip():
                    pages_text.append(page_text)

            doc.close()
            return "\n".join(pages_text) if pages_text else None

        except Exception:
            return None

    def _extract_with_ocr(self, pdf_bytes: bytes) -> str | None:
        """Извлечение текста через OCR (для сканированных документов)."""
        if not self.OCR_AVAILABLE:
            return None

        try:
            pdf_file = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_file, filetype="pdf")

            pages_text = []

            for page in doc:
                # Конвертируем страницу в изображение 300 DPI
                pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                image = Image.open(io.BytesIO(pix.tobytes("png")))

                # OCR
                page_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
                if page_text and page_text.strip():
                    pages_text.append(page_text.strip())

            doc.close()
            return "\n\n".join(pages_text) if pages_text else None

        except Exception:
            return None


if __name__ == "__main__":
    converter = PDFToTextConverter()