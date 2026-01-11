import fitz
import io
import pdfplumber
from PIL import Image
from .Tesseract_searcher import init_tesseract
import pytesseract
from RFC_logging_system.LoggerFactory import get_logger

class PDFToTextConverter:
    """Конвертер PDF в текст с поддержкой OCR."""

    def __init__(self):
        # Looking for tessearc on PC
        self.logger = get_logger("PDFConverter")
        self.OCR_AVAILABLE, self.TESSERACT_PATH = init_tesseract()

    def convert(self, pdf_bytes: bytes, use_fallback: bool = True, use_ocr: bool = True) -> str | None:
        """
        Конвертирует PDF в текст.
        """
        # Попытка 1: pdfplumber
        self.logger.info("Attempting to extract text with pdfplumber")
        text = self._extract_with_pdfplumber(pdf_bytes)
        if text:
            self.logger.info("Successfully extracted text with pdfplumber")
            return text
        else:
            self.logger.info("Failed to extract text with pdfplumber, trying fallback method")

        # Попытка 2: PyMuPDF
        if use_fallback:
            text = self._extract_with_pymupdf(pdf_bytes)
            if text:
                self.logger.info("Successfully extracted text with PyMuPDF")
                return text
            else:
                self.logger.info("Failed to extract text with PyMuPDF, trying OCR")

        # Попытка 3: OCR
        if use_ocr:
            text = self._extract_with_ocr(pdf_bytes)
            if text:
                self.logger.info("Successfully extracted text with OCR")
                return text
            else:
                self.logger.warning("Failed to extract text with OCR")

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

                result = "\n".join(pages_text) if pages_text else None
                if result:
                    self.logger.info(f"Successfully extracted text from {len(pages_text)} pages using pdfplumber")
                else:
                    self.logger.warning("No text found using pdfplumber")
                return result

        except Exception as e:
            self.logger.error(f"Error extracting text with pdfplumber: {e}")
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
            result = "\n".join(pages_text) if pages_text else None
            if result:
                self.logger.info(f"Successfully extracted text from {len(pages_text)} pages using PyMuPDF")
            else:
                self.logger.warning("No text found using PyMuPDF")
            return result

        except Exception as e:
            self.logger.error(f"Error extracting text with PyMuPDF: {e}")
            return None

    def _extract_with_ocr(self, pdf_bytes: bytes) -> str | None:
        """Извлечение текста через OCR (для сканированных документов)."""
        if not self.OCR_AVAILABLE:
            self.logger.warning("OCR not available, skipping OCR extraction")
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
            result = "\n\n".join(pages_text) if pages_text else None
            if result:
                self.logger.info(f"Successfully extracted text from {len(pages_text)} pages using OCR")
            else:
                self.logger.warning("No text found using OCR")
            return result

        except Exception as e:
            self.logger.error(f"Error extracting text with OCR: {e}")
            return None


if __name__ == "__main__":
    converter = PDFToTextConverter()