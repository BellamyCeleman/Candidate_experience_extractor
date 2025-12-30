import fitz
import io
import pdfplumber
import logging
from PIL import Image
from MainMenuImports_and_variables import module_logger

# ================================
# FILEMANAGER CLASS
# ================================

class PDF_to_TXT_converter:
    def _extract_text_with_ocr(self, pdf_bytes: bytes) -> str:
        """
        Extract text from a PDF using OCR (for scanned documents).

        Args:
            pdf_bytes: The contents of the PDF file as bytes

        Returns:
            str: Extracted text or None if extraction failed
        """
        if not OCR_AVAILABLE:
            module_logger.send_message(
                level_log=logging.WARNING,
                logger_name="FileManager",
                message=f"OCR is not available. Tesseract: {TESSERACT_PATH or 'not found'}",
                send_to_azure=True
            )
            return None

        try:
            full_text = []
            pdf_file = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_file, filetype="pdf")

            module_logger.send_message(
                level_log=logging.INFO,
                logger_name="FileManager",
                message=f"Starting OCR processing for {len(doc)} pages (using {TESSERACT_PATH})",
                send_to_azure=True
            )

            for page_num, page in enumerate(doc, 1):
                try:
                    # Convert page to a high-resolution image
                    mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                    pix = page.get_pixmap(matrix=mat)

                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))

                    # Apply OCR
                    page_text = pytesseract.image_to_string(
                        image,
                        lang='eng',
                        config='--psm 6'
                    )

                    if page_text and page_text.strip():
                        full_text.append(page_text.strip())
                        module_logger.send_message(
                            level_log=logging.DEBUG,
                            logger_name="FileManager",
                            message=f"OCR extracted {len(page_text)} chars from page {page_num}",
                            send_to_azure=True
                        )
                    else:
                        module_logger.send_message(
                            level_log=logging.DEBUG,
                            logger_name="FileManager",
                            message=f"Page {page_num}: No text extracted by OCR",
                            send_to_azure=True
                        )

                except Exception as page_error:
                    module_logger.send_message(
                        level_log=logging.WARNING,
                        logger_name="FileManager",
                        message=f"OCR error on page {page_num}: {page_error}",
                        send_to_azure=True
                    )
                    continue

            doc.close()

            if full_text:
                text = "\n\n".join(full_text)
                module_logger.send_message(
                    level_log=logging.INFO,
                    logger_name="FileManager",
                    message=f"Successfully extracted {len(full_text)} pages from PDF using OCR",
                    send_to_azure=True
                )
                return text
            else:
                module_logger.send_message(
                    level_log=logging.WARNING,
                    logger_name="FileManager",
                    message="No text extracted from PDF using OCR",
                    send_to_azure=True
                )
                return None

        except Exception as e:
            module_logger.send_message(
                level_log=logging.ERROR,
                logger_name="FileManager",
                message=f"OCR processing failed: {e}",
                send_to_azure=True
            )
            return None

    def convert_pdf_to_text(self, pdf_bytes: bytes, use_fallback: bool = True, use_ocr: bool = True):
        """
        Convert PDF bytes into text using pdfplumber, fallback to PyMuPDF,
        and OCR for scanned documents.

        Args:
            pdf_bytes: PDF file contents as bytes
            use_fallback: Use PyMuPDF if pdfplumber fails
            use_ocr: Use OCR if regular text extraction fails

        Returns:
            str: Extracted text or None if extraction failed
        """
        pdf_file = None
        try:
            full_text = []
            pdf_file = io.BytesIO(pdf_bytes)

            # ✅ Attempt 1: Use pdfplumber (better for structured PDFs)
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    # Check if there are pages
                    if pdf.pages and len(pdf.pages) > 0:
                        pages_count = len(pdf.pages)
                        module_logger.send_message(
                            level_log=logging.DEBUG,
                            logger_name="FileManager",
                            message=f"pdfplumber: PDF has {pages_count} pages",
                            send_to_azure=True
                        )

                        for page_num, page in enumerate(pdf.pages, 1):
                            try:
                                # Enhanced text extraction
                                page_text = page.extract_text(
                                    x_tolerance=3,
                                    y_tolerance=3,
                                    layout=False
                                )

                                if page_text and page_text.strip():
                                    full_text.append(page_text)
                                else:
                                    module_logger.send_message(
                                        level_log=logging.DEBUG,
                                        logger_name="FileManager",
                                        message=f"pdfplumber page {page_num}: No text extracted",
                                        send_to_azure=True
                                    )

                            except Exception as page_error:
                                module_logger.send_message(
                                    level_log=logging.WARNING,
                                    logger_name="FileManager",
                                    message=f"pdfplumber error on page {page_num}: {page_error}",
                                    send_to_azure=True
                                )
                                continue
                    else:
                        module_logger.send_message(
                            level_log=logging.WARNING,
                            logger_name="FileManager",
                            message="pdfplumber: PDF has no pages, will try fallback",
                            send_to_azure=True
                        )

                    if full_text:
                        text = "\n".join(full_text)
                        module_logger.send_message(
                            level_log=logging.INFO,
                            logger_name="FileManager",
                            message=f"Successfully extracted {len(full_text)} pages from PDF using pdfplumber",
                            send_to_azure=True
                        )
                        return text

            except Exception as plumber_error:
                module_logger.send_message(
                    level_log=logging.WARNING,
                    logger_name="FileManager",
                    message=f"pdfplumber failed: {plumber_error}",
                    send_to_azure=True
                )

            # ✅ Attempt 2: Fallback to PyMuPDF (fitz) - more reliable for complex PDFs
            if use_fallback and not full_text:
                module_logger.send_message(
                    level_log=logging.DEBUG,
                    logger_name="FileManager",
                    message="Trying fallback method with PyMuPDF...",
                    send_to_azure=True
                )
                try:
                    pdf_file.seek(0)  # Return to the beginning of the file
                    doc = fitz.open(stream=pdf_file, filetype="pdf")

                    module_logger.send_message(
                        level_log=logging.DEBUG,
                        logger_name="FileManager",
                        message=f"PyMuPDF: PDF has {len(doc)} pages",
                        send_to_azure=True
                    )

                    for page_num, page in enumerate(doc, 1):
                        try:
                            page_text = page.get_text(
                                option="text",
                                sort=True
                            )

                            if page_text and page_text.strip():
                                full_text.append(page_text)
                            else:
                                module_logger.send_message(
                                    level_log=logging.DEBUG,
                                    logger_name="FileManager",
                                    message=f"PyMuPDF page {page_num}: No text extracted",
                                    send_to_azure=True
                                )

                        except Exception as page_error:
                            module_logger.send_message(
                                level_log=logging.WARNING,
                                logger_name="FileManager",
                                message=f"PyMuPDF error on page {page_num}: {page_error}",
                                send_to_azure=True
                            )
                            continue

                    doc.close()

                    if full_text:
                        text = "\n".join(full_text)
                        module_logger.send_message(
                            level_log=logging.INFO,
                            logger_name="FileManager",
                            message=f"Successfully extracted {len(full_text)} pages from PDF using PyMuPDF",
                            send_to_azure=True
                        )
                        return text

                except Exception as fitz_error:
                    module_logger.send_message(
                        level_log=logging.ERROR,
                        logger_name="FileManager",
                        message=f"PyMuPDF fallback also failed: {fitz_error}",
                        send_to_azure=True
                    )

            # ✅ Attempt 3: OCR for scanned documents
            if use_ocr and not full_text:
                module_logger.send_message(
                    level_log=logging.INFO,
                    logger_name="FileManager",
                    message="No text extracted by regular methods. Trying OCR...",
                    send_to_azure=True
                )

                ocr_text = self._extract_text_with_ocr(pdf_bytes)
                if ocr_text:
                    return ocr_text

            # ✅ If nothing worked
            module_logger.send_message(
                level_log=logging.ERROR,
                logger_name="FileManager",
                message="Failed to extract text from PDF using all available methods",
                send_to_azure=True
            )
            return None

        except Exception as e:
            module_logger.send_message(
                level_log=logging.ERROR,
                logger_name="FileManager",
                message=f"Unexpected error in convert_pdf_to_text: {e}",
                send_to_azure=True
            )
            return None
        finally:
            # Guaranteed file closure
            if pdf_file:
                try:
                    pdf_file.close()
                except:
                    pass


if __name__ == "__main__":
    file_manager = PDF_to_TXT_converter()
