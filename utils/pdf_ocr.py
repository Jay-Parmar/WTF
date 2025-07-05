import pdfplumber
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_any_pdf(pdf_path):
    # Try text extraction first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            texts = []
            all_empty = True
            for page in pdf.pages:
                txt = page.extract_text()
                if txt and txt.strip():
                    all_empty = False
                texts.append(txt or "")
            if not all_empty:
                return "\n".join(texts)
    except Exception as e:
        print(f"pdfplumber error: {e}")
    # Fallback to OCR
    try:
        images = convert_from_path(pdf_path)
        ocr_texts = [pytesseract.image_to_string(img) for img in images]
        return "\n".join(ocr_texts)
    except Exception as e:
        print(f"OCR error: {e}")
        return ""
