# resume_parser.py

import os
import fitz  # PyMuPDF
import pdfplumber
import tempfile
import pytesseract
import logging
from pdf2image import convert_from_path

# Suppress PyMuPDF warnings
logging.getLogger('fitz').setLevel(logging.ERROR)

class ResumeParser:
    """
    Parser for extracting text from resume PDFs.
    Uses multiple strategies to maximize text extraction quality.
    """

    def __init__(self, ocr_enabled=False):
        """
        Initialize the resume parser.

        Parameters:
        -----------
        ocr_enabled : bool
            Whether to use OCR as a fallback for image-based PDFs
        """
        self.ocr_enabled = ocr_enabled

    def extract_text_pymupdf(self, pdf_path):
        """Extract text using PyMuPDF (fast and reliable)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text with PyMuPDF: {e}")
            return ""

    def extract_text_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber (good for structured content)"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"Error extracting text with pdfplumber: {e}")
            return ""

    def extract_text_ocr(self, pdf_path):
        """Extract text using OCR (for image-based PDFs)"""
        if not self.ocr_enabled:
            return ""

        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            text = ""

            # Process each page
            for image in images:
                # Perform OCR on the image
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"

            return text
        except Exception as e:
            print(f"Error extracting text with OCR: {e}")
            return ""

    def parse(self, file_path):
        """
        Extract text from a resume file.
        Tries multiple strategies and returns the best result.

        Parameters:
        -----------
        file_path : str
            Path to the resume file (PDF)

        Returns:
        --------
        dict
            Dictionary containing extracted text and metadata
        """
        # Check if file exists and is PDF
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        if not file_path.lower().endswith('.pdf'):
            return {"error": "Only PDF files are supported"}

        # Strategy 1: PyMuPDF extraction
        text_pymupdf = self.extract_text_pymupdf(file_path)

        # Strategy 2: PDFPlumber extraction
        text_pdfplumber = self.extract_text_pdfplumber(file_path)

        # Select the best extraction (heuristic: choose the longer text)
        text = text_pymupdf if len(text_pymupdf) >= len(text_pdfplumber) else text_pdfplumber

        # If text is very short, try OCR as a last resort
        if len(text.strip()) < 100 and self.ocr_enabled:
            text_ocr = self.extract_text_ocr(file_path)
            if len(text_ocr) > len(text):
                text = text_ocr
                extraction_method = "ocr"
            else:
                extraction_method = "pymupdf" if len(text_pymupdf) >= len(text_pdfplumber) else "pdfplumber"
        else:
            extraction_method = "pymupdf" if len(text_pymupdf) >= len(text_pdfplumber) else "pdfplumber"

        text = self.clean_extracted_text(text)

        return {
            "text": text,
            "extraction_method": extraction_method,
            "char_count": len(text),
            "word_count": len(text.split()),
            "filename": os.path.basename(file_path)
        }


    def clean_extracted_text(self, text):
        """Clean up extracted text by removing extra whitespace and fixing common issues"""
        import re

        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)

        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)

        # Fix common PDF extraction issues
        text = text.replace('•', '\n•')  # Ensure bullet points start on new lines

        # Remove page numbers and headers/footers (basic approach)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip lines that look like page numbers
            if re.match(r'^[0-9]+$', line.strip()):
                continue
            # Skip lines that look like headers/footers (often contain page numbers)
            if re.search(r'Page [0-9]+ of [0-9]+', line):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()



# Example usage
if __name__ == "__main__":
    parser = ResumeParser(ocr_enabled=True)
    result = parser.parse("path/to/resume.pdf")
    print(f"Extracted {result['word_count']} words using {result['extraction_method']}")
    print("First 500 characters:")
    print(result['text'][:500])