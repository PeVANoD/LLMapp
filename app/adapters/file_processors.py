from app.core.interfaces import IFileProcessor
from typing import List, Dict
from PIL import Image
import pytesseract
import PyPDF2
import docx
import logging

logger = logging.getLogger(__name__)

class FileProcessor(IFileProcessor):
    def __init__(self, tesseract_path: str = r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def process_file(self, file_path: str) -> str:
        try:
            if file_path.endswith(('.png', '.jpg', '.jpeg')):
                return pytesseract.image_to_string(Image.open(file_path))
            elif file_path.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    return "\n".join([page.extract_text() for page in reader.pages])
            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return f"Error processing file: {str(e)}"

    def extract_text_from_image(self, image: Image.Image) -> str:
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return f"Error extracting text from image: {str(e)}"
