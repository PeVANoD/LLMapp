from app.core.interfaces import IFileProcessor
from typing import List, Dict, Optional
from PIL import Image
import pytesseract
import PyPDF2
import docx
import logging
import io
import os
from pdf2image import convert_from_bytes
import tempfile

logger = logging.getLogger(__name__)

class FileProcessor(IFileProcessor):
    def __init__(self, tesseract_path: str = None, languages: str = 'rus+eng'):
        """
        Initialize file processor with Tesseract OCR
        
        Args:
            tesseract_path: Optional path to Tesseract executable
            languages: Languages for OCR (default: rus+eng)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.languages = languages
        self._check_tesseract()

    def _check_tesseract(self):
        """Verify Tesseract is installed and working"""
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.error(f"Tesseract not found or not working: {str(e)}")
            raise RuntimeError("Tesseract OCR is not properly installed")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy"""
        try:
            # Convert to grayscale
            image = image.convert('L')
            
            # Enhance contrast
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Apply sharpening
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            return image
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            return image

    def process_file(self, file_path: str) -> str:
        """
        Process file and extract text content
        
        Args:
            file_path: Path to file to process
            
        Returns:
            Extracted text content
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Process image file
                image = Image.open(file_path)
                return self.extract_text_from_image(image)
                
            elif file_path.lower().endswith('.pdf'):
                # Process PDF file
                return self._process_pdf(file_path)
                
            elif file_path.lower().endswith('.docx'):
                # Process Word document
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
                
            else:
                # Process as plain text
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise RuntimeError(f"Error processing file: {str(e)}")

    def _process_pdf(self, file_path: str) -> str:
        """
        Process PDF file - try text extraction first, fallback to OCR
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            # First try to extract text directly
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() for page in reader.pages])
                
                # If we got reasonable amount of text, return it
                if len(text.strip()) > 100:
                    return text
                    
            # If direct extraction failed, use OCR
            logger.info("Falling back to OCR for PDF")
            with open(file_path, 'rb') as f:
                return self._ocr_pdf(f.read())
                
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")

    def _ocr_pdf(self, pdf_bytes: bytes) -> str:
        """
        Convert PDF pages to images and perform OCR
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Extracted text from all pages
        """
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes)
            
            full_text = []
            for i, image in enumerate(images):
                try:
                    text = self.extract_text_from_image(image)
                    full_text.append(f"Page {i+1}:\n{text}")
                except Exception as e:
                    logger.error(f"Error processing page {i+1}: {str(e)}")
                    full_text.append(f"Page {i+1}: [OCR failed]")
                    
            return "\n\n".join(full_text)
        except Exception as e:
            logger.error(f"PDF OCR failed: {str(e)}")
            raise RuntimeError("Failed to perform OCR on PDF")

    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Configure Tesseract parameters
            config = f'--oem 3 --psm 6 -l {self.languages}'
            
            # Perform OCR
            text = pytesseract.image_to_string(
                processed_image,
                config=config
            )
            
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            raise RuntimeError(f"Failed to extract text from image: {str(e)}")

    def process_uploaded_file(self, file_bytes: bytes, filename: str) -> str:
        """
        Process uploaded file from bytes
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            
        Returns:
            Extracted text content
        """
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
                
            try:
                return self.process_file(tmp_path)
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing uploaded file {filename}: {str(e)}")
            raise RuntimeError(f"Error processing file: {str(e)}")