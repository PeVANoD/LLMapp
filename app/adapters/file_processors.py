import os
import tempfile
from typing import Tuple, Optional
import PyPDF2
import docx
from fastapi import UploadFile
from pptx import Presentation
import logging
from PIL import Image
import pytesseract
from docx import Document
import pandas as pd
from io import BytesIO
import chardet
from app.configuration.files import MAX_FILE_SIZE, ALLOWED_FILE_TYPES

logger = logging.getLogger(__name__)

async def process_uploaded_file(file: UploadFile, file_type: str) -> dict:
    """Обрабатывает загруженный файл и возвращает информацию о нем"""
    # Проверка типа файла
    if file_type not in ALLOWED_FILE_TYPES:
        raise ValueError(f"Invalid file type: {file_type}")
    
    if file.content_type not in ALLOWED_FILE_TYPES[file_type]:
        raise ValueError(f"Unsupported content type: {file.content_type} for type {file_type}")

    # Проверка размера файла
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File size {file_size} exceeds maximum allowed size {MAX_FILE_SIZE}")

    # Обработка содержимого
    try:
        if file_type == 'image':
            contents = await file.read()
            import base64
            content = f"data:{file.content_type};base64,{base64.b64encode(contents).decode('utf-8')}"
        else:
            # Создаем временный файл для обработки
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                contents = await file.read()
                temp_file.write(contents)
                temp_file_path = temp_file.name
            
            # Извлекаем текст из файла
            extracted_text, _ = extract_text_from_file(temp_file_path, file.content_type)
            content = extracted_text if extracted_text else "Бинарный файл, текст не извлечен"
            
            # Удаляем временный файл
            os.unlink(temp_file_path)
        
        return {
            "type": file_type,
            "name": file.filename,
            "content": content,
            "content_type": file.content_type,
            "size": file_size
        }
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")
    finally:
        await file.close()

def extract_text_from_pdf(file_stream) -> str:
    """Извлекает текст из PDF"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(file_stream)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}", exc_info=True)
    return text

def extract_text_from_file(file_path: str, file_type: str) -> Tuple[Optional[str], Optional[bytes]]:
    """Извлекает текст из файлов различных форматов"""
    try:
        text = None
        image_data = None
        
        if file_type == 'application/pdf':
            text = extract_text_from_pdf(file_path)
        elif file_type == 'text/plain':
            with open(file_path, 'rb') as f:
                content = f.read()
                encoding = chardet.detect(content)['encoding'] or 'utf-8'
                text = content.decode(encoding, errors='replace')
        elif file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                          'application/msword']:
            text = extract_text_from_docx(file_path)
        elif file_type in ['application/vnd.openxmlformats-officedocument.presentationml.presentation',
                         'application/vnd.ms-powerpoint']:
            text = extract_text_from_pptx(file_path)
        elif file_type.startswith('image/'):
            text = extract_text_from_image(file_path)
            with open(file_path, 'rb') as f:
                image_data = f.read()
        
        return text, image_data
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
        return None, None

def extract_text_from_pdf(file_path: str) -> str:
    """Извлекает текст из PDF"""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}", exc_info=True)
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Извлекает текст из DOCX"""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"DOCX extraction error: {str(e)}", exc_info=True)
        # Fallback для поврежденных DOCX файлов
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                encoding = chardet.detect(content)['encoding'] or 'utf-8'
                return content.decode(encoding, errors='replace')
        except Exception as fallback_e:
            logger.error(f"DOCX fallback extraction failed: {str(fallback_e)}", exc_info=True)
            return ""

def extract_text_from_pptx(file_path: str) -> str:
    """Извлекает текст из PPTX"""
    try:
        prs = Presentation(file_path)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"PPTX extraction error: {str(e)}", exc_info=True)
        return ""

def extract_text_from_image(file_path: str) -> str:
    """Извлекает текст из изображения с помощью OCR"""
    try:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}", exc_info=True)
        return ""