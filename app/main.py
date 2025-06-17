import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional, List, Dict
from pydantic import BaseModel
from PIL import Image
import io
from app.core.services import EmbeddingService
from app.core.interfaces import IEmbeddingService

# Импорт интерфейсов и реализаций
from app.core.interfaces import (
    ILLMClient,
    IEmbeddingService,
    IFileProcessor,
    IWebSearch,
    IChatStorage,
    IFileStorage
)

from app.adapters.llm_clients import OllamaClient, LMStudioClient
from app.adapters.file_processors import FileProcessor
from app.adapters.web_search import SerpAPISearch
from app.infrastructure.storage import SQLiteChatStorage, FileStorage
from app.core.interfaces import IEmbeddingService

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация приложения
app = FastAPI(
    title="Local LLM Chat API",
    version="1.0.0",
    description="API for interacting with locally deployed LLMs with multimodal support",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Конфигурация
class Config:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
    LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistral:7b")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    FILE_STORAGE_DIR = os.getenv("FILE_STORAGE_DIR", "file_storage")
    TESSERACT_PATH = os.getenv("TESSERACT_PATH", r'C:\Program Files\Tesseract-OCR\tesseract.exe')

# Инициализация зависимостей
def get_llm_client() -> ILLMClient:
    try:
        return OllamaClient(Config.OLLAMA_URL)
    except Exception as e:
        logger.warning(f"Failed to connect to Ollama: {str(e)}, trying LM Studio")
        try:
            return LMStudioClient(Config.LM_STUDIO_URL)
        except Exception as e:
            logger.error(f"Failed to connect to both Ollama and LM Studio: {str(e)}")
            raise HTTPException(status_code=500, detail="No LLM backend available")

def get_embedding_service() -> IEmbeddingService:
    return EmbeddingService(Config.EMBEDDING_MODEL)

def get_file_processor() -> IFileProcessor:
    return FileProcessor(Config.TESSERACT_PATH)

def get_web_searcher() -> IWebSearch:
    return SerpAPISearch()

def get_chat_storage() -> IChatStorage:
    return SQLiteChatStorage()

def get_file_storage() -> IFileStorage:
    return FileStorage(Config.FILE_STORAGE_DIR)

# Модели запросов/ответов
class MessageModel(BaseModel):
    role: str
    content: str

class ChatCreateResponse(BaseModel):
    chat_id: str

class MessageRequest(BaseModel):
    message: str
    use_web: bool = False
    use_embeddings: bool = False
    model: Optional[str] = None

class MessageResponse(BaseModel):
    response: str
    files: Optional[List[str]] = None

class ChatHistoryResponse(BaseModel):
    history: List[MessageModel]

class EmbeddingCreateRequest(BaseModel):
    text: str

class EmbeddingSearchRequest(BaseModel):
    query: str
    top_k: int = 3

class EmbeddingSearchResult(BaseModel):
    text: str
    score: float

class EmbeddingSearchResponse(BaseModel):
    results: List[EmbeddingSearchResult]

class FileUploadResponse(BaseModel):
    filename: str
    content_preview: str
    file_path: str

# API Endpoints
@app.get("/")
async def root():
    """Корневой эндпоинт для проверки работы API"""
    return {
        "message": "Local LLM Chat API is running!",
        "docs": "/docs",
        "redoc": "/redoc"
    }
@app.post("/chats", response_model=ChatCreateResponse)
def create_chat(storage: IChatStorage = Depends(get_chat_storage)):
    """Создает новый чат и возвращает его ID"""
    try:
        chat_id = storage.create_chat()
        logger.info(f"Created new chat: {chat_id}")
        return {"chat_id": chat_id}
    except Exception as e:
        logger.error(f"Error creating chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create chat")

@app.post("/chats/{chat_id}/messages", response_model=MessageResponse)
async def send_message(
    chat_id: str,
    message: str = Form(...),
    use_web: bool = Form(False),
    use_embeddings: bool = Form(False),
    model: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    llm: ILLMClient = Depends(get_llm_client),
    storage: IChatStorage = Depends(get_chat_storage),
    file_processor: IFileProcessor = Depends(get_file_processor),
    web_searcher: IWebSearch = Depends(get_web_searcher),
    embedding_service: IEmbeddingService = Depends(get_embedding_service),
    file_storage: IFileStorage = Depends(get_file_storage)
):
    """Отправляет сообщение в указанный чат"""
    try:
        # Проверка существования чата
        if not storage.get_history(chat_id):
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Обработка файла, если есть
        file_content = ""
        extracted_text = ""
        files_in_response = []
        
        if file:
            file_data = await file.read()
            file_path = file_storage.save_file(file_data, file.filename)
            
            # Для изображений - обработка как мультимодального ввода
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image = Image.open(io.BytesIO(file_data))
                    
                    # Попытка мультимодального ответа, если поддерживается
                    if hasattr(llm, 'generate_response_with_image'):
                        extracted_text = file_processor.extract_text_from_image(image)
                        storage.add_message(chat_id, {"role": "user", "content": f"{message}\n[Image: {file.filename}]\nExtracted text: {extracted_text}"})
                        
                        # Генерация ответа с изображением
                        response = llm.generate_response_with_image(
                            storage.get_history(chat_id),
                            image,
                            model or Config.DEFAULT_MODEL
                        )
                        storage.add_message(chat_id, {"role": "assistant", "content": response})
                        return {"response": response, "files": [file.filename]}
                    
                    # Если мультимодальность не поддерживается - просто извлечь текст
                    extracted_text = file_processor.extract_text_from_image(image)
                    file_content = f"\n[Прикреплено изображение: {file.filename}]\nИзвлеченный текст: {extracted_text}"
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
                    file_content = f"\n[Ошибка обработки изображения: {str(e)}]"
            else:
                # Обработка других типов файлов
                extracted_text = file_processor.process_file(file_path)
                file_content = f"\n[Прикреплён файл: {file.filename}]\n{extracted_text}"
                
                # Генерация файла в ответ (пример - создание PDF с ответом)
                # В реальном приложении здесь была бы логика генерации файлов
                files_in_response.append(file.filename)
        
        # Добавление сообщения пользователя
        full_message = message + file_content
        storage.add_message(chat_id, {"role": "user", "content": full_message})
        
        # Использование эмбеддингов, если требуется
        if use_embeddings:
            similar_texts = embedding_service.search_similar(message, top_k=3)
            if similar_texts:
                context = "\n".join([f"Контекст из базы знаний: {res['text']}" for res in similar_texts])
                storage.add_message(chat_id, {"role": "system", "content": context})
        
        # Веб-поиск при необходимости
        if use_web:
            web_data = web_searcher.search(message)
            storage.add_message(chat_id, {"role": "system", "content": f"Веб-контекст: {web_data}"})
        
        # Генерация ответа
        response = llm.generate_response(
            storage.get_history(chat_id),
            model or Config.DEFAULT_MODEL
        )
        storage.add_message(chat_id, {"role": "assistant", "content": response})
        
        return {"response": response, "files": files_in_response}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/chats/{chat_id}/messages", response_model=ChatHistoryResponse)
def get_chat_history(
    chat_id: str,
    storage: IChatStorage = Depends(get_chat_storage)
):
    """Возвращает историю сообщений чата"""
    try:
        history = storage.get_history(chat_id)
        if not history:
            raise HTTPException(status_code=404, detail="Chat not found or empty")
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get chat history")

@app.delete("/chats/{chat_id}")
def delete_chat(
    chat_id: str,
    storage: IChatStorage = Depends(get_chat_storage)
):
    """Удаляет чат и всю его историю"""
    try:
        storage.delete_chat(chat_id)
        logger.info(f"Deleted chat: {chat_id}")
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Error deleting chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete chat")

@app.post("/embeddings", response_model=dict)
def create_embedding(
    request: EmbeddingCreateRequest,
    embedding_service: IEmbeddingService = Depends(get_embedding_service)
):
    """Создает эмбеддинг для текста"""
    try:
        embedding = embedding_service.create_embedding(request.text)
        logger.info(f"Created embedding for text: {request.text[:50]}...")
        return {"status": "created", "text": request.text}
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create embedding")

@app.post("/embeddings/search", response_model=EmbeddingSearchResponse)
def search_embeddings(
    request: EmbeddingSearchRequest,
    embedding_service: IEmbeddingService = Depends(get_embedding_service)
):
    """Ищет похожие тексты по эмбеддингам"""
    try:
        results = embedding_service.search_similar(request.query, request.top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search embeddings")

@app.post("/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    file_processor: IFileProcessor = Depends(get_file_processor),
    file_storage: IFileStorage = Depends(get_file_storage)
):
    """Загружает файл и возвращает извлеченный текст"""
    try:
        file_data = await file.read()
        file_path = file_storage.save_file(file_data, file.filename)
        
        content = file_processor.process_file(file_path)
        return {
            "filename": file.filename,
            "content_preview": content[:1000] + "..." if len(content) > 1000 else content,
            "file_path": file_path
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.get("/files/{filename}")
def download_file(
    filename: str,
    file_storage: IFileStorage = Depends(get_file_storage)
):
    """Скачивает ранее загруженный файл"""
    try:
        file_data = file_storage.get_file(filename)
        return FileResponse(
            io.BytesIO(file_data),
            media_type="application/octet-stream",
            filename=filename
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download file")

@app.delete("/files/{filename}")
def delete_file(
    filename: str,
    file_storage: IFileStorage = Depends(get_file_storage)
):
    """Удаляет загруженный файл"""
    try:
        file_storage.delete_file(filename)
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete file")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")