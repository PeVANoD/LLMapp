import os
import logging
from fastapi import FastAPI, HTTPException, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict
from app.core.interfaces import ILLMClient, IChatStorage
from app.infrastructure.storage import SQLiteChatStorage
from app.adapters.llm_clients import LMStudioClient
from pydantic import BaseModel
from app.config import Config

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
    docs_url="/docs",
    redoc_url="/redoc"
)

# Подключаем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Настраиваем Jinja2 для шаблонов
templates = Jinja2Templates(directory="templates")

# Конфигурация
class Config:
    LM_STUDIO_URL = "http://localhost:1234/v1"  # Порт должен совпадать с настройками LM Studio
    ACTIVE_MODEL = "local-model"  # Фиксированное имя для LM Studio API
    DEFAULT_MODEL = "google/gemma-3.1b"

def get_llm_client() -> ILLMClient:
    try:
        return LMStudioClient()  # Теперь используем конфиг по умолчанию
    except Exception as e:
        logger.error(f"LM Studio connection error: {str(e)}")
        raise HTTPException(status_code=500, detail="LM Studio unavailable")

def get_chat_storage() -> IChatStorage:
    return SQLiteChatStorage()

# Модели запросов
class MessageRequest(BaseModel):
    message: str
    model: str = Config.DEFAULT_MODEL

# Эндпоинты
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chats", response_class=HTMLResponse)
async def list_chats(request: Request, storage: IChatStorage = Depends(get_chat_storage)):
    chats = storage.get_all_chats()
    return templates.TemplateResponse(
        "chats_list.html",
        {"request": request, "chats": chats}
    )

@app.post("/chats/new", response_class=RedirectResponse)
async def create_new_chat(storage: IChatStorage = Depends(get_chat_storage)):
    chat_id = storage.create_chat()
    return RedirectResponse(f"/chat/{chat_id}", status_code=303)

@app.post("/chats/{chat_id}/delete", response_class=RedirectResponse)
async def delete_chat_redirect(
    chat_id: str,
    storage: IChatStorage = Depends(get_chat_storage)
):
    storage.delete_chat(chat_id)
    return RedirectResponse("/chats", status_code=303)

@app.get("/chat/{chat_id}", response_class=HTMLResponse)
async def get_chat_page(
    request: Request,
    chat_id: str,
    storage: IChatStorage = Depends(get_chat_storage)
):
    history = storage.get_history(chat_id)
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "chat_id": chat_id, "history": history}
    )

@app.post("/api/chats/{chat_id}/messages")
async def post_message_to_chat(
    chat_id: str,
    request_data: MessageRequest,
    llm: ILLMClient = Depends(get_llm_client),
    storage: IChatStorage = Depends(get_chat_storage)
):
    try:
        # Добавляем сообщение пользователя
        storage.add_message(chat_id, {
            "role": "user",
            "content": request_data.message
        })
        
        # Генерируем ответ
        response_text = llm.generate_response(
            messages=storage.get_history(chat_id),
            model=request_data.model
        )
        
        # Проверяем, что ответ не пустой
        if not response_text:
            raise ValueError("Empty response from LLM")
            
        # Добавляем ответ ассистента
        storage.add_message(chat_id, {
            "role": "assistant",
            "content": response_text
        })
        
        return {"response": response_text}
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"LLM processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)