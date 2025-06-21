import os
import logging
from fastapi import FastAPI, HTTPException, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import PlainTextResponse 
from typing import List, Dict
from app.core.interfaces import ILLMClient, IChatStorage
from app.infrastructure.storage import SQLiteChatStorage
from app.adapters.llm_clients import LMStudioClient
from pydantic import BaseModel,Field
from app.config import Config
from typing import Optional


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
    model: str = Field(default=Config.DEFAULT_MODEL, description="Модель для генерации ответа")
    use_web: bool = Field(default=False, description="Использовать веб-поиск")
    max_tokens: Optional[int] = Field(
        default=None,
        ge=50,
        le=4000,
        description="Максимальное количество токенов (50-4000). Если None, используется значение по умолчанию модели"
    )

# Эндпоинты
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

import sqlite3
@app.get("/chats", response_class=HTMLResponse)
async def list_chats(request: Request, storage: IChatStorage = Depends(get_chat_storage)):
    """Страница со списком всех чатов"""
    try:
        chats = storage.get_all_chats()
        return templates.TemplateResponse(
            "chats_list.html",
            {"request": request, "chats": chats}
        )
    except sqlite3.OperationalError as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Database error. Please try again later."
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
    """Страница чата с названием"""
    history = storage.get_history(chat_id)
    chat_name = storage.get_chat_name(chat_id)
    
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "chat_id": chat_id,
            "chat_name": chat_name if chat_name != chat_id[:8] else None,
            "history": history
        }
    )

#----------------------------------------------------#
#----------------------------------------------------#
#---------------------ДЛЯ--WEB-----------------------#
#----------------------------------------------------#
#----------------------------------------------------#

from app.core.interfaces import IWebSearch
from app.adapters.web_search import DuckDuckGoSearch
from app.adapters.web_search import GoogleSearch 
from datetime import datetime

def get_web_search() -> Optional[IWebSearch]:
    # Можно переключаться между Google и DuckDuckGo
    if Config.USE_DUCKDUCKGO:
        return DuckDuckGoSearch()
    elif Config.GOOGLE_API_KEY and Config.GOOGLE_ENGINE_ID:
        return GoogleSearch(Config.GOOGLE_API_KEY, Config.GOOGLE_ENGINE_ID)
    return None

@app.post("/api/chats/{chat_id}/messages")
async def post_message_to_chat(
    chat_id: str,
    request_data: MessageRequest,
    llm: ILLMClient = Depends(get_llm_client),
    storage: IChatStorage = Depends(get_chat_storage),
    web_search: Optional[IWebSearch] = Depends(get_web_search)
):
    try:
        # Сохраняем сообщение пользователя
        storage.add_message(chat_id, {"role": "user", "content": request_data.message})
        
        # Получаем веб-результаты (если включено)
        web_context = ""
        if request_data.use_web and web_search:
            search_results = web_search.search(request_data.message)
            web_context = (
                f"🔍 Результаты поиска ({datetime.now().strftime('%d.%m.%Y %H:%M')}):\n"
                f"{search_results}\n\n"
            )
            storage.add_message(chat_id, {"role": "system", "content": web_context})

        # Формируем промт для LLM
        messages = [
            {"role": "system", "content": "Ты - полезный AI ассистент. " + 
             ("Используй следующие данные при ответе:\n" + web_context if web_context else "")},
            {"role": "user", "content": request_data.message}
        ]
        
        # Генерируем ответ
        response = llm.generate_response(
            messages=messages,
            model=request_data.model,
            max_tokens=request_data.max_tokens
        )
        
        storage.add_message(chat_id, {"role": "assistant", "content": response})
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


#----------------------------------------------------#
#----------------------------------------------------#
#----------------------------------------------------#
#----------------------------------------------------#
#----------------------------------------------------#


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


class RenameChatRequest(BaseModel):
    new_name: str

@app.post("/api/chats/{chat_id}/rename")
async def rename_chat(
    chat_id: str,
    request: RenameChatRequest,
    storage: IChatStorage = Depends(get_chat_storage)
):
    """Эндпоинт для переименования чата"""
    try:
        storage.rename_chat(chat_id, request.new_name)
        return {"status": "success", "new_name": request.new_name}
    except ValueError as e:
        # Ошибки валидации (пустое имя, чат не найден)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка переименования чата {chat_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера при переименовании чата"
        )

@app.get("/api/chats/{chat_id}/name")
async def get_chat_name(
    chat_id: str,
    storage: IChatStorage = Depends(get_chat_storage)
):
    return {"name": storage.get_chat_name(chat_id)}


@app.get("/debug-web-search")
async def debug_web_search(query: str, web_search: IWebSearch = Depends(get_web_search)):
    content = web_search.search(query)
    return {"query": query, "results": content}  

@app.get("/test-ddg")
async def test_ddg(query: str):
    searcher = DuckDuckGoSearch()
    return PlainTextResponse(searcher.search(query))