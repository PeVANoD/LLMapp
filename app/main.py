import json
import os
from fastapi import FastAPI, File, Form, Request, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import logging
from app.adapters.file_processors import extract_text_from_file
from app.adapters.llm_clients import MultiLLMClient
from app.config import Config
from app.infrastructure.storage import SQLiteChatStorage
from app.configuration.files import MAX_FILE_SIZE, ALLOWED_FILE_TYPES

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация хранилища и клиента LLM
chat_storage = SQLiteChatStorage()
llm_client = MultiLLMClient(chat_storage)

# Модели Pydantic
class Message(BaseModel):
    role: str
    content: str

class ChatCreate(BaseModel):
    name: Optional[str] = None
    provider: str
    model: str  # Добавлено поле модели

class FileInfo(BaseModel):
    type: str  # 'file' or 'image'
    name: str
    content: Optional[str] = None

class MessageRequest(BaseModel):
    message: str
    model: str
    use_web_search: bool = False
    files: List[FileInfo] = []


# Вспомогательные функции
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
            content = (await file.read()).decode('utf-8')
        
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

def _prepare_messages_for_llm(
    self,
    history: List[Dict],
    message: str,
    file_infos: List[FileInfo],
    use_web_search: bool,
    chat_id: str
) -> List[Dict]:
    """Подготавливает сообщения для LLM с учетом контекста и файлов"""
    messages = []

    # Системное сообщение
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant. Use contextual information if it would be helpful in your response."
    }

    # Добавление контекста из других чатов
    if len(message) > 10:  # Не ищем контекст для очень коротких сообщений
        similar_messages = chat_storage.search_across_chats(
            llm_client.embedding_service.create_embedding(message)
        )
        if similar_messages:
            context_text = "\n".join(
                f"From chat '{msg['chat_name']}': {msg['text']}"
                for msg in similar_messages[:3] if msg['chat_id'] != chat_id
            )
            system_message["content"] += f"\n\nAdditional context:\n{context_text}"

    messages.append(system_message)

    # Добавление истории чата с файлами
    for msg in history:
        content = msg["content"]
        if msg.get("files"):
            file_content = "\n".join(
                f"File {f['name']} ({f['type']}): {f['content'][:1000]}..."
                if len(f['content']) > 1000 else f"File {f['name']} ({f['type']}): {f['content']}"
                for f in msg["files"]
            )
            content += f"\n\nAttached files:\n{file_content}"
        
        messages.append({
            "role": msg["role"],
            "content": content
        })

    # Веб-поиск (если активирован)
    if use_web_search:
        try:
            search_results = llm_client.perform_web_search(message)
            if search_results:
                search_text = "\n".join(
                    f"- [{res['title']}]({res['link']}): {res['snippet']}"
                    for res in search_results[:3]
                )
                messages[-1]["content"] += f"\n\nWeb search results:\n{search_text}"
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            messages[-1]["content"] += "\n\n[Web search unavailable]"

    return messages

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/chats/{provider}", response_class=HTMLResponse)
async def provider_chats(request: Request, provider: str):
    if provider not in ['ollama', 'lm_studio']:
        return RedirectResponse("/")
    
    try:
        chats = chat_storage.get_all_chats(provider)
        provider_display = "Ollama" if provider == "ollama" else "LM Studio"
        
        return templates.TemplateResponse("chat_list.html", {
            "request": request,
            "provider": provider,
            "provider_display": provider_display,
            "chats": chats,
            "has_chats": len(chats) > 0
        })
    except Exception as e:
        logger.error(f"Error loading chats: {str(e)}", exc_info=True)
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Ошибка загрузки чатов: {str(e)}",
            "provider": provider
        })
   
@app.get("/chat/new/{provider}", response_class=HTMLResponse)
async def new_chat(request: Request, provider: str):
    if provider not in ['ollama', 'lm_studio']:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    try:
        client = llm_client.get_client(provider)
        if not client:
            raise HTTPException(status_code=400, detail="Invalid provider")
        
        models = client.list_models()
        if not models:
            raise HTTPException(
                status_code=503,
                detail=f"No models available for {provider}. Please check your local LLM service."
            )
        
        provider_display = "Ollama" if provider == "ollama" else "LM Studio"
        
        return templates.TemplateResponse("new_chat.html", {
            "request": request,
            "provider": provider,
            "provider_display": provider_display,
            "models": models
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in new_chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load models: {str(e)}"
        )

@app.post("/api/chats", response_model=dict)
async def create_chat(chat_data: ChatCreate):
    if not chat_data.model:
        raise HTTPException(status_code=400, detail="Model is required")
    
    chat_id = chat_storage.create_chat(chat_data.provider, chat_data.model)

    if chat_data.name:
        chat_storage.rename_chat(chat_id, chat_data.name)
    return {"chat_id": chat_id}

@app.get("/chat/{chat_id}", response_class=HTMLResponse)
async def get_chat(request: Request, chat_id: str):
    chat = next((c for c in chat_storage.get_all_chats() if c["chat_id"] == chat_id), None)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    models = llm_client.get_client(chat["provider"]).list_models()
    history = chat_storage.get_history(chat_id)
    
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "chat_id": chat_id,
        "provider": chat["provider"],
        "models": models,
        "current_model": chat["model"],
        "history": history,
        "chat_name": chat_storage.get_chat_name(chat_id) or f"Chat {chat_id[:8]}"
    })

@app.get("/api/search", response_model=List[dict])
async def search_messages(query: str):
    try:
        embedding = llm_client.embedding_service.create_embedding(query)
        results = chat_storage.search_across_chats(embedding)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/{chat_id}/message")
async def add_message(
    chat_id: str,
    message: str = Form(...),
    model: str = Form(...),
    use_web_search: bool = Form(False),   
    files: List[UploadFile] = File([]),
    file_types: List[str] = Form([])
):
    try:
        # 1. Получаем информацию о чате
        chat = next((c for c in chat_storage.get_all_chats() if c["chat_id"] == chat_id), None)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # 2. Обновляем модель, если она изменилась
        if model != chat["model"]:
            chat_storage.update_chat_model(chat_id, model)
        
        # 3. Обрабатываем загруженные файлы
        file_infos = []
        for file, file_type in zip(files, file_types):
            try:
                file_info = await process_uploaded_file(file, file_type)
                file_infos.append(file_info)
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
                file_infos.append({
                    "type": file_type,
                    "name": file.filename,
                    "content": f"[Error processing file: {str(e)}]"
                })
                continue
        
        # 4. Создаем эмбеддинг сообщения
        embedding = llm_client.embedding_service.create_embedding(message)
        
        # 5. Сохраняем сообщение пользователя с файлами (если есть)
        user_message = {"role": "user", "content": message}
        if file_infos:
            user_message["files"] = file_infos
        chat_storage.add_message(chat_id, user_message)
        chat_storage.save_embedding(chat_id, message, embedding)

        # 6. Получаем историю чата
        history = chat_storage.get_history(chat_id)
        
        # 7. Ищем релевантный контекст из других чатов
        context_messages = []
        if len(message) > 10:  # Не ищем контекст для очень коротких сообщений
            similar_messages = chat_storage.search_across_chats(embedding)
            for msg in similar_messages[:3]:  # Берем топ-3
                if msg['chat_id'] != chat_id:  # Исключаем сообщения из текущего чата
                    context_messages.append({
                        'role': 'context',
                        'content': f"From chat '{msg['chat_name']}': {msg['text']}"
                    })
        
        # 8. Веб-поиск
        search_context = ""
        if use_web_search:
            try:
                search_results = llm_client.perform_web_search(message)
                if search_results:
                    search_context = "\n\nWeb search results:\n"
                    search_context += "\n".join(
                        f"- [{result['title']}]({result['link']}): {result['snippet']}"
                        for result in search_results[:3]
                    )
            except Exception as e:
                logger.error(f"Web search failed: {str(e)}")
                search_context = "\n\n[Web search unavailable]"
        
        # 9. Подготавливаем сообщения для LLM включая содержимое файлов
        messages_for_llm = []
        
        # 9.1. Добавляем системное сообщение с инструкциями
        system_message = {
            "role": "system",
            "content": "Ты полезный ассистент. Используй контекст и отвечай всегда правильно и формально."
        }
        
        # 9.2. Добавляем контекст из других чатов
        if context_messages:
            context_text = "\n".join([msg['content'] for msg in context_messages])
            system_message['content'] += f"\n\nAdditional context:\n{context_text}"
        
        messages_for_llm.append(system_message)
        
        # 9.3. Добавляем историю текущего чата с содержимым файлов
        # Добавляем историю и файлы
        for msg in history:
            content = msg["content"]
            
            # Добавляем содержимое файлов к сообщению
            if "files" in msg:
                file_content = "\n[Attached files]:\n"
                for file in msg["files"]:
                    if file['type'].startswith('image/'):
                        file_content += f"- Image: {file['name']}\n"
                    else:
                        # Для текстовых файлов добавляем содержимое
                        file_content += f"- File: {file['name']}\n{file['content'][:2000]}\n\n"
                content += file_content
                
            messages_for_llm.append({
                "role": msg["role"],
                "content": content
            })
        
        # 9.4. Добавляем результаты веб-поиска к последнему сообщению
        if search_context:
            messages_for_llm[-1]["content"] += search_context
        
        # 10. Генерируем ответ
        client = llm_client.get_client(chat["provider"])
        try:
            response = client.generate_response(
                messages=messages_for_llm,
                model=model,
                use_web_search=use_web_search
            )
            
            # 11. Сохраняем ответ ассистента
            assistant_message = {"role": "assistant", "content": response}
            chat_storage.add_message(chat_id, assistant_message)
            
            return {
                "response": response,
                "used_context": bool(context_messages),
                "web_search_performed": use_web_search,
                "files_processed": len(file_infos)
            }
            
        except ValueError as e:
            logger.error(f"Response generation error: {str(e)}")
            raise HTTPException(status_code=502, detail=str(e))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
       
@app.get("/api/providers/{provider}/models")
async def get_provider_models(provider: str):
    client = llm_client.get_client(provider)
    if not client:
        raise HTTPException(status_code=404, detail="Provider not found")
    return {"models": client.list_models()}

@app.put("/api/chats/{chat_id}/rename")
async def rename_chat(chat_id: str, request: Request):
    try:
        data = await request.json()
        new_name = data.get("name", "").strip()
        
        if not new_name:
            raise HTTPException(status_code=400, detail="New name cannot be empty")
            
        if len(new_name) > 100:
            raise HTTPException(status_code=400, detail="Name is too long (max 100 chars)")
            
        if not chat_storage.rename_chat(chat_id, new_name):
            raise HTTPException(status_code=404, detail="Chat not found")
            
        return {"status": "success", "new_name": new_name}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        logger.error(f"Error renaming chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    chat_storage.delete_chat(chat_id)
    return {"status": "success"}

@app.get("/api/health/{provider}")
async def health_check(provider: str):
    client = llm_client.get_client(provider)
    try:
        models = client.list_models()
        return {"status": "ok", "models": bool(models)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    
@app.post("/api/upload/{chat_id}")
async def upload_file(
    chat_id: str,
    request: Request,
    file: UploadFile = File(...),
    message: str = Form(""),
    model: str = Form(...),
    use_web_search: bool = Form(False)
):
    try:
        # Создаем временную директорию, если ее нет
        upload_dir = f"uploads/{chat_id}"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Сохраняем файл
        file_path = f"{upload_dir}/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Извлекаем текст из файла
        extracted_text, _ = extract_text_from_file(file_path, file.content_type)
        
        # Формируем сообщение с содержимым файла
        file_message = {
            "role": "user",
            "content": f"{message}\n\n[Прикрепленный файл: {file.filename}]\n{extracted_text if extracted_text else 'Бинарный файл, содержимое недоступно'}"
        }
        
        # Добавляем файл в хранилище
        file_info = {
            "name": file.filename,
            "type": file.content_type,
            "content": extracted_text or ""
        }
        chat_storage.add_message_with_file(chat_id, file_message, file_info)
        
        # Получаем историю с новым сообщением
        history = chat_storage.get_history(chat_id)
        
        # Генерируем ответ
        client = llm_client.get_client_for_chat(chat_id)
        response = client.generate_response(
            messages=history,
            model=model,
            use_web_search=use_web_search
        )
        
        # Сохраняем ответ ассистента
        assistant_message = {"role": "assistant", "content": response}
        chat_storage.add_message(chat_id, assistant_message)
        
        return {
            "response": response,
            "filename": file.filename,
            "extracted_text": extracted_text
        }
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/embedding")
async def test_embedding(text: str = "test text"):
    """Тестовый эндпоинт для проверки работы эмбеддингов"""
    try:
        # Создаем эмбеддинг
        embedding = llm_client.embedding_service.create_embedding(text)
        
        # Сохраняем в базу (используем test_chat_id для тестов)
        test_chat_id = "test_chat_123"
        chat_storage.save_embedding(test_chat_id, text, embedding)
        
        # Ищем похожие тексты
        similar = chat_storage.find_similar_texts(embedding)
        
        return {
            "original_text": text,
            "embedding_length": len(embedding),
            "similar_texts": similar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))