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

if not os.path.exists("chats.db"):
    from app.infrastructure.storage import SQLiteChatStorage
    storage = SQLiteChatStorage()
    storage._init_db()
# Модели Pydantic
class Message(BaseModel):
    role: str
    content: str

class ChatCreate(BaseModel):
    name: Optional[str] = None
    provider: str
    model: str

class FileInfo(BaseModel):
    type: str  # 'file' or 'image'
    name: str
    content: Optional[str] = None

class MessageRequest(BaseModel):
    message: str
    model: str
    use_web_search: bool = False
    files: List[FileInfo] = []

async def process_uploaded_file(file: UploadFile, file_type: str) -> dict:
    """Обрабатывает загруженный файл и возвращает информацию о нем"""
    if file_type not in ALLOWED_FILE_TYPES:
        raise ValueError(f"Invalid file type: {file_type}")
    
    if file.content_type not in ALLOWED_FILE_TYPES[file_type]:
        raise ValueError(f"Unsupported content type: {file.content_type} for type {file_type}")

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File size {file_size} exceeds maximum allowed size {MAX_FILE_SIZE}")

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

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
    chat = chat_storage.get_chat(chat_id)
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
    request: Request,
    message: str = Form(...),
    model: str = Form(...),
    use_web_search: bool = Form(False),
    files: List[UploadFile] = File([])
):
    try:
        form_data = await request.form()
        file_types = form_data.getlist("file_types")
        
        file_infos = []
        full_content = message
        
        for file, file_type in zip(files, file_types):
            try:
                file_info = await process_uploaded_file(file, file_type)
                file_infos.append(file_info)
                
                if file_info["content"]:
                    full_content += f"\n\n[Файл: {file_info['name']}]\n{file_info['content']}"
                else:
                    full_content += f"\n\n[Файл: {file_info['name']} - содержимое недоступно]"            
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue

        chat = chat_storage.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        if model != chat["model"]:
            chat_storage.update_chat_model(chat_id, model)
        
        # Create embedding from the full content (message + files)
        embedding = llm_client.embedding_service.create_embedding(full_content)
        
        user_message = {
            "role": "user", 
            "content": full_content,
            "files": file_infos
        }
        
        chat_storage.add_message(chat_id, user_message)
        chat_storage.save_embedding(chat_id, full_content, embedding)

        history = chat_storage.get_history(chat_id)
        
        client = llm_client.get_client(chat["provider"])
        response = client.generate_response(
            messages=history,
            model=model,
            use_web_search=use_web_search
        )
        
        assistant_message = {"role": "assistant", "content": response}
        chat_storage.add_message(chat_id, assistant_message)
        
        return {
            "response": response,
            "used_context": False,
            "web_search_performed": use_web_search,
            "files_processed": len(file_infos)
        }
    except Exception as e:
        logger.error(f"Unexpected error in add_message: {str(e)}", exc_info=True)
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
        upload_dir = f"uploads/{chat_id}"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = f"{upload_dir}/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        extracted_text, _ = extract_text_from_file(file_path, file.content_type)
        
        file_message = {
            "role": "user",
            "content": f"{message}\n\n[Attached file: {file.filename}]\n{extracted_text if extracted_text else 'Binary file, content not available'}"
        }
        
        file_info = {
            "name": file.filename,
            "type": file.content_type,
            "content": extracted_text or ""
        }
        chat_storage.add_message_with_file(chat_id, file_message, file_info)
        
        history = chat_storage.get_history(chat_id)
        
        client = llm_client.get_client_for_chat(chat_id)
        response = client.generate_response(
            messages=history,
            model=model,
            use_web_search=use_web_search
        )
        
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
    try:
        embedding = llm_client.embedding_service.create_embedding(text)
        test_chat_id = "test_chat_123"
        chat_storage.save_embedding(test_chat_id, text, embedding)
        similar = chat_storage.find_similar_texts(embedding)
        
        return {
            "original_text": text,
            "embedding_length": len(embedding),
            "similar_texts": similar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))