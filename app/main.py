# app/main.py (основные изменения)
import json
import os
from fastapi import FastAPI, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from app.adapters.file_processors import FileProcessor
from app.adapters.llm_clients import MultiLLMClient
from app.infrastructure.storage import SQLiteChatStorage
import logging
from fastapi import UploadFile, File
from typing import List
import os
import uuid
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

chat_storage = SQLiteChatStorage()
llm_client = MultiLLMClient()

# Модели Pydantic
class Message(BaseModel):
    role: str
    content: str

class ChatCreate(BaseModel):
    name: Optional[str] = None
    provider: str
    model: str  # Добавлено поле модели

# Добавьте эту модель
class FileInfo(BaseModel):
    type: str  # 'file' or 'image'
    name: str
    content: Optional[str] = None

# Модифицируйте MessageRequest
class MessageRequest(BaseModel):
    message: str
    model: str
    use_web_search: bool = False
    files: List[FileInfo] = []
    


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
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

@app.post("/api/process-file")
async def process_file(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        processor = FileProcessor()
        text = processor.process_uploaded_file(file_bytes, file.filename)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

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
        # Get chat info
        chat = next((c for c in chat_storage.get_all_chats() if c["chat_id"] == chat_id), None)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Update model if changed
        if model != chat["model"]:
            chat_storage.update_chat_model(chat_id, model)
        
        # Process files
        file_infos = []
        for file, file_type in zip(files, file_types):
            content = await process_uploaded_file(file, file_type)
            file_infos.append({
                "type": file_type,
                "name": file.filename,
                "content": content
            })
        
        # Add user message
        user_message = {"role": "user", "content": message}
        if file_infos:
            user_message["files"] = file_infos
        chat_storage.add_message(chat_id, user_message)
        
        # Get full history
        history = chat_storage.get_history(chat_id)
        
        # If web search is enabled, perform search
        search_context = ""
        if use_web_search:
            try:
                search_results = llm_client.perform_web_search(message)
                if search_results:
                    search_context = "\n\nHere are some recent web search results that might be relevant:\n"
                    search_context += "\n".join(
                        f"- [{result['title']}]({result['link']}): {result['snippet']}"
                        for result in search_results[:3]
                    )
            except Exception as e:
                logger.error(f"Web search failed: {str(e)}")
                search_context = "\n\n[Web search unavailable at this time]"
        
        # Generate response - include file content in context
        client = llm_client.get_client(chat["provider"])
        try:
            # Prepare messages with file content
            messages_with_context = []
            for msg in history:
                if msg.get("files"):
                    file_content = "\n".join(
                        f"File {f['name']} ({f['type']}): {f['content'][:1000]}..."
                        for f in msg["files"]
                    )
                    messages_with_context.append({
                        "role": msg["role"],
                        "content": f"{msg['content']}\n\nAttached files:\n{file_content}"
                    })
                else:
                    messages_with_context.append(msg)
            
            if search_context:
                messages_with_context[-1]["content"] += search_context
            
            response = client.generate_response(
                messages=messages_with_context,
                model=model
            )
            
            # Add assistant response
            assistant_message = {"role": "assistant", "content": response}
            chat_storage.add_message(chat_id, assistant_message)
            
            return {"response": response}
            
        except ValueError as e:
            logger.error(f"Response generation error: {str(e)}")
            raise HTTPException(status_code=502, detail=str(e))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

async def process_uploaded_file(file: UploadFile, file_type: str) -> str:
    """Process uploaded file and return its content as string"""
    try:
        if file_type == 'image':
            # For images, read as base64
            contents = await file.read()
            import base64
            return f"data:{file.content_type};base64,{base64.b64encode(contents).decode('utf-8')}"
        else:
            # For text files, read as text
            return (await file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file {file.filename}")
    
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