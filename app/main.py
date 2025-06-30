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
from app.utils.benchmark import benchmark
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
import time
from app.utils.text_metrics import calculate_text_metrics, calculate_file_complexity
import numpy as np
from app.utils.math_helpers import cosine_similarity, moving_average

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

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")

# Добавим хранение метрик при обработке файлов
@benchmark.time_it('text')
def process_text_metrics(text: str, file_type: str = 'text'):
    if file_type == 'text':
        return calculate_text_metrics(text)
    else:
        return calculate_file_complexity(text, file_type)

@app.get("/api/benchmark/file_processing")
async def get_file_benchmark_stats():
    return benchmark.get_all_stats()

@app.get("/api/benchmark/text_metrics")
async def analyze_text_metrics(text: str):
    start_time = time.perf_counter()
    metrics = calculate_text_metrics(text)
    processing_time = time.perf_counter() - start_time
    benchmark.add_metrics('text', processing_time, metrics['complexity_score'])
    return metrics

@app.get("/api/benchmark/file_complexity")
async def get_file_complexity_stats():
    return {
        file_type: benchmark.get_advanced_stats(file_type)
        for file_type in benchmark.stats.keys()
    }

@app.get("/api/benchmark/complexity_vs_time")
async def get_complexity_time_correlation():
    data = []
    for file_type, file_data in benchmark.stats.items():
        times = list(file_data['times'])
        complexities = list(file_data['complexities'])
        ratios = list(file_data['ratios'])
        
        for i in range(len(times)):
            data.append({
                'type': file_type,
                'complexity': complexities[i],
                'time': times[i],
                'ratio': ratios[i]
            })
    return data

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def calculate_chat_efficiency(history):
    processing_times = []
    complexities = []
    
    for msg in history:
        # Get processing time - either from direct field or metrics
        processing_time = msg.get('processing_time')
        if processing_time is None and 'metrics' in msg:
            processing_time = msg['metrics'].get('processing_time')
            
        # Get complexity - either from direct field or metrics
        complexity = msg.get('complexity_score')
        if complexity is None and 'metrics' in msg:
            complexity = msg['metrics'].get('complexity_score')
            
        if processing_time is not None and complexity is not None:
            processing_times.append(processing_time)
            complexities.append(complexity)
    
    if len(processing_times) < 2:
        return None, None
    
    try:
        correlation = np.corrcoef(processing_times, complexities)[0, 1]
        efficiency_ratios = [t/c for t, c in zip(processing_times, complexities) if c > 0]
        avg_efficiency_ratio = np.mean(efficiency_ratios) if efficiency_ratios else None
        return avg_efficiency_ratio, correlation
    except Exception as e:
        logger.error(f"Correlation calculation error: {str(e)}")
        return None, None

@app.get("/api/chat/{chat_id}/metrics")
async def get_chat_metrics(chat_id: str):
    try:
        chat = chat_storage.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        history = chat_storage.get_history(chat_id)
        
        # Initialize default metrics if not present
        message_metrics = []
        valid_metrics = []
        
        for msg in history:
            # Calculate metrics if not present
            if 'metrics' not in msg:
                msg['metrics'] = calculate_text_metrics(msg.get('content', ''))
                msg['complexity_score'] = msg['metrics']['complexity_score']
            
            # Ensure processing_time exists
            if 'processing_time' not in msg:
                msg['processing_time'] = None
            
            message_metrics.append({
                "id": msg.get('id'),
                "role": msg['role'],
                "content": msg['content'],
                "metrics": msg['metrics'],
                "processing_time": msg['processing_time'],
                "complexity_score": msg['complexity_score']
            })
            
            if msg['processing_time'] is not None and msg['complexity_score'] is not None:
                valid_metrics.append({
                    "processing_time": msg['processing_time'],
                    "complexity_score": msg['complexity_score']
                })
        
        # Calculate chat-wide metrics
        all_text = " ".join([msg['content'] for msg in history if msg['content']])
        chat_metrics = calculate_text_metrics(all_text)
        
        # Calculate efficiency metrics if we have enough data
        efficiency_ratio, correlation = None, None
        if len(valid_metrics) >= 2:
            try:
                processing_times = [m['processing_time'] for m in valid_metrics]
                complexities = [m['complexity_score'] for m in valid_metrics]
                
                correlation_matrix = np.corrcoef(processing_times, complexities)
                correlation = float(correlation_matrix[0, 1])
                
                efficiency_ratios = [t/c for t, c in zip(processing_times, complexities) if c > 0]
                efficiency_ratio = np.mean(efficiency_ratios) if efficiency_ratios else None
            except Exception as e:
                logger.error(f"Error calculating metrics: {str(e)}")
        
        return {
            "chat_id": chat_id,
            "chat_metrics": {
                'word_count': chat_metrics['word_count'],
                'sentence_count': chat_metrics['sentence_count'],
                'avg_word_length': chat_metrics['avg_word_length'],
                'avg_sentence_length': chat_metrics['avg_sentence_length'],
                'unique_word_count': chat_metrics['unique_word_count'],
                'lexical_diversity': chat_metrics['lexical_diversity'],
                'complexity_score': chat_metrics['complexity_score'],
                "efficiency_ratio": efficiency_ratio,
                "correlation": correlation
            },
            "message_metrics": message_metrics
        }
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load metrics")
    
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
    
    # Вычисляем метрики для чата
    metrics = {}
    try:
        metrics_response = await get_chat_metrics(chat_id)
        if "chat_metrics" in metrics_response:
            metrics = metrics_response["chat_metrics"]
    except Exception:
        metrics = {}  # В случае ошибки используем пустые метрики
    
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "chat_id": chat_id,
        "provider": chat["provider"],
        "models": models,
        "current_model": chat["model"],
        "history": history,
        "chat_name": chat_storage.get_chat_name(chat_id) or f"Chat {chat_id[:8]}",
        "metrics": metrics  # Передаем метрики в шаблон
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
    start_time = time.perf_counter()
    try:
        form_data = await request.form()
        file_types = form_data.getlist("file_types")
        
        file_infos = []
        full_content = message
        
        # Process files and calculate their metrics
        for file, file_type in zip(files, file_types):
            try:
                file_start_time = time.perf_counter()
                file_info = await process_uploaded_file(file, file_type)
                file_infos.append(file_info)
                
                if file_info["content"]:
                    # Calculate file metrics
                    metrics = calculate_text_metrics(file_info["content"])
                    complexity = metrics['complexity_score']
                    processing_time = time.perf_counter() - file_start_time
                    
                    benchmark.add_metrics(
                        file_type=file_type,
                        processing_time=processing_time,
                        complexity=complexity
                    )
                    
                    full_content += f"\n\n[File: {file_info['name']}]\n{file_info['content']}"
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue

        chat = chat_storage.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        if model != chat["model"]:
            chat_storage.update_chat_model(chat_id, model)

        # Create embedding for the full content
        embedding = llm_client.embedding_service.create_embedding(full_content)
        
        # Calculate metrics for the full message content
        text_metrics = calculate_text_metrics(full_content)
        user_processing_time = time.perf_counter() - start_time
        
        # Store user message with all metrics
        user_message = {
            "role": "user",
            "content": full_content,
            "files": file_infos,
            "metrics": text_metrics,
            "processing_time": user_processing_time,
            "complexity_score": text_metrics['complexity_score']
        }
        chat_storage.add_message(chat_id, user_message)
        chat_storage.save_embedding(chat_id, full_content, embedding)

        # Get chat history
        history = chat_storage.get_history(chat_id)
        
        # Generate response using the correct client method
        response_start = time.perf_counter()
        client = llm_client.get_client(chat["provider"])
        response = llm_client.generate_response(
            chat_id=chat_id,
            message=full_content,
            model=model,
            use_web_search=use_web_search,
            history=history,
            files=file_infos  # Добавьте эту строку
        )
        response_time = time.perf_counter() - response_start
        
        # Calculate metrics for AI response
        assistant_metrics = calculate_text_metrics(response)
        
        # Store assistant message with all metrics
        assistant_message = {
            "role": "assistant",
            "content": response,
            "metrics": assistant_metrics,
            "processing_time": response_time,
            "complexity_score": assistant_metrics['complexity_score']
        }
        chat_storage.add_message(chat_id, assistant_message)
        
        # Log the processing metrics
        logger.info(f"Processing time: {response_time:.4f}")
        logger.info(f"Complexity score: {assistant_metrics['complexity_score']:.2f}")
        
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
    use_web_search: bool = Form(False)):
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
            chat_id=chat_id,
            message=message,  # Добавить текущее сообщение
            model=model,
            use_web_search=use_web_search,
            history=history,
            files=file_info  # Добавить информацию о файлах
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