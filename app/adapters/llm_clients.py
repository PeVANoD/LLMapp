# app/adapters/llm_clients.py
import json
from typing import List, Dict, Optional
import requests
import logging
from app.adapters.web_search import WebSearchService
from app.core.services import EmbeddingService
from app.config import Config
from app.infrastructure.storage import SQLiteChatStorage

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self):
        self.base_url = Config.OLLAMA_URL

    def list_models(self) -> List[str]:
        """List all available Ollama models"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}")
            return []

    def generate_response(self, messages: List[Dict], model: str, **kwargs) -> str:
        try:
            request_data = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                }
            }
            
            logger.debug(f"Sending to Ollama: {request_data}")
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=120
            )
            
            logger.debug(f"Ollama raw response: {response.text}")
            
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("message", {}).get("content", "No response generated")
            
        except Exception as e:
            logger.error(f"Ollama error: {str(e)}")
            raise
        
class LMStudioClient:
    def __init__(self):
        self.base_url = Config.LM_STUDIO_URL

    def generate_response(self, messages: List[Dict], model: str, **kwargs) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", Config.DEFAULT_MAX_TOKENS),
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LM Studio error: {str(e)}")
            raise

    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and 'data' in data:
                return [model["id"] for model in data["data"]]
            return []
        except Exception as e:
            logger.error(f"Error listing LM Studio models: {str(e)}")
            return []

class MultiLLMClient:
    def __init__(self, chat_storage: SQLiteChatStorage):
        self.clients = {
            'ollama': OllamaClient(),
            'lm_studio': LMStudioClient()
        }
        
        self.chat_storage = chat_storage
        self.web_search_service = WebSearchService()
        self.embedding_service = EmbeddingService()
    
    def get_client(self, provider: str):
        return self.clients.get(provider)
    
    def perform_web_search(self, query: str) -> List[Dict]:
        return self.web_search_service.search(query)
    
    def format_context(self, similar_messages: List[Dict]) -> str:
        """Форматирует контекст для LLM"""
        if not similar_messages:
            return ""
            
        context = "Relevant information from other chats:\n"
        for i, msg in enumerate(similar_messages, 1):
            context += f"{i}. [From chat '{msg['chat_name']}']: {msg['text']}\n"
        return context
    
    def prepare_messages(self, 
                         history: List[Dict],
                         files_context: str = "",
                         use_web_search: bool = False) -> List[Dict]:
        messages = []
        
        # Системное сообщение
        system_message = {
            "role": "system",
            "content": "Ты помощник-ассистент. Отвечай на том же языке, что и вопрос."
        }
        messages.append(system_message)
        
        # История чата с файлами
        for msg in history:
            content = msg["content"]
            messages.append({
                "role": msg["role"],
                "content": content
            })
        
        # Веб-поиск (если активирован)
        if use_web_search:
            try:
                # Берем последний запрос пользователя
                last_user_query = next(
                    (m["content"] for m in reversed(history) if m["role"] == "user"), 
                    ""
                )
                if last_user_query:
                    search_results = self.perform_web_search(last_user_query)
                    search_text = "\n".join(
                        f"- {res['title']}: {res['snippet']}" 
                        for res in search_results
                    )
                    messages[-1]["content"] += f"\n\n[Результаты поиска]:\n{search_text}"
            except Exception as e:
                logger.error(f"Web search error: {str(e)}")
        
        return messages
    
    def generate_response(
        self,
        chat_id: str,
        model: str,
        use_web_search: bool = False
    ) -> str:
        # 1. Получаем историю чата
        history = self.chat_storage.get_history(chat_id)
        
        # 3. Подготавливаем сообщения для LLM
        messages = []
        
        # Системное сообщение
        messages.append({
            "role": "system",
            "content": "Ты помощник-ассистент. Отвечай на том же языке, что и вопрос."
        })
        
        # История чата
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # 4. Получаем информацию о чате (провайдер)
        chat = self.chat_storage.get_chat(chat_id)
        if not chat:
            raise ValueError(f"Chat {chat_id} not found")
        provider = chat["provider"]
        
        # 5. Выбираем клиент по провайдеру
        client = self.get_client(provider)
        if not client:
            raise ValueError(f"Provider {provider} not supported")
        
        # 6. Генерируем ответ
        return client.generate_response(
            messages=messages,
            model=model
        )
