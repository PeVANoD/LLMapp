# app/adapters/llm_clients.py
import json
from typing import List, Dict, Optional
import requests
import logging
from app.adapters.web_search import WebSearchService
from app.core.services import EmbeddingService
from app.config import Config

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
    def __init__(self):
        self.clients = {
            'ollama': OllamaClient(),
            'lm_studio': LMStudioClient()
        }
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
                    context_messages: List[Dict] = None,
                    web_search_results: str = None,
                    files_context: str = None) -> List[Dict]:
        """Подготавливает полный набор сообщений для LLM с указанием языка"""
        messages = []
        
        # 1. Системное сообщение с явным указанием языка
        system_message = {
            "role": "system",
            "content": """Ты помощник-ассистент, будешь отвечать на том же языка,
            что и в вопросе. Чаще всего на русском. Текущий контекст диалога:"""
        }
        
        if context_messages:
            context_text = "\n".join([f"- {msg['text']}" for msg in context_messages])
            system_message["content"] += f"\n\nСхожий контекст из других чатов:\n{context_text}"
        
        messages.append(system_message)
        
        # 2. История чата
        for msg in history:
            content = msg["content"]
            
            if msg.get("files") and files_context:
                content += f"\n\n[Прикрепленные файлы]:\n{files_context}"
                
            messages.append({
                "role": msg["role"],
                "content": content
            })
        
        # 3. Веб-поиск
        if web_search_results and messages:
            messages[-1]["content"] += f"\n\n[Результаты поиска]:\n{web_search_results}"
        
        return messages
    
    def generate_response(self,
                     messages: List[Dict],
                     model: str,
                     context_messages: List[Dict] = None,
                     web_search_results: str = None,
                     files_context: str = None) -> str:
        """Генерирует ответ с учетом языка пользователя"""
        # Анализируем язык последнего сообщения пользователя
        last_user_message = next(
            (msg for msg in reversed(messages) if msg['role'] == 'user'),
            None
        )
        user_language = "ru" if last_user_message and self._is_russian(last_user_message['content']) else "en"
        
        prepared_messages = self.prepare_messages(
            history=messages,
            context_messages=context_messages,
            web_search_results=web_search_results,
            files_context=files_context
        )
        
        # Добавляем явное указание языка в последнее системное сообщение
        prepared_messages[0]['content'] += "\n\nТекущий язык общения: русский. Отвечай на русском."
        
        client = self._get_client_for_model(model)
        return client.generate_response(prepared_messages, model)

    def _is_russian(self, text: str) -> bool:
        """Проверяет, содержит ли текст русские буквы"""
        return any('а' <= char <= 'я' or 'А' <= char <= 'Я' for char in text)
    
    def _get_client_for_model(self, model: str):
        """Определяет, какой клиент использовать для модели"""
        for client in self.clients.values():
            if model in client.list_models():
                return client
        return None
    
    def get_relevant_context(self, query: str, chat_id: str, top_k: int = 3) -> List[Dict]:
        """Находит релевантные сообщения из других чатов"""
        try:
            embedding = self.embedding_service.create_embedding(query)
            similar_messages = chat_storage.search_across_chats(embedding)
            
            # Фильтруем сообщения из текущего чата
            return [msg for msg in similar_messages[:top_k] 
                   if msg.get('chat_id') != chat_id]
                    
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return []