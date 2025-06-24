# app/adapters/llm_clients.py
import json
from typing import List, Dict, Optional
import requests
import logging
from app.adapters.web_search import WebSearchService
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
                timeout=60
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
                timeout=60
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
        self.web_search_service = WebSearchService()  # Renamed from web_search to web_search_service
    def generate_response(self, messages: List[Dict], model: str) -> str:
        """Generate response with proper context handling"""
        client = self._get_client_for_model(model)
        if not client:
            raise ValueError("No suitable client found for model")
        
        return client.generate_response(messages, model)
    def get_client(self, provider: str):
        return self.clients.get(provider)
    
    def perform_web_search(self, query: str) -> List[Dict]:  # Renamed method
        return self.web_search_service.search(query)