import requests
import logging
from typing import List, Dict
from PIL import Image
import io
import base64
from app.config import Config  # Импортируем конфиг
from typing import Optional

logger = logging.getLogger(__name__)

class LMStudioClient:
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or Config.LM_STUDIO_URL
        self.model = model or Config.LM_STUDIO_MODEL

    def generate_response(self, messages: str | List[Dict], model: str, **kwargs) -> str:
        """Генерация ответа через LM Studio API"""
        try:
            # Форматируем сообщения в правильный формат
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": kwargs.get("max_tokens", 2000)
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            # Добавляем логирование для отладки
            logger.debug(f"LM Studio request: {payload}")
            logger.debug(f"LM Studio response: {response.text}")
            
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"LM Studio error: {str(e)}")
            return f"Ошибка генерации ответа: {str(e)}"