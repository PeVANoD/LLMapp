import requests
import logging
from typing import List, Dict
from PIL import Image
import io
import base64
from app.config import Config  # Импортируем конфиг

logger = logging.getLogger(__name__)

class LMStudioClient:
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or Config.LM_STUDIO_URL
        self.model = model or Config.LM_STUDIO_MODEL

    def generate_response(self, messages: List[Dict], model: str = None) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model or self.model,
                    "messages": messages,
                    "temperature": Config.DEFAULT_TEMPERATURE,
                    "max_tokens": Config.DEFAULT_MAX_TOKENS
                },
                timeout=300
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"LM Studio error: {str(e)}")
            raise