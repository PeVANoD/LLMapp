from app.core.interfaces import ILLMClient
from typing import List, Dict
import requests
from PIL import Image
import base64
import io
import logging

logger = logging.getLogger(__name__)

class OllamaClient(ILLMClient):
    def __init__(self, base_url: str = "http://localhost:11434/api"):
        self.base_url = base_url
        self.session = requests.Session()

    def generate_response(self, messages: List[Dict], model: str) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}/chat",
                json={"model": model, "messages": messages},
                timeout=60
            )
            response.raise_for_status()
            return response.json()['message']['content']
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def generate_response_with_image(self, messages: List[Dict], image: Image.Image, model: str) -> str:
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Add image to messages
            visual_messages = [{
                "role": "user",
                "content": "image",
                "images": [img_str]
            }] + messages

            response = self.session.post(
                f"{self.base_url}/chat",
                json={"model": model, "messages": visual_messages},
                timeout=90
            )
            response.raise_for_status()
            return response.json()['message']['content']
        except Exception as e:
            logger.error(f"Error generating response with image: {str(e)}")
            raise

class LMStudioClient(ILLMClient):
    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url
        self.session = requests.Session()

    def generate_response(self, messages: List[Dict], model: str) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json={"messages": messages, "temperature": 0.7},
                timeout=60
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def generate_response_with_image(self, messages: List[Dict], image: Image.Image, model: str) -> str:
        raise NotImplementedError("LM Studio doesn't support multimodal models yet")