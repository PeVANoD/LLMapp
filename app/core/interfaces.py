from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from PIL import Image
import numpy as np

class ILLMClient(ABC):
    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int] = None
    ) -> str:
        pass


class IEmbeddingService(ABC):
    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def search_similar(self, query: str, top_k: int) -> List[Dict]:
        pass

class IFileProcessor(ABC):
    @abstractmethod
    def process_file(self, file_path: str) -> str:
        pass
    
    @abstractmethod
    def extract_text_from_image(self, image: Image.Image) -> str:
        pass

class IWebSearch(ABC):
    @abstractmethod
    def search(self, query: str) -> str:
        pass

class IChatStorage(ABC):
    @abstractmethod
    def create_chat(self) -> str:
        pass
    
    @abstractmethod
    def add_message(self, chat_id: str, message: Dict):
        pass
    
    @abstractmethod
    def get_history(self, chat_id: str) -> List[Dict]:
        pass
    
    @abstractmethod
    def delete_chat(self, chat_id: str):
        pass
    
    @abstractmethod
    def save_to_disk(self):
        pass
    
    @abstractmethod
    def load_from_disk(self):
        pass

class IFileStorage(ABC):
    @abstractmethod
    def save_file(self, file: bytes, filename: str) -> str:
        pass
    
    @abstractmethod
    def get_file(self, filename: str) -> bytes:
        pass
    
    @abstractmethod
    def delete_file(self, filename: str):
        pass
