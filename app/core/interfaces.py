from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union

class ILLMClient(ABC):
    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: Optional[int] = None
    ) -> str:
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        pass

class IEmbeddingService(ABC):
    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        """Создает векторное представление текста"""
        pass
    
    @abstractmethod
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Вычисляет косинусную схожесть между векторами"""
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
    
    @abstractmethod
    def save_embedding(self, chat_id: str, text: str, embedding: List[float]) -> bool:
        pass
    
    @abstractmethod
    def find_similar_texts(self, embedding: List[float], top_k: int = 3) -> List[Dict]:
        pass