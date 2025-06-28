# config.py
class Config:
    # LM Studio/Ollama settings
    LM_STUDIO_URL = "http://localhost:1234"
    OLLAMA_URL = "http://localhost:11434"
    
    # Generation parameters
    DEFAULT_MAX_TOKENS = 2048
    
    # Настройки веб-поиска
    GOOGLE_API_KEY = "AIzaSyAHPRjvGWv0pYzOAZ9bciEcKYwdpCTQhLU"
    GOOGLE_ENGINE_ID = "33c2a3ffbb1174a60"
    MAX_SEARCH_RESULTS = 3
    USE_DUCKDUCKGO = True
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Модель для эмбеддингов по умолчанию
    SIMILARITY_THRESHOLD = 0.7  # Порог схожести для использования контекста
    MAX_CONTEXT_LENGTH = 1000  # Максимальная длина контекста в токенах
    # Параметры генерации
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_MODEL = "mistral"  # Теперь используем mistral по умолчанию
    # Web search settings
    WEB_SEARCH_ENABLED = False
    
    @classmethod
    def update_config(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
