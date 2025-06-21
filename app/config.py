class Config:
    # Настройки LM Studio/Ollama
    LM_STUDIO_URL = "http://localhost:1234/v1"  # Прямое указание значения
    OLLAMA_URL = "http://localhost:11434/api"   # Добавил для полноты
    LM_STUDIO_MODEL = "local-model"                # Фиксированное значение
    
    # Настройки веб-поиска
    
    GOOGLE_API_KEY = "AIzaSyAHPRjvGWv0pYzOAZ9bciEcKYwdpCTQhLU"  # Например: "AIzaSyBkEQY..."
    GOOGLE_ENGINE_ID = "33c2a3ffbb1174a60"  # Например: "0123456789abcdef"
    MAX_SEARCH_RESULTS = 3  # Количество результатов для LLM
    USE_DUCKDUCKGO = True
    WEB_SEARCH_ENABLED = True  # True/False вместо строки
    
    # Параметры генерации
    DEFAULT_MAX_TOKENS = 2048  # Число вместо строки
    DEFAULT_MODEL = "mistral:7b"  # Модель по умолчанию
    
    # Настройки хранилища
    FILE_STORAGE_DIR = "file_storage"
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Для Windows


