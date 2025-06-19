import os

class Config:
    # Настройки LM Studio
    LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    LM_STUDIO_MODEL = "local-model"  # Фиксированное имя для API
    
    # Параметры генерации
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2048
    
    # Настройки хранилища
    FILE_STORAGE_DIR = "file_storage"