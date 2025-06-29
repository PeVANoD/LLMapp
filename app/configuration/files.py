# Максимальный размер файла (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Разрешенные типы файлов
ALLOWED_FILE_TYPES = {
    'file': [
        'text/plain',
        'application/pdf',
        'application/json',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/csv',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/rtf',
        'text/x-python',  # Добавьте другие текстовые типы при необходимости
    ],
    'image': [
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp',
    ]
}

# Дополнительно: разрешенные расширения файлов
ALLOWED_EXTENSIONS = {
    '.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', 
    '.csv', '.json', '.rtf', '.py', '.jpg', '.jpeg',
    '.png', '.gif', '.webp'
}