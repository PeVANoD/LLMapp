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
    ],
    'image': [
        'image/jpeg',
        'image/png',
        'image/gif',
    ]
}