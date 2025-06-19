from app.core.interfaces import IChatStorage, IFileStorage
import uuid
import json
import os
from typing import Dict, List
import logging
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

class SQLiteChatStorage(IChatStorage):
    def __init__(self, db_path: str = "chats.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_names (
                    chat_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
                )
            """)
            conn.commit()

    def create_chat(self) -> str:
        chat_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO chats (chat_id) VALUES (?)", (chat_id,))
            conn.commit()
        return chat_id

    def add_message(self, chat_id: str, message: Dict):
        with sqlite3.connect(self.db_path) as conn:
            # Check if chat exists
            cursor = conn.execute("SELECT 1 FROM chats WHERE chat_id = ?", (chat_id,))
            if not cursor.fetchone():
                raise ValueError(f"Chat {chat_id} not found")
            
            conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, message['role'], message['content'])
            )
            conn.commit()

    def get_history(self, chat_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
                (chat_id,)
            )
            return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]

    def delete_chat(self, chat_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
            conn.commit()

    def save_to_disk(self):
        pass  # SQLite already saves to disk

    def load_from_disk(self):
        pass  # SQLite loads automatically

    def get_all_chats(self) -> List[Dict]:
        """Получение всех чатов с именами"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    c.chat_id, 
                    datetime(c.created_at, 'localtime') as created_at,
                    COUNT(m.id) as message_count,
                    cn.name
                FROM chats c
                LEFT JOIN messages m ON c.chat_id = m.chat_id
                LEFT JOIN chat_names cn ON c.chat_id = cn.chat_id
                GROUP BY c.chat_id
                ORDER BY c.created_at DESC
            """)
            return [
                {
                    "chat_id": row[0],
                    "created_at": row[1],
                    "message_count": row[2],
                    "name": row[3]  # Может быть None, если имя не задано
                }
                for row in cursor.fetchall()
            ]
        
    def rename_chat(self, chat_id: str, new_name: str) -> None:
        """Переименование чата"""
        if not new_name.strip():
            raise ValueError("Имя чата не может быть пустым")
        
        with sqlite3.connect(self.db_path) as conn:
            # Проверяем существование чата
            cursor = conn.execute("SELECT 1 FROM chats WHERE chat_id = ?", (chat_id,))
            if not cursor.fetchone():
                raise ValueError(f"Чат {chat_id} не найден")
            
            # Обновляем или добавляем имя
            conn.execute("""
                INSERT OR REPLACE INTO chat_names (chat_id, name)
                VALUES (?, ?)
            """, (chat_id, new_name.strip()))
            conn.commit()

    def get_chat_name(self, chat_id: str) -> str:
    #"""Получаем имя чата (возвращает None если имя не задано)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name FROM chat_names WHERE chat_id = ?", (chat_id,))
            result = cursor.fetchone()
            return result[0] if result else None


class FileStorage(IFileStorage):
    def __init__(self, storage_dir: str = "file_storage"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save_file(self, file: bytes, filename: str) -> str:
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(file)
        return filepath

    def get_file(self, filename: str) -> bytes:
        filepath = os.path.join(self.storage_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filename} not found")
        with open(filepath, 'rb') as f:
            return f.read()

    def delete_file(self, filename: str):
        filepath = os.path.join(self.storage_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)



