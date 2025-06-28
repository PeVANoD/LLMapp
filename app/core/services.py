import uuid
from app.core.interfaces import IEmbeddingService, IChatStorage
import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import sqlite3
import logging

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
            CREATE TABLE IF NOT EXISTS message_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER,
                file_name TEXT,
                file_type TEXT,
                file_content TEXT,
                FOREIGN KEY(message_id) REFERENCES messages(id)
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
            # Store message text
            conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, message["role"], message["content"])
            )
            
            # If message has files, store them in a separate table
            if "files" in message:
                for file_info in message["files"]:
                    conn.execute(
                        "INSERT INTO message_files (message_id, file_name, file_type, file_content) VALUES (?, ?, ?, ?)",
                        (conn.execute("SELECT last_insert_rowid()").fetchone()[0],
                        file_info["name"],
                        file_info["type"],
                        file_info["content"])
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



class EmbeddingService:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.cache = {}  # Простое кэширование в памяти
        
    def create_embedding(self, text: str) -> List[float]:
        if text in self.cache:
            return self.cache[text]
            
        embedding = self.model.encode(text).tolist()
        self.cache[text] = embedding
        return embedding
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        try:
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0