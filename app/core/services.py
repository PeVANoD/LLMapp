import uuid
from app.core.interfaces import IEmbeddingService, IChatStorage, IFileStorage
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
class EmbeddingService(IEmbeddingService):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect("embeddings.db") as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT UNIQUE,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def create_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        embedding_bytes = embedding.tobytes()
        
        with sqlite3.connect("embeddings.db") as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (text, embedding) VALUES (?, ?)",
                (text, embedding_bytes)
            )
            conn.commit()
        
        return embedding.tolist()

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embed = self.model.encode(query)
        
        with sqlite3.connect("embeddings.db") as conn:
            cursor = conn.execute("SELECT text, embedding FROM embeddings")
            results = []
            
            for text, embedding_bytes in cursor.fetchall():
                try:
                    embed = np.frombuffer(embedding_bytes, dtype=np.float32)
                    similarity = float(np.dot(query_embed, embed))
                    results.append({"text": text, "score": similarity})
                except Exception as e:
                    logger.error(f"Error processing embedding: {str(e)}")
                    continue
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]