from contextlib import contextmanager
from app.core.interfaces import IChatStorage
import uuid
import json
import os
from typing import Dict, List, Optional
import logging
import sqlite3
from datetime import datetime
import numpy as np

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
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
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
            conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
            )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_content TEXT NOT NULL,
                    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_chat ON embeddings(chat_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_text ON embeddings(text)")
            conn.commit()

    def create_chat(self, provider: str, model: str) -> str:
        chat_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chats (chat_id, provider, model) VALUES (?, ?, ?)",
                (chat_id, provider, model)
            )
            conn.commit()
        return chat_id
    
    def add_message_with_file(self, chat_id: str, message: Dict, file_info: Dict):
        """Adds message with attached file"""
        with sqlite3.connect(self.db_path) as conn:
            # Store message
            cursor = conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?) RETURNING id",
                (chat_id, message["role"], message["content"])
            )
            message_id = cursor.fetchone()[0]
            
            # Store file
            conn.execute(
                """INSERT INTO message_files 
                (message_id, file_name, file_type, file_content, image_data) 
                VALUES (?, ?, ?, ?, ?)""",
                (message_id, 
                file_info["name"],
                file_info["type"],
                file_info["content"],
                file_info.get("image_data"))
            )
            conn.commit()

    def rename_chat(self, chat_id: str, new_name: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO chat_names (chat_id, name) VALUES (?, ?)",
                (chat_id, new_name)
            )
            conn.commit()

    def delete_chat(self, chat_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM chat_names WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
            conn.commit()

    def add_message(self, chat_id: str, message: Dict):
        with sqlite3.connect(self.db_path) as conn:
            # Сохраняем основное сообщение
            conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, message["role"], message["content"])
            )
            message_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            
            # Сохраняем файлы, если они есть
            if "files" in message:
                for file_info in message["files"]:
                    conn.execute(
                        "INSERT INTO message_files (message_id, file_name, file_type, file_content) VALUES (?, ?, ?, ?)",
                        (message_id,
                        file_info["name"],
                        file_info["type"],
                        file_info["content"])
                    )
            conn.commit()
    def get_message_files(self, message_id: int) -> List[Dict]:
        """Gets files attached to message"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_name, file_type, file_content, image_data FROM message_files WHERE message_id = ?",
                (message_id,)
            )
            return [{
                "name": row[0],
                "type": row[1], 
                "content": row[2],
                "image_data": row[3]
            } for row in cursor.fetchall()]

    def get_history(self, chat_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            # Получаем основные сообщения
            cursor = conn.execute(
                "SELECT id, role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
                (chat_id,)
            )
            messages = []
            for row in cursor.fetchall():
                msg_id, role, content = row
                message = {"id": msg_id, "role": role, "content": content}
                
                # Получаем прикрепленные файлы для этого сообщения
                file_cursor = conn.execute(
                    "SELECT file_name, file_type, file_content FROM message_files WHERE message_id = ?",
                    (msg_id,)
                )
                files = []
                for file_row in file_cursor.fetchall():
                    files.append({
                        "name": file_row[0],
                        "type": file_row[1],
                        "content": file_row[2]
                    })
                
                if files:
                    message["files"] = files
                    
                messages.append(message)
                
            return messages
        
    def delete_chat(self, chat_id: str):
        """Удаляет чат и все связанные с ним данные (сообщения, имена, эмбеддинги)"""
        with sqlite3.connect(self.db_path) as conn:
            # Удаляем в правильном порядке из-за foreign key constraints
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM chat_names WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM embeddings WHERE chat_id = ?", (chat_id,))  # Новая строка
            conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
            conn.commit()

    def rename_chat(self, chat_id: str, new_name: str) -> bool:
        """Renames a chat. Returns True if successful."""
        if not new_name.strip():
            return False
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Проверяем существование чата
                cursor = conn.execute(
                    "SELECT 1 FROM chats WHERE chat_id = ?",
                    (chat_id,)
                )
                if not cursor.fetchone():
                    return False
                    
                conn.execute(
                    "INSERT OR REPLACE INTO chat_names (chat_id, name) VALUES (?, ?)",
                    (chat_id, new_name.strip())
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error renaming chat {chat_id}: {str(e)}")
            return False

    def get_chat_name(self, chat_id: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM chat_names WHERE chat_id = ?",
                (chat_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_chat_provider(self, chat_id: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT provider FROM chats WHERE chat_id = ?",
                (chat_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_chat_model(self, chat_id: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT model FROM chats WHERE chat_id = ?",
                (chat_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def update_chat_model(self, chat_id: str, model: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE chats SET model = ? WHERE chat_id = ?",
                (model, chat_id)
            )
            conn.commit()

    def save_to_disk(self):
        """No special action needed for SQLite as it's already disk-based"""
        pass

    def load_from_disk(self):
        """No special action needed for SQLite as it's already disk-based"""
        pass

    def get_chat(self, chat_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # This allows accessing columns by name
            cursor = conn.execute("""
                SELECT 
                    c.chat_id,
                    c.provider,
                    c.model,
                    datetime(c.created_at, 'localtime') as created_at,
                    cn.name
                FROM chats c
                LEFT JOIN chat_names cn ON c.chat_id = cn.chat_id
                WHERE c.chat_id = ?
            """, (chat_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "chat_id": row["chat_id"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "created_at": row["created_at"],
                    "name": row["name"]
                }
            return None
    def add_file_to_last_message(self, chat_id: str, file_info: Dict) -> bool:
        """Добавляет файл к последнему сообщению в чате"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Находим ID последнего сообщения
                cursor = conn.execute(
                    "SELECT id FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (chat_id,)
                )
                last_msg_id = cursor.fetchone()
                if not last_msg_id:
                    return False
                
                # Сохраняем файл
                conn.execute(
                    """INSERT INTO message_files 
                    (message_id, file_name, file_type, file_content) 
                    VALUES (?, ?, ?, ?)""",
                    (last_msg_id[0], file_info["name"], file_info["type"], file_info["content"])
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding file: {str(e)}")
            return False
        
    def get_all_chats(self, provider: Optional[str] = None) -> List[Dict]:
        """Get all chats, optionally filtered by provider"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                query = """
                    SELECT 
                        c.chat_id,
                        c.provider,
                        c.model,
                        datetime(c.created_at, 'localtime') as created_at,
                        cn.name,
                        (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.chat_id) as message_count
                    FROM chats c
                    LEFT JOIN chat_names cn ON c.chat_id = cn.chat_id
                    {where_clause}
                    ORDER BY c.created_at DESC
                """
                
                where_clause = "WHERE c.provider = ?" if provider else ""
                params = (provider,) if provider else ()
                
                cursor = conn.execute(query.format(where_clause=where_clause), params)
                return [
                    {
                        "chat_id": row["chat_id"],
                        "provider": row["provider"],
                        "model": row["model"],
                        "created_at": row["created_at"],
                        "message_count": row["message_count"],
                        "name": row["name"] or f"Чат {row['chat_id'][:8]}"
                    }
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            logger.error(f"Database error in get_all_chats: {str(e)}")
            return []
        
    def save_embedding(self, chat_id: str, text: str, embedding: List[float]) -> bool:
        try:
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO embeddings (chat_id, text, embedding) VALUES (?, ?, ?)",
                    (chat_id, text, embedding_bytes)
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving embedding: {str(e)}")
            return False

    def find_similar_texts(self, embedding: List[float], top_k: int = 3) -> List[Dict]:
        """Находит наиболее похожие тексты по вектору"""
        try:
            query_embed = np.array(embedding, dtype=np.float32)
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT id, text, embedding FROM embeddings")
                
                for row_id, text, embedding_bytes in cursor.fetchall():
                    try:
                        embed = np.frombuffer(embedding_bytes, dtype=np.float32)
                        similarity = float(np.dot(query_embed, embed))
                        results.append({"id": row_id, "text": text, "score": similarity})
                    except Exception as e:
                        logger.error(f"Error processing embedding: {str(e)}")
                        continue
                    
                # Сортируем по убыванию схожести
                results.sort(key=lambda x: x['score'], reverse=True)
                return results[:top_k]
                    
        except Exception as e:
            logger.error(f"Error finding similar texts: {str(e)}")
            return []
    def get_chat(self, chat_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM chats WHERE chat_id = ?",
                (chat_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None  
    def search_across_chats(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        try:
            query_embed = np.array(query_embedding, dtype=np.float32)
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT 
                        e.id,
                        e.text,
                        e.embedding,
                        e.chat_id,
                        COALESCE(cn.name, 'Chat ' || substr(e.chat_id, 1, 8)) as chat_name,
                        datetime(e.timestamp) as message_time
                    FROM embeddings e
                    LEFT JOIN chat_names cn ON e.chat_id = cn.chat_id
                    ORDER BY e.timestamp DESC
                    LIMIT 1000  -- Ограничиваем для производительности
                """)
                
                # Получаем все записи и вычисляем схожесть в Python
                rows = cursor.fetchall()
                for row in rows:
                    try:
                        embed = np.frombuffer(row['embedding'], dtype=np.float32)
                        similarity = float(np.dot(query_embed, embed))
                        
                        results.append({
                            'id': row['id'],
                            'text': row['text'],
                            'chat_id': row['chat_id'],
                            'chat_name': row['chat_name'],
                            'message_time': row['message_time'],
                            'score': similarity
                        })
                    except Exception as e:
                        logger.error(f"Error processing embedding row: {str(e)}")
                        continue
            
            # Сортируем по схожести и берем топ-N
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in search_across_chats: {str(e)}", exc_info=True)
            return []