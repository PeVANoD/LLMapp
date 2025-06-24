from contextlib import contextmanager
from app.core.interfaces import IChatStorage
import uuid
import json
import os
from typing import Dict, List, Optional
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
            conn.execute("DROP TABLE IF EXISTS messages")
            conn.execute("DROP TABLE IF EXISTS chat_names")
            conn.execute("DROP TABLE IF EXISTS chats")
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
            conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, message["role"], message["content"])
            )
            conn.commit()

    def get_history(self, chat_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp",
                (chat_id,)
            )
            return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]

    def delete_chat(self, chat_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM chat_names WHERE chat_id = ?", (chat_id,))
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