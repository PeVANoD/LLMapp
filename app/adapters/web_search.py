import requests
import logging
from typing import Optional, Dict, List
from app.core.interfaces import IWebSearch
from app.config import Config

logger = logging.getLogger(__name__)

from duckduckgo_search import DDGS
from typing import List, Dict
from datetime import datetime

class DuckDuckGoSearch(IWebSearch):
    def search(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = []
                # Добавляем текущую дату в запрос для актуальности
                dated_query = f"{query} {datetime.now().year}"
                for result in ddgs.text(dated_query, max_results=5):
                    # Добавляем дату получения результатов
                    result['retrieved_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    results.append(result)
                return self._format_results(results)
        except Exception as e:
            return f"Ошибка поиска: {str(e)}"

    def _format_results(self, results: List[Dict]) -> str:
        formatted = []
        for item in results:
            date_info = f"🔹 Данные получены: {item.get('retrieved_date', 'дата неизвестна')}"
            formatted.append(
                f"📌 {item.get('title', 'Без названия')}\n"
                f"{item.get('body', 'Нет описания')}\n"
                f"{date_info}\n"
                f"🌐 {item.get('href', '#')}"
            )
        return "\n\n".join(formatted) if formatted else "Ничего не найдено."

class GoogleSearch(IWebSearch):
    def __init__(self):
        self.api_key = Config.GOOGLE_API_KEY
        self.engine_id = Config.GOOGLE_ENGINE_ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str) -> str:
        """Выполняет поиск через Google Custom Search API"""
        try:
            params = {
                "key": self.api_key,
                "cx": self.engine_id,
                "q": query,
                "num": 5,  # Количество результатов
                "hl": "ru"  # Язык результатов
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._format_results(response.json())
            
        except Exception as e:
            logger.error(f"Google Search error: {str(e)}")
            return f"Ошибка поиска: {str(e)}"

    def _format_results(self, data: Dict) -> str:
        """Форматирует результаты для вывода"""
        items = data.get("items", [])
        if not items:
            return "По вашему запросу ничего не найдено."
        
        formatted = []
        for item in items:
            title = item.get("title", "Без названия")
            snippet = item.get("snippet", "Нет описания")
            link = item.get("link", "#")
            formatted.append(f"🔍 {title}\n{snippet}\n🌐 {link}")
        
        return "\n\n".join(formatted)