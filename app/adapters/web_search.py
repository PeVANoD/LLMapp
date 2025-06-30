import requests
from typing import List, Dict
import logging
from app.config import Config
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

class WebSearchService:
    def search(self, query: str, force_duckduckgo: bool = False) -> List[Dict]:
        """Perform web search and return results with DuckDuckGo fallback"""
        try:
            # Всегда используем DuckDuckGo если принудительно включен
            if force_duckduckgo:
                return self._search_with_duckduckgo(query)
                
            # Иначе используем конфигурацию из настроек
            if Config.USE_DUCKDUCKGO:
                return self._search_with_duckduckgo(query)
            else:
                return self._search_with_google(query)
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []

    def _search_with_google(self, query: str) -> List[Dict]:
        """Search using Google Custom Search API"""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": Config.GOOGLE_API_KEY,
            "cx": Config.GOOGLE_ENGINE_ID,
            "q": query,
            "num": Config.MAX_SEARCH_RESULTS
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return [{
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", "")
        } for item in data.get("items", [])]

    def _search_with_duckduckgo(self, query: str) -> List[Dict]:
        """Search using DuckDuckGo with improved result formatting"""
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=Config.MAX_SEARCH_RESULTS)
            return [{
                "title": self._clean_text(r.get("title", "")),
                "link": r.get("href", ""),
                "snippet": self._clean_text(r.get("body", ""))
            } for r in results]
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better LLM consumption"""
        return (text
                .replace("\n", " ")
                .replace("\t", " ")
                .strip())