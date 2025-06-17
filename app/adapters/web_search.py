from app.core.interfaces import IWebSearch
import requests
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SerpAPISearch(IWebSearch):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('SERPAPI_KEY')
        if not self.api_key:
            logger.warning("SERPAPI_KEY not found in environment variables")
        
    def search(self, query: str) -> str:
        if not self.api_key:
            return "Web search is not configured. Please set SERPAPI_KEY environment variable."
        
        try:
            params = {
                'q': query,
                'api_key': self.api_key,
                'engine': 'google'
            }
            response = requests.get('https://serpapi.com/search', params=params, timeout=30)
            response.raise_for_status()
            
            results = response.json().get('organic_results', [])
            return "\n".join([res['snippet'] for res in results[:3]])
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return f"Web search error: {str(e)}"
