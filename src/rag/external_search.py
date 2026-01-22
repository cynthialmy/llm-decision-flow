"""External search clients with allowlist controls."""
from __future__ import annotations

from typing import List, Dict, Any
from urllib.parse import urlparse
import httpx

from src.config import settings


class ExternalSearchClient:
    """Serper + Wikipedia search with domain allowlist."""

    def __init__(self):
        self.serper_key = settings.serper_api_key
        self.allowlist = self._parse_allowlist(settings.external_search_allowlist)

    @staticmethod
    def _parse_allowlist(raw: str) -> List[str]:
        if not raw:
            return []
        return [entry.strip().lower() for entry in raw.split(",") if entry.strip()]

    def _allowed_domain(self, url: str) -> bool:
        if not self.allowlist:
            return False
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        return any(host.endswith(domain) for domain in self.allowlist)

    def serper_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.serper_key:
            raise ValueError("SERPER_API_KEY is required for external search.")

        headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": max_results}
        with httpx.Client(timeout=settings.frontier_timeout_s) as client:
            response = client.post("https://google.serper.dev/search", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        items = []
        for result in data.get("organic", [])[:max_results]:
            url = result.get("link", "")
            if not self._allowed_domain(url):
                continue
            items.append({
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "url": url,
                "source": urlparse(url).netloc,
            })
        return items

    def wikipedia_search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        }
        try:
            with httpx.Client(timeout=settings.frontier_timeout_s) as client:
                response = client.get("https://en.wikipedia.org/w/api.php", params=params)
                response.raise_for_status()
                data = response.json()
        except Exception:
            return []

        items = []
        for result in data.get("query", {}).get("search", [])[:max_results]:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            items.append({
                "title": title,
                "snippet": snippet,
                "url": url,
                "source": "wikipedia.org",
            })
        return items

    def search(self, query: str) -> List[Dict[str, Any]]:
        results = []
        if settings.allow_external_search:
            if self.serper_key:
                results.extend(self.serper_search(query))
            results.extend(self.wikipedia_search(query))
        return results
