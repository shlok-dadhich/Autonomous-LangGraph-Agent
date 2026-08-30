"""Tavily search provider — wraps TavilyClient into SearchProvider."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from app.connectors.base import RawDocument
from app.core.config import get_settings
from app.providers.search.base import SearchQuery


class TavilySearchProvider:
    provider_name = "tavily"

    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or (settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else None) or os.getenv("TAVILY_API_KEY")

    def search(self, query: SearchQuery) -> list[RawDocument]:
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not configured")
        from tavily import TavilyClient

        client = TavilyClient(api_key=self.api_key)
        resp = client.search(query=query.query, max_results=query.max_results, search_depth="basic", include_answer=False)
        docs: list[RawDocument] = []
        for r in resp.get("results", []):
            docs.append(
                RawDocument(
                    title=r.get("title", "").strip(),
                    url=r.get("url", "").strip(),
                    description=(r.get("content", "") or r.get("snippet", ""))[:500].strip(),
                    source="tavily",
                    published_at=datetime.now(timezone.utc).isoformat(),
                    metadata={"raw": r},
                )
            )
        return docs
