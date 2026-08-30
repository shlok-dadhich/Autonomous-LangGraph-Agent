"""News/Tavily adapter."""

from __future__ import annotations

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery

@register("tavily")
@register("news")
class TavilyConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        from src.tools.tavily_client import fetch_tavily_results
        res = fetch_tavily_results(interest_profile={"topics": query.topics, "keywords": query.keywords}, max_results=query.max_results)
        arts = res.get("raw_articles", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
        return [RawDocument(title=a.get("title",""), url=a.get("url",""), description=a.get("description",""), source=a.get("source","tavily"), published_at=a.get("published_date"), metadata=a) for a in arts]
