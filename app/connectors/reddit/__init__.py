"""Reddit/social signals adapter via Tavily."""

from __future__ import annotations

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery

@register("reddit")
@register("social_signals")
class RedditConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        from src.tools.social_signal_client import fetch_social_signals
        res = fetch_social_signals(interest_profile={"topics": query.topics, "keywords": query.keywords}, max_results=query.max_results)
        arts = res.get("raw_articles", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
        return [RawDocument(title=a.get("title",""), url=a.get("url",""), description=a.get("description",""), source=a.get("source","social_signals"), metadata=a) for a in arts]
