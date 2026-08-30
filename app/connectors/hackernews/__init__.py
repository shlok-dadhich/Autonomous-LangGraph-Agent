"""HackerNews adapter."""

from __future__ import annotations

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery


@register("hackernews")
class HackerNewsConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        from src.tools.hn_client import fetch_hn_stories
        res = fetch_hn_stories(interest_profile={"topics": query.topics, "keywords": query.keywords}, min_score=query.extra.get("min_score",50), max_items=query.max_results)
        arts = res.get("raw_articles", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
        return [RawDocument(title=a.get("title",""), url=a.get("url",""), description=a.get("description",""), source=a.get("source","hackernews"), published_at=a.get("published_date"), metadata=a) for a in arts]
