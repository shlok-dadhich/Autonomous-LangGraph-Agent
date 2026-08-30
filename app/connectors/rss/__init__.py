"""RSS adapter."""

from __future__ import annotations

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery

@register("rss")
class RSSConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        from src.tools.rss_client import fetch_rss_sources
        feed_specs = query.extra.get("feed_specs")
        res = fetch_rss_sources(feed_specs=feed_specs)
        arts = res.get("raw_articles", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
        return [RawDocument(title=a.get("title",""), url=a.get("url",""), description=a.get("description",""), source=a.get("source","rss"), published_at=a.get("published_date"), metadata=a) for a in arts]
