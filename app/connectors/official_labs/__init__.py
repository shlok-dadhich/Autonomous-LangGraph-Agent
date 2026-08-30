"""Official AI Labs — RSS wrapper (OpenAI/Anthropic/Google/Meta/NVIDIA etc)."""

from __future__ import annotations
from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery
@register("official_labs")
class OfficialLabsConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        # delegates to rss fetcher via config/sources.yaml official_labs.feeds
        from src.tools.rss_client import fetch_rss_sources
        feeds = query.extra.get("feeds") or []
        if not feeds:
            return []
        res = fetch_rss_sources(feed_specs=feeds)
        arts = res.get("raw_articles", []) if isinstance(res, dict) else []
        return [RawDocument(title=a.get("title",""), url=a.get("url",""), description=a.get("description",""), source=a.get("source","official-labs")) for a in arts]
