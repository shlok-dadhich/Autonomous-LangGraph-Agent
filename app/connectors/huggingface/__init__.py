"""HuggingFace Daily Papers adapter."""

from __future__ import annotations

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery


@register("huggingface")
class HFConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        from src.tools.hf_client import fetch_hf_daily_papers
        res = fetch_hf_daily_papers(limit=query.max_results)
        arts = res.get("raw_articles", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
        return [RawDocument(title=a.get("title",""), url=a.get("url",""), description=a.get("description",""), source=a.get("source","huggingface-daily"), metadata=a) for a in arts]
