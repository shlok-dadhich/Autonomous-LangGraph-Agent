"""ArXiv adapter — wraps src/tools/arxiv_client."""

from __future__ import annotations

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery

@register("arxiv")
class ArxivConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        from src.tools.arxiv_client import fetch_arxiv_papers
        cats = query.extra.get("categories", ["cs.AI", "cs.LG"])
        days = query.extra.get("days_back", 7)
        max_results = query.max_results
        res = fetch_arxiv_papers(categories=cats, days_back=days, max_results=max_results)
        # src client returns wrapped dict via safe_execute
        arts = res.get("raw_articles", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
        return [RawDocument(title=a.get("title",""), url=a.get("url",""), description=a.get("description",""), source=a.get("source","arxiv"), published_at=a.get("published_date"), metadata=a) for a in arts]
