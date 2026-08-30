"""Semantic Scholar adapter."""

from __future__ import annotations

import requests

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery


@register("semantic_scholar")
class SemanticScholarConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        q = " ".join(query.topics + query.keywords) or "artificial intelligence"
        try:
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": q, "limit": query.max_results, "fields": "title,url,abstract,authors"},
                timeout=15,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            out: list[RawDocument] = []
            for p in data.get("data", [])[: query.max_results]:
                out.append(
                    RawDocument(
                        title=p.get("title", ""),
                        url=p.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId')}",
                        description=p.get("abstract", "") or "",
                        source="semantic_scholar",
                        author=", ".join(a.get("name", "") for a in p.get("authors", [])[:2]),
                        external_id=p.get("paperId"),
                        metadata=p,
                    )
                )
            return out
        except Exception:
            return []
