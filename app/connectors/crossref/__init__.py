"""Crossref adapter — DOI metadata."""

from __future__ import annotations

import requests

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery


@register("crossref")
class CrossrefConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        q = " ".join(query.topics + query.keywords) or "artificial intelligence"
        try:
            resp = requests.get(
                "https://api.crossref.org/works",
                params={"query": q, "rows": query.max_results},
                timeout=15,
            )
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
            out: list[RawDocument] = []
            for it in items[: query.max_results]:
                title = (it.get("title") or [""])[0]
                out.append(
                    RawDocument(
                        title=title,
                        url=it.get("URL") or it.get("DOI", ""),
                        description="; ".join(it.get("abstract", "")[:500] if isinstance(it.get("abstract"), str) else []) if it.get("abstract") else "",
                        source="crossref",
                        external_id=it.get("DOI"),
                        metadata=it,
                    )
                )
            return out
        except Exception:
            return []
