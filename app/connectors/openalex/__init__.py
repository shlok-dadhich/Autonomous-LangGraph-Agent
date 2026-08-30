"""OpenAlex adapter — works, citations."""

from __future__ import annotations

import requests

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery


@register("openalex")
class OpenAlexConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        q = " ".join(query.topics + query.keywords) or "artificial intelligence"
        try:
            resp = requests.get(
                "https://api.openalex.org/works",
                params={"search": q, "per-page": query.max_results},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            out: list[RawDocument] = []
            for w in data.get("results", [])[: query.max_results]:
                title = w.get("title") or w.get("display_name") or ""
                # author
                authors = ", ".join(a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])[:2])
                out.append(
                    RawDocument(
                        title=title,
                        url=w.get("id") or w.get("doi") or "",
                        description=w.get("abstract") or "",
                        source="openalex",
                        author=authors,
                        external_id=w.get("doi"),
                        metadata=w,
                    )
                )
            return out
        except Exception:
            return []
