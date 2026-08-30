"""Semantic Scholar adapter — stub."""

from __future__ import annotations

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery

@register("semantic_scholar")
class SemanticScholarConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []
