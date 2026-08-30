"""Crossref adapter — stub."""

from __future__ import annotations

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery

@register("crossref")
class CrossrefConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []
