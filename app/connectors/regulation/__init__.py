"""Regulation adapters — stub for Phase 6."""

from __future__ import annotations

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery

@register("regulation")
class RegulationConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []
