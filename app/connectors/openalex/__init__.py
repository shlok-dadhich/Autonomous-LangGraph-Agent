"""OpenAlex adapter — stub for Phase 5."""

from __future__ import annotations

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery

@register("openalex")
class OpenAlexConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []  # TODO Phase 5
