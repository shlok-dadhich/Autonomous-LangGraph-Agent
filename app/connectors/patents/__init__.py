"""Patents — USPTO."""

from __future__ import annotations
from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery
@register("patents")
class PatentsConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []
