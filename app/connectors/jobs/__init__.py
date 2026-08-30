"""Jobs — Greenhouse etc."""

from __future__ import annotations
from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery
@register("jobs")
class JobsConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []
