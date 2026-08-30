"""Lobste.rs community."""

from __future__ import annotations
from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery
@register("lobsters")
class LobstersConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []
