"""Funding — Crunchbase etc."""

from __future__ import annotations
from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery
@register("funding")
class FundingConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []
