"""Startups/Product Hunt."""

from __future__ import annotations
from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery
@register("startups")
@register("product_hunt")
class StartupsConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []  # needs PRODUCT_HUNT_TOKEN
