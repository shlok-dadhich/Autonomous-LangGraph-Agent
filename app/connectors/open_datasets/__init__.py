"""Open datasets / Common Crawl."""

from __future__ import annotations
from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery
@register("open_datasets")
@register("common_crawl")
class OpenDatasetsConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []
