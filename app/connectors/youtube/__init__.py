"""YouTube — secondary evidence (talks/demos)."""

from __future__ import annotations
from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery
@register("youtube")
class YoutubeConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []  # needs YOUTUBE_API_KEY; captions have auth constraints
