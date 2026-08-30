"""GitHub adapter — stub for Phase 5 (releases/trending)."""

from __future__ import annotations

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery

@register("github")
class GithubConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []
