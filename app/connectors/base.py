"""SourceConnector protocol + shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SourceQuery:
    topics: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    max_results: int = 10
    extra: dict = field(default_factory=dict)

@dataclass
class RawDocument:
    title: str
    url: str
    description: str = ""
    source: str = "unknown"
    published_at: str | None = None
    author: str | None = None
    external_id: str | None = None
    metadata: dict = field(default_factory=dict)

class SourceConnector(Protocol):
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        ...
