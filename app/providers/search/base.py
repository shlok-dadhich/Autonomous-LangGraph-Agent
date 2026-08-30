"""SearchProvider protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.connectors.base import RawDocument


@dataclass
class SearchQuery:
    query: str
    max_results: int = 10
    extra: dict = field(default_factory=dict)


class SearchProvider(Protocol):
    provider_name: str

    def search(self, query: SearchQuery) -> list[RawDocument]:
        ...
