"""RerankerProvider protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RerankResult:
    index: int
    score: float
    document: str


class RerankerProvider(Protocol):
    provider_name: str

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        ...
