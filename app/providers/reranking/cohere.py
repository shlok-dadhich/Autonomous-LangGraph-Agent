"""Cohere reranker provider — Stage-2 reranking (optional)."""

from __future__ import annotations

import os

from app.providers.reranking.base import RerankResult


class CohereReranker:
    provider_name = "cohere"

    def __init__(self, api_key: str | None = None, model: str = "rerank-english-v3.0"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[RerankResult]:
        if not self.api_key:
            # Fallback: lexical scoring when no key (keeps pipeline working)
            return self._lexical_fallback(query, documents, top_k)
        try:
            import cohere  # type: ignore

            client = cohere.Client(self.api_key)
            resp = client.rerank(model=self.model, query=query, documents=documents, top_n=top_k or len(documents))
            return [RerankResult(index=r.index, score=float(r.relevance_score), document=documents[r.index]) for r in resp.results]
        except Exception:
            return self._lexical_fallback(query, documents, top_k)

    def _lexical_fallback(self, query: str, documents: list[str], top_k: int | None) -> list[RerankResult]:
        # Simple Jaccard over tokens
        import re

        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for idx, doc in enumerate(documents):
            d_tokens = set(re.findall(r"\w+", doc.lower()))
            score = len(q_tokens & d_tokens) / max(1, len(q_tokens | d_tokens))
            scored.append(RerankResult(index=idx, score=score, document=doc))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: top_k or len(scored)]
