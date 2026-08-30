"""MiniLM embedding provider — Stage-1 retriever (keeps existing ranker)."""

from __future__ import annotations

import gc
from functools import lru_cache


class MiniLMProvider:
    provider_name = "local"
    model_name = "all-MiniLM-L6-v2"
    dimension = 384

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.model_name
        self._model = None

    @lru_cache(maxsize=1)
    def _load(self):
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(self.model_name, device="cpu")
        return m.to("cpu")

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        try:
            import torch

            with torch.no_grad():
                embs = model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
                return [list(map(float, e)) for e in embs]
        finally:
            gc.collect()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
