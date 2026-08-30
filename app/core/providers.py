"""Model Gateway registry — maps task -> provider."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelGateway:
    llm_fast: str = "groq:llama-3.1-8b"
    llm_reasoning: str = "groq:llama-3.3-70b-versatile"
    llm_cheap: str = "groq:llama-3.1-8b"
    embedding: str = "local:all-MiniLM-L6-v2"
    reranker: str | None = None
    search: str = "tavily"
    email: str = "smtp"

gateway = ModelGateway()
