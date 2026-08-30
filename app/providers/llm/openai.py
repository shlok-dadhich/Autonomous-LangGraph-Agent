"""OpenAI LLM provider."""

from __future__ import annotations

import os

from app.core.config import get_settings
from app.providers.llm.base import LLMRequest, LLMResult


class OpenAIProvider:
    provider_name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        settings = get_settings()
        groq_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or groq_key
        self.model = model

    def complete_sync(self, request: LLMRequest) -> LLMResult:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        resp = client.chat.completions.create(model=request.model or self.model, messages=messages, temperature=request.temperature, max_tokens=request.max_tokens or 900)
        content = resp.choices[0].message.content or "{}"
        usage = getattr(resp, "usage", None)
        return LLMResult(content=content, model=request.model or self.model, provider=self.provider_name, tokens_in=getattr(usage, "prompt_tokens", None) if usage else None, tokens_out=getattr(usage, "completion_tokens", None) if usage else None)

    async def complete(self, request: LLMRequest) -> LLMResult:
        return self.complete_sync(request)
