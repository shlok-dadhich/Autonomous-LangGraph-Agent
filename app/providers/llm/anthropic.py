"""Anthropic LLM provider."""

from __future__ import annotations

import os

from app.providers.llm.base import LLMRequest, LLMResult


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model

    def complete_sync(self, request: LLMRequest) -> LLMResult:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        # Anthropic uses system separately
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        messages = [{"role": m.role, "content": m.content} for m in request.messages if m.role != "system"]
        resp = client.messages.create(model=request.model or self.model, max_tokens=request.max_tokens or 900, system=system or None, messages=messages)
        content = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return LLMResult(content=content, model=request.model or self.model, provider=self.provider_name, tokens_in=getattr(resp.usage, "input_tokens", None) if hasattr(resp, "usage") else None, tokens_out=getattr(resp.usage, "output_tokens", None) if hasattr(resp, "usage") else None)

    async def complete(self, request: LLMRequest) -> LLMResult:
        return self.complete_sync(request)
