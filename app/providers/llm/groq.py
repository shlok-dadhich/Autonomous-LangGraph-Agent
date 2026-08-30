"""Groq LLM provider — implements LLMProvider via Groq SDK."""

from __future__ import annotations

import json
import os
import time

from loguru import logger

from app.core.config import get_settings
from app.providers.llm.base import LLMMessage, LLMRequest, LLMResult


class GroqProvider:
    provider_name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        settings = get_settings()
        key = api_key or (settings.groq_api_key.get_secret_value() if settings.groq_api_key else None) or os.getenv("GROQ_API_KEY")
        self.api_key = key
        self.model = model or settings.groq_model
        self.max_retries = 2

    def _is_retryable(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status in {429, 500, 502, 503, 504}:
            return True
        msg = f"{type(exc).__name__} {str(exc)}".lower()
        return any(tok in msg for tok in ("rate", "timeout", "temporarily", "overloaded", "unavailable"))

    def complete_sync(self, request: LLMRequest) -> LLMResult:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not configured")
        from groq import Groq

        client = Groq(api_key=self.api_key)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        model = request.extra.get("model") or request.model or self.model
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens or 900,
                )
                content = resp.choices[0].message.content or "{}"
                # If schema requested, validate JSON
                if request.schema:
                    try:
                        data = json.loads(content)
                        # If single object expected but got list, keep as is
                        request.schema.model_validate(data) if isinstance(data, dict) else None
                    except Exception as e:
                        logger.warning(f"[groq] schema validation failed: {e}")
                usage = getattr(resp, "usage", None)
                return LLMResult(
                    content=content,
                    model=model,
                    provider=self.provider_name,
                    tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
                    tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
                    raw={"id": getattr(resp, "id", None)},
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    raise
                delay = min(2**attempt, 8)
                logger.warning(f"[groq] attempt {attempt+1} failed {type(exc).__name__}: {exc}, retry in {delay}s")
                time.sleep(delay)
        if last_exc:
            raise last_exc
        raise RuntimeError("Groq failed without exception")

    async def complete(self, request: LLMRequest) -> LLMResult:
        # No native async; delegate to sync
        return self.complete_sync(request)

    # Compatibility shim for legacy writer
    def invoke(self, messages: list[dict], max_tokens: int = 900) -> str:
        req = LLMRequest(messages=[LLMMessage(**m) for m in messages], model=self.model, max_tokens=max_tokens)
        return self.complete_sync(req).content
