"""LLMProvider protocol — provider-agnostic LLM interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel


@dataclass
class LLMMessage:
    role: str  # system/user/assistant
    content: str


@dataclass
class LLMResult:
    content: str
    model: str
    provider: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost: float | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class LLMRequest:
    messages: list[LLMMessage]
    model: str
    temperature: float = 0.2
    max_tokens: int | None = None
    schema: type[BaseModel] | None = None  # for structured output
    extra: dict = field(default_factory=dict)


class LLMProvider(Protocol):
    provider_name: str

    async def complete(self, request: LLMRequest) -> LLMResult:
        """Generate completion; if schema provided, return validated JSON."""
        ...

    def complete_sync(self, request: LLMRequest) -> LLMResult:
        ...
