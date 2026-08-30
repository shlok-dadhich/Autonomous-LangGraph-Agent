"""Entity extraction + normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Alias map for normalization (lower -> canonical)
_ALIASES: dict[str, str] = {
    "openai": "OpenAI",
    "open ai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "google deepmind": "Google DeepMind",
    "deepmind": "Google DeepMind",
    "meta": "Meta",
    "nvidia": "NVIDIA",
    "microsoft": "Microsoft",
    "mistral": "Mistral",
    "cohere": "Cohere",
    "xai": "xAI",
    "langgraph": "LangGraph",
    "langchain": "LangChain",
    "hugging face": "Hugging Face",
    "huggingface": "Hugging Face",
    "github": "GitHub",
    "gpt-4": "GPT-4",
    "gpt-5": "GPT-5",
    "claude": "Claude",
    "llama": "Llama",
}

_KINDS: dict[str, str] = {
    "OpenAI": "Company",
    "Anthropic": "Company",
    "Google": "Company",
    "Google DeepMind": "Company",
    "Meta": "Company",
    "NVIDIA": "Company",
    "Microsoft": "Company",
    "Mistral": "Company",
    "Cohere": "Company",
    "xAI": "Company",
    "Hugging Face": "Company",
    "GitHub": "Company",
    "LangGraph": "Framework",
    "LangChain": "Framework",
    "GPT-4": "Model",
    "GPT-5": "Model",
    "Claude": "Model",
    "Llama": "Model",
}


@dataclass
class EntityMention:
    canonical_name: str
    kind: str
    aliases: list[str]
    span: str


def extract_entities(title: str, description: str = "") -> list[EntityMention]:
    text = f"{title} {description}".lower()
    found: dict[str, EntityMention] = {}
    for alias, canon in _ALIASES.items():
        if alias in text:
            if canon not in found:
                found[canon] = EntityMention(
                    canonical_name=canon,
                    kind=_KINDS.get(canon, "Company"),
                    aliases=[alias],
                    span=alias,
                )
            else:
                if alias not in found[canon].aliases:
                    found[canon].aliases.append(alias)
    # Also capture GitHub repos via regex
    for m in re.finditer(r"github\.com/([\w.-]+/[\w.-]+)", text):
        repo = m.group(1)
        name = f"github.com/{repo}"
        if name not in found:
            found[name] = EntityMention(canonical_name=name, kind="Repository", aliases=[name], span=name)
    # Model names with numbers
    for m in re.finditer(r"\b(gpt-?\d[\w.-]*|claude[\w.-]*|llama[\w.-]*)\b", text):
        name = m.group(1)
        canon = name.upper() if name.lower().startswith("gpt") else name
        if canon not in found:
            found[canon] = EntityMention(canonical_name=canon, kind="Model", aliases=[name], span=name)
    return list(found.values())


def normalize_entity(name: str) -> str:
    return _ALIASES.get(name.lower().strip(), name.strip())
