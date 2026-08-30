"""Source quality — tiers + dynamic reputation."""

from __future__ import annotations

from dataclasses import dataclass

# Domain → tier mapping
_TIER_MAP: dict[str, str] = {
    "arxiv.org": "A",
    "openalex.org": "A",
    "semanticscholar.org": "A",
    "crossref.org": "A",
    "openai.com": "A",
    "anthropic.com": "A",
    "deepmind.google": "A",
    "ai.google": "A",
    "research.google": "A",
    "nvidia.com": "A",
    "microsoft.com": "A",
    "huggingface.co": "A",
    "github.com": "C",
    "news.ycombinator.com": "C",
    "reddit.com": "C",
    "reuters.com": "B",
    "techcrunch.com": "B",
    "theverge.com": "B",
    "wired.com": "B",
}

_TIER_LABEL = {"A": "PRIMARY", "B": "SECONDARY", "C": "COMMUNITY", "D": "UNKNOWN"}


@dataclass
class SourceScore:
    tier: str  # A/B/C/D
    label: str  # PRIMARY etc
    score: float  # 0-1
    reason: str


def score_source(domain_or_source: str, historical_reliability: float | None = None) -> SourceScore:
    """Score a source by tier + optional historical reliability."""
    key = domain_or_source.lower().strip()
    # try bare domain extraction
    if "/" in key or "." in key:
        from urllib.parse import urlparse

        parsed = urlparse(key if "://" in key else f"https://{key}")
        host = parsed.netloc.lower().replace("www.", "")
        # try longest suffix match
        tier = "D"
        for dom, t in _TIER_MAP.items():
            if host == dom or host.endswith(f".{dom}"):
                tier = t
                break
        else:
            # also try source name like 'arxiv'
            tier = _TIER_MAP.get(key.split()[0], "D")
    else:
        tier = _TIER_MAP.get(key, "D")

    base = {"A": 0.95, "B": 0.75, "C": 0.45, "D": 0.25}[tier]
    if historical_reliability is not None:
        base = 0.7 * base + 0.3 * historical_reliability
    return SourceScore(tier=tier, label=_TIER_LABEL[tier], score=round(base, 3), reason=f"tier {tier} for {domain_or_source}")
