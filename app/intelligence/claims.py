"""Claim extraction — sentences as claims with evidence refs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Claim:
    text: str
    claim_type: str = "fact"  # fact/quote/metric
    confidence: float = 0.5
    evidence_refs: list[dict] = field(default_factory=list)  # [{document_id, span, url}]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_claims(title: str, description: str, document_id: str, url: str) -> list[Claim]:
    """Naive claim extractor: title + each description sentence is a claim."""
    claims: list[Claim] = []
    title = title.strip()
    if title:
        claims.append(
            Claim(
                text=title,
                claim_type="fact",
                confidence=0.70,
                evidence_refs=[{"document_id": document_id, "span": title[:120], "url": url}],
            )
        )
    # split description into sentences, take up to 3 strongest
    sentences = [s.strip() for s in _SENT_SPLIT.split(description) if s.strip()]
    for sent in sentences[:3]:
        if len(sent) < 20:
            continue
        # detect metric
        is_metric = bool(re.search(r"\d+%", sent) or "benchmark" in sent.lower())
        claims.append(
            Claim(
                text=sent[:280],
                claim_type="metric" if is_metric else "fact",
                confidence=0.60 if is_metric else 0.50,
                evidence_refs=[{"document_id": document_id, "span": sent[:120], "url": url}],
            )
        )
    return claims
