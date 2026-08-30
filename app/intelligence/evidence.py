"""Evidence & source agreement — confidence from independent sources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvidenceBundle:
    claim_text: str
    sources: list[str]
    source_tiers: list[str]  # A/B/C/D
    confidence: str  # High/Medium/Low
    source_count: int
    has_primary: bool


TIER_WEIGHT = {"A": 1.0, "B": 0.7, "C": 0.4, "D": 0.2}


def evaluate_evidence(sources: list[str], tiers: list[str]) -> EvidenceBundle:
    """Compute confidence from source agreement."""
    count = len(sources)
    has_primary = "A" in tiers
    # weighted score
    score = sum(TIER_WEIGHT.get(t, 0.3) for t in tiers) / max(1, len(tiers))
    # diversity bonus
    unique_tiers = len(set(tiers))
    score += 0.1 if unique_tiers >= 2 else 0
    # count bonus
    if count >= 4 and has_primary:
        conf = "High"
    elif count >= 2 and score >= 0.6:
        conf = "Medium"
    elif count >= 1 and has_primary:
        conf = "Medium"
    else:
        conf = "Low"
    return EvidenceBundle(
        claim_text="",
        sources=sources,
        source_tiers=tiers,
        confidence=conf,
        source_count=count,
        has_primary=has_primary,
    )


def detect_contradiction(claims: list[str]) -> dict | None:
    """Cheap contradiction: same numeric metric with different values."""
    import re

    nums = []
    for c in claims:
        m = re.search(r"(\d+)%", c)
        if m:
            nums.append(int(m.group(1)))
    if len(nums) >= 2 and max(nums) - min(nums) >= 15:
        return {
            "type": "metric_conflict",
            "values": nums,
            "explanation": f"Claims report {min(nums)}% vs {max(nums)}% — metrics may not be directly comparable (different evaluation conditions).",
        }
    return None
