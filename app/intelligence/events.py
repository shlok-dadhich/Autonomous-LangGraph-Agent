"""Event extraction — 18 types, cheap classifier first."""

from __future__ import annotations

from dataclasses import dataclass

EVENT_TYPES = [
    "MODEL_RELEASE",
    "PRODUCT_RELEASE",
    "PAPER_RELEASE",
    "DATASET_RELEASE",
    "GITHUB_RELEASE",
    "FUNDING",
    "ACQUISITION",
    "PARTNERSHIP",
    "BENCHMARK_RESULT",
    "RESEARCH_RESULT",
    "SECURITY_INCIDENT",
    "VULNERABILITY",
    "REGULATION",
    "POLICY_CHANGE",
    "COMPANY_ANNOUNCEMENT",
    "OPEN_SOURCE_RELEASE",
    "CONFERENCE",
    "JOB_SIGNAL",
]

# Keyword rules for cheap classifier (lowercase)
_KEYWORDS: dict[str, list[str]] = {
    "MODEL_RELEASE": ["model release", "introducing", "announcing", "gpt-", "llama", "claude", "gemini", "mistral", "released model"],
    "PAPER_RELEASE": ["arxiv", "paper", "preprint", "published paper"],
    "DATASET_RELEASE": ["dataset release", "new dataset", "open dataset"],
    "GITHUB_RELEASE": ["github release", "release tag", "v1.", "v2.", "changelog"],
    "FUNDING": ["raises", "funding", "series a", "series b", "investment"],
    "ACQUISITION": ["acquires", "acquisition", "acquired"],
    "PARTNERSHIP": ["partnership", "collaboration", "joint venture"],
    "BENCHMARK_RESULT": ["benchmark", "leaderboard", "sota", "state of the art", "accuracy"],
    "RESEARCH_RESULT": ["research", "study", "finding", "experiment"],
    "SECURITY_INCIDENT": ["breach", "incident", "hacked", "exploit"],
    "VULNERABILITY": ["vulnerability", "cve-", "security flaw"],
    "REGULATION": ["regulation", "eu ai act", "nist", "compliance", "law"],
    "POLICY_CHANGE": ["policy", "terms of service", "guideline"],
    "COMPANY_ANNOUNCEMENT": ["announces", "announcement", "launching"],
    "OPEN_SOURCE_RELEASE": ["open source", "open-source", "apache 2.0", "mit license", "github.com"],
    "CONFERENCE": ["conference", "neurips", "icml", "iclr", "summit"],
    "JOB_SIGNAL": ["hiring", "job opening", "careers", "hiring engineers"],
    "PRODUCT_RELEASE": ["product release", "new product", "available now", "launches"],
}


@dataclass
class EventResult:
    event_type: str
    confidence: float
    entities: list[str]
    evidence: str


def classify_event(title: str, description: str = "") -> EventResult:
    """Rule-based cheap classifier. Returns top event_type."""
    text = f"{title} {description}".lower()
    best_type = "COMPANY_ANNOUNCEMENT"
    best_score = 0.0
    for etype, kws in _KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in text)
        score = hits / max(1, len(kws))  # normalize
        # boost if title contains keyword
        if any(kw in title.lower() for kw in kws):
            score += 0.2
        if score > best_score:
            best_score = score
            best_type = etype
    # confidence is best_score + small prior
    conf = min(0.99, max(0.35, best_score + 0.3))
    if best_score == 0:
        conf = 0.40  # default low confidence
    return EventResult(event_type=best_type, confidence=round(conf, 3), entities=[], evidence=title[:200])


def extract_events(title: str, description: str = "", source: str = "") -> list[EventResult]:
    """Extract 1-2 events per document."""
    primary = classify_event(title, description)
    results = [primary]
    # If strong secondary signal (e.g., both MODEL_RELEASE and OPEN_SOURCE), add second
    text = f"{title} {description}".lower()
    for etype, kws in _KEYWORDS.items():
        if etype == primary.event_type:
            continue
        if any(kw in text for kw in kws[:2]) and len(results) < 2:
            # weak second event
            results.append(EventResult(event_type=etype, confidence=0.45, entities=[], evidence=title[:120]))
            break
    return results
