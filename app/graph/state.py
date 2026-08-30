"""Typed graph state — extends legacy GraphState with intelligence fields."""

from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict
import operator


class GraphState(TypedDict, total=False):
    # legacy
    interest_profile: dict
    profile: dict
    mode: str
    trusted_domains: List[str]
    raw_articles: Annotated[list, operator.add]
    unique_articles: list
    filtered_articles: list
    email_draft_content: list
    email_html_content: Optional[str]
    sent_article_ids: List[str]
    thread_id: Optional[str]
    logs: Annotated[list, operator.add]
    error: Optional[str]

    # intelligence (Phase 2+)
    documents: list  # normalized Document dicts with canonical_url, content_hash, etc.
    clusters: list  # StoryClusterResult dicts
    events: list  # EventResult dicts
    entities: list  # EntityMention dicts
    claims: list  # Claim dicts
    evidence: list  # EvidenceBundle dicts
    source_scores: list  # SourceScore dicts
    metrics: dict  # per-stage metrics
