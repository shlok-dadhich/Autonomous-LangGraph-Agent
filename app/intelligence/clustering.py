"""Story clustering — content-level deduplication.

Replaces URL-only dedupe with event/story clusters.
Signals: lexical (title Jaccard), content_hash, title_hash, entity overlap, time window.
Content-level (not URL-level) per Feedly benchmark.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from app.domain.documents.identity import content_hash, title_hash


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class ClusterInput:
    id: str
    title: str
    text: str = ""
    source: str = ""
    published_at: str | None = None
    entities: list[str] = field(default_factory=list)


@dataclass
class StoryClusterResult:
    cluster_id: str
    cluster_confidence: float
    cluster_reason: str
    document_ids: list[str]
    title: str  # representative


def _title_sim(a: str, b: str) -> float:
    # combine jaccard + hash equality boost
    if title_hash(a) == title_hash(b):
        return 1.0
    return _jaccard(_tokens(a), _tokens(b))


def cluster_documents(
    docs: list[ClusterInput],
    title_threshold: float = 0.55,
    content_threshold: float = 0.85,
    entity_boost: float = 0.15,
) -> list[StoryClusterResult]:
    """Agglomerative clustering — cheap, deterministic, CPU-only.

    Steps:
    1. Block by first 3 title tokens to reduce O(n^2) (not yet; n is small)
    2. For each doc, compare to existing clusters via title + content_hash + entity overlap
    3. Emit cluster with confidence + reason
    """
    if not docs:
        return []

    # Precompute hashes
    c_hashes = {d.id: content_hash(d.title + " " + d.text) for d in docs}

    clusters: list[list[ClusterInput]] = []
    cluster_meta: list[dict] = []

    for doc in docs:
        best_idx = -1
        best_score = 0.0
        best_reason = ""
        for idx, members in enumerate(clusters):
            # compare to representative (first doc)
            rep = members[0]
            t_sim = _title_sim(doc.title, rep.title)
            c_sim = 1.0 if c_hashes[doc.id] == c_hashes[rep.id] else 0.0
            # entity overlap
            e_overlap = len(set(doc.entities) & set(rep.entities)) / max(1, len(set(doc.entities) | set(rep.entities))) if doc.entities or rep.entities else 0.0
            # combined
            score = max(t_sim, c_sim) + (entity_boost if e_overlap > 0 else 0)
            # time window not used yet (all docs within window)
            if score > best_score and (t_sim >= title_threshold or c_sim >= content_threshold):
                best_score = score
                best_idx = idx
                reasons = []
                if c_sim >= content_threshold:
                    reasons.append("content_hash")
                if t_sim >= title_threshold:
                    reasons.append(f"title_jaccard={t_sim:.2f}")
                if e_overlap > 0:
                    reasons.append(f"entity_overlap={e_overlap:.2f}")
                best_reason = "+".join(reasons) if reasons else f"score={score:.2f}"

        if best_idx >= 0:
            clusters[best_idx].append(doc)
            # update confidence as max of existing + new edge
            cluster_meta[best_idx]["confidence"] = max(cluster_meta[best_idx]["confidence"], min(0.99, best_score))
            if best_reason and best_reason not in cluster_meta[best_idx]["reason"]:
                cluster_meta[best_idx]["reason"] += f"|{best_reason}"
        else:
            clusters.append([doc])
            cluster_meta.append({"confidence": 0.95 if len(doc.title) > 10 else 0.70, "reason": "seed"})

    results: list[StoryClusterResult] = []
    for members, meta in zip(clusters, cluster_meta):
        # representative = longest title or first
        rep = max(members, key=lambda d: len(d.title))
        ids = [d.id for d in members]
        # cluster_id deterministic from sorted doc ids
        cid = str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(sorted(ids))))
        conf = meta["confidence"]
        # downgrade confidence if single-doc cluster with weak title
        if len(members) == 1:
            conf = min(conf, 0.80)
        else:
            conf = max(conf, 0.82)  # multi-doc cluster is higher confidence
        results.append(
            StoryClusterResult(
                cluster_id=cid,
                cluster_confidence=round(conf, 3),
                cluster_reason=meta["reason"],
                document_ids=ids,
                title=rep.title,
            )
        )
    return results


def quick_dedupe_by_hash(docs: list[ClusterInput]) -> list[ClusterInput]:
    """Fast pre-filter: drop exact content duplicates before embedding."""
    seen: dict[str, ClusterInput] = {}
    for d in docs:
        h = content_hash(d.title + " " + d.text)
        if h not in seen:
            seen[h] = d
    return list(seen.values())
