"""Observability — per-stage metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StageMetrics:
    stage: str
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    counts: dict = field(default_factory=dict)
    success: bool = True

    def finish(self, **counts):
        self.ended_at = time.time()
        self.counts.update(counts)

    @property
    def latency_ms(self) -> float:
        end = self.ended_at or time.time()
        return (end - self.started_at) * 1000


# Standard metric keys per plan §38
# source_success_rate, source_latency, articles_fetched, documents_normalized,
# clusters_created, duplicates_removed, stories_ranked, stories_selected,
# citation_coverage, ranking_confidence, etc.
