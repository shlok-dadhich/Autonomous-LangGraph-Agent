"""Feature flags."""

from __future__ import annotations

FLAGS = {
    "new_ranker": False,
    "new_clusterer": False,
    "adaptive_delivery": False,
    "ask_intelligence": False,
    "trend_engine": False,
    "source_expansion": False,
    "new_email_template": False,
}

def is_enabled(flag: str) -> bool:
    return FLAGS.get(flag, False)
