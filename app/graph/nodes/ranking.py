"""Ranking node — placeholder for Phase 3 composite ranker."""

from __future__ import annotations

from loguru import logger
# Phase 2: just pass through; Phase 3 will implement composite scorer

def ranking_node(state: dict) -> dict:
    docs = state.get("documents", [])
    logger.info(f"[ranking] placeholder for {len(docs)} docs")
    return {"logs": [{"level": "info", "message": f"[ranking] placeholder {len(docs)} docs"}]}
