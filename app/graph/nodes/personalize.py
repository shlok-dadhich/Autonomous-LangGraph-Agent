"""Personalization node — placeholder Phase 4."""

from __future__ import annotations

from loguru import logger

def personalize_node(state: dict) -> dict:
    logger.info("[personalize] placeholder")
    return {"logs": [{"level": "info", "message": "[personalize] placeholder"}]}
