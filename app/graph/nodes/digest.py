"""Digest generation node — placeholder."""

from __future__ import annotations

from loguru import logger

def generate_digest_node(state: dict) -> dict:
    logger.info("[digest] placeholder")
    return {"logs": [{"level": "info", "message": "[digest] placeholder"}]}
