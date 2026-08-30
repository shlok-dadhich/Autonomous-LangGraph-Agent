"""Delivery node — idempotent placeholder."""

from __future__ import annotations

from loguru import logger


def delivery_node(state: dict) -> dict:
    logger.info("[delivery] placeholder")
    return {"logs": [{"level": "info", "message": "[delivery] placeholder"}]}
