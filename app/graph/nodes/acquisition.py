"""Acquisition nodes — thin wrappers over connectors."""

from __future__ import annotations

from loguru import logger


def acquisition_node(state: dict) -> dict:
    logger.info("[acquisition] placeholder — use connectors registry")
    return {"logs": [{"level": "info", "message": "[acquisition] placeholder"}]}
