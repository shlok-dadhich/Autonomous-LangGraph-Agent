"""Logging setup for the platform.

Wraps loguru with a single entry point; keeps the existing file rotation
while adding a structured prefix for future OTel integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: Path | str = "logs", level: str = "DEBUG") -> Path:
    """Configure console + file logging; returns log file path."""
    logger.remove()

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "worker.log"

    logger.add(
        sys.stderr,
        format=(
            "<level>{time:YYYY-MM-DD HH:mm:ss}</level> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day",
        retention="14 days",
    )
    logger.info(f"Logging configured. File: {log_file}")
    return log_file
