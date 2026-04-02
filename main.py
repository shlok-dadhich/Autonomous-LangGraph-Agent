"""Persistent entrypoint for the autonomous newsletter worker."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from src.core.database import DatabaseManager, ensure_data_dir, run_monthly_checkpoint_housekeeping
from src.core.scheduler import WorkerScheduler
from src.services.telegram_bot import AlertService

_ACTIVE_SCHEDULER: WorkerScheduler | None = None


def setup_logging() -> None:
    """Configure console and file logging for worker runtime."""
    logger.remove()

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "worker.log"

    logger.add(
        sys.stderr,
        format=(
            "<level>{time:YYYY-MM-DD HH:mm:ss}</level> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level="DEBUG",
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


def load_profile(profile_path: str = "config/profile.json") -> dict:
    """Load user interest profile from JSON file."""
    profile_file = Path(profile_path)
    if not profile_file.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    with open(profile_file, "r", encoding="utf-8") as profile_handle:
        profile = json.load(profile_handle)

    logger.info(f"Loaded profile from {profile_path}")
    logger.debug(f"Profile topics: {profile.get('topics', [])}")
    logger.debug(f"Profile keywords: {profile.get('keywords', [])}")
    return profile


def run_research_phase(profile: dict, thread_id: str) -> dict:
    """Execute the full LangGraph pipeline once."""
    logger.info("=" * 80)
    logger.info("RUN: NEWSLETTER PIPELINE")
    logger.info("=" * 80)

    from src.graph.blueprint import research_graph

    initial_state = {
        "interest_profile": profile,
        "raw_articles": [],
        "unique_articles": [],
        "filtered_articles": [],
        "email_draft_content": [],
        "email_html_content": "",
        "sent_article_ids": [],
        "thread_id": thread_id,
        "logs": [],
        "error": None,
    }

    logger.info("Starting graph execution...")
    final_state = research_graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )
    logger.success("Graph execution completed successfully")
    return final_state


def summarize_run(final_state: dict, elapsed_seconds: float) -> bool:
    """Log a compact execution summary and return success status."""
    raw_articles = final_state.get("raw_articles", [])
    filtered_articles = final_state.get("filtered_articles", [])
    logs = final_state.get("logs", [])
    error = final_state.get("error")

    source_counts: dict[str, int] = {}
    for article in filtered_articles:
        source = article.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    logger.info("=" * 80)
    logger.info("RUN SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total raw articles harvested: {len(raw_articles)}")
    logger.info(f"Total filtered articles: {len(filtered_articles)}")
    logger.info(f"Email draft payload size: {len(final_state.get('email_draft_content', []))}")
    logger.info(f"Execution time: {elapsed_seconds:.2f}s")

    if source_counts:
        logger.info("Filtered article sources:")
        for source, count in sorted(source_counts.items()):
            logger.info(f"  - {source}: {count}")

    log_levels: dict[str, int] = {}
    for log_entry in logs:
        level = log_entry.get("level", "unknown")
        log_levels[level] = log_levels.get(level, 0) + 1

    if log_levels:
        logger.info("Graph log level counts:")
        for level, count in sorted(log_levels.items()):
            logger.info(f"  - {level}: {count}")

    if error:
        logger.error(f"Pipeline reported state error: {error}")
        return False

    if not filtered_articles:
        logger.warning("Pipeline completed but produced no filtered articles.")
        return False

    logger.success("Run completed with valid output payload.")
    return True


def run_pipeline_once() -> None:
    """Run one full newsletter cycle, raising on fatal failures."""
    alert_service = AlertService()
    thread_id = f"ai-weekly-{datetime.now().strftime('%Y-%U')}"
    final_error_alert_sent = False

    start_time = time.perf_counter()
    try:
        profile = load_profile()
        final_state = run_research_phase(profile, thread_id=thread_id)
        success = summarize_run(final_state, time.perf_counter() - start_time)
        final_error = final_state.get("error")

        if final_error:
            alert_service.send_error(
                error_message=str(final_error),
                thread_id=thread_id,
            )
            final_error_alert_sent = True

        if not success:
            state_error = final_error or "Pipeline run completed with invalid output"
            raise RuntimeError(state_error)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        if not final_error_alert_sent:
            alert_service.send_error(
                error_message=f"{type(exc).__name__}: {str(exc)}",
                thread_id=thread_id,
                traceback_text=traceback_text,
            )
        raise


def run_monthly_housekeeping() -> None:
    """Monthly housekeeping job for checkpoint retention."""
    deleted_rows = run_monthly_checkpoint_housekeeping(days_to_keep=30)
    logger.info(
        f"[housekeeping] Checkpoint cleanup finished. deleted_rows={deleted_rows}"
    )


def _check_required_env() -> None:
    """Backward-compatible alias for verify_secrets."""
    verify_secrets()


def _check_data_volume() -> None:
    data_dir = ensure_data_dir()
    probe_file = data_dir / ".write_probe"

    with open(probe_file, "w", encoding="utf-8") as probe_handle:
        probe_handle.write(datetime.now().isoformat())
    probe_file.unlink(missing_ok=True)

    usage = os.statvfs(str(data_dir)) if hasattr(os, "statvfs") else None
    if usage:
        free_bytes = usage.f_bavail * usage.f_frsize
        logger.info(f"[system-check] Data volume writable. Free bytes: {free_bytes}")
    else:
        logger.info("[system-check] Data volume writable.")


def _check_database_connections() -> None:
    db_manager = DatabaseManager()
    history_conn = sqlite3.connect(str(db_manager.db_path))
    history_conn.execute("SELECT 1")
    history_conn.close()

    checkpoint_path = ensure_data_dir() / "checkpoints.db"
    checkpoint_conn = sqlite3.connect(str(checkpoint_path))
    checkpoint_conn.execute("PRAGMA journal_mode=WAL;")
    checkpoint_conn.close()

    logger.info("[system-check] SQLite history and checkpoint DB connections are healthy.")


def verify_secrets() -> None:
    """Validate that all required secrets are present before startup."""
    required_keys = ["TAVILY_API_KEY", "GROQ_API_KEY", "SMTP_APP_PASS", "TELEGRAM_BOT_TOKEN"]
    missing = [key for key in required_keys if not os.getenv(key)]

    if missing:
        logger.critical("Missing required secrets: {}", ", ".join(missing))
        raise EnvironmentError("Missing required environment variables: " + ", ".join(missing))

    logger.info("[system-check] Required secrets are present.")


def system_check() -> None:
    """Perform startup checks before entering scheduler loop."""
    logger.info("=" * 80)
    logger.info("SYSTEM CHECK")
    logger.info("=" * 80)

    load_dotenv(override=False)
    verify_secrets()
    _check_data_volume()
    _check_database_connections()

    logger.success("System check passed. Worker is ready.")


def run_system_check() -> None:
    """Backward-compatible alias for startup validation."""
    system_check()


def _shutdown_handler(signum: int, _frame: Any) -> None:
    global _ACTIVE_SCHEDULER

    signal_name = signal.Signals(signum).name if signum in [s.value for s in signal.Signals] else str(signum)
    logger.warning(f"Received {signal_name}. Initiating graceful shutdown...")

    if _ACTIVE_SCHEDULER is not None:
        _ACTIVE_SCHEDULER.shutdown(wait=True)


def register_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _shutdown_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown_handler)


def main() -> int:
    """Start persistent newsletter worker."""
    global _ACTIVE_SCHEDULER

    try:
        setup_logging()
        system_check()

        register_signal_handlers()
        timezone = os.getenv("NEWSLETTER_TIMEZONE", "UTC")
        _ACTIVE_SCHEDULER = WorkerScheduler(
            job_func=run_pipeline_once,
            housekeeping_func=run_monthly_housekeeping,
            timezone=timezone,
        )

        logger.info("Starting persistent scheduler loop...")
        _ACTIVE_SCHEDULER.start()
        return 0

    except (EnvironmentError, FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error(f"Startup check failed: {type(exc).__name__}: {str(exc)}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Worker interrupted by keyboard signal.")
        return 0
    except Exception as exc:
        logger.error(f"Fatal worker failure: {type(exc).__name__}: {str(exc)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
