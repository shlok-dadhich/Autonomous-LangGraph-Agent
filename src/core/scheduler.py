"""Persistent scheduler for autonomous newsletter execution."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Callable

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger


class WorkerScheduler:
    """Run the newsletter pipeline on a fixed weekly schedule."""

    def __init__(
        self,
        job_func: Callable[[], None],
        housekeeping_func: Callable[[], None] | None = None,
        timezone: str | None = None,
    ) -> None:
        self.job_func = job_func
        self.housekeeping_func = housekeeping_func
        self.timezone = timezone or os.getenv("NEWSLETTER_TIMEZONE", "UTC")
        self.scheduler = BlockingScheduler(timezone=self.timezone)
        self.job_id = "newsletter_weekly_job"
        self.housekeeping_job_id = "checkpoint_housekeeping_job"
        self.trigger: CronTrigger | None = None
        self.housekeeping_trigger: CronTrigger | None = None

    def _on_job_event(self, event: JobExecutionEvent) -> None:
        if event.exception:
            logger.error(
                f"[scheduler] Job execution failed for '{event.job_id}': {str(event.exception)}"
            )
        else:
            logger.success(f"[scheduler] Job execution finished successfully for '{event.job_id}'")
        self.log_next_run_time()

    def _register_job(self) -> None:
        trigger = CronTrigger(day_of_week="mon", hour=8, minute=0, timezone=self.timezone)
        self.trigger = trigger

        self.scheduler.add_job(
            func=self.job_func,
            trigger=trigger,
            id=self.job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
        )

        logger.info(
            "[scheduler] Registered weekly cron job: every Monday at 08:00 "
            f"({self.timezone})"
        )

        if self.housekeeping_func is not None:
            housekeeping_trigger = CronTrigger(day=1, hour=0, minute=0, timezone=self.timezone)
            self.housekeeping_trigger = housekeeping_trigger
            self.scheduler.add_job(
                func=self.housekeeping_func,
                trigger=housekeeping_trigger,
                id=self.housekeeping_job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=3600,
            )
            logger.info(
                "[scheduler] Registered monthly housekeeping job: day 1 at 00:00 "
                f"({self.timezone})"
            )

    def log_next_run_time(self) -> None:
        next_run = None
        if self.trigger is not None:
            now = datetime.now(self.scheduler.timezone)
            next_run = self.trigger.get_next_fire_time(previous_fire_time=None, now=now)

        if next_run:
            logger.info(f"[scheduler] Next Scheduled Run (weekly): {next_run.isoformat()}")
        else:
            logger.warning("[scheduler] Next Scheduled Run (weekly) unavailable.")

        if self.housekeeping_trigger is not None:
            now = datetime.now(self.scheduler.timezone)
            housekeeping_next_run = self.housekeeping_trigger.get_next_fire_time(
                previous_fire_time=None,
                now=now,
            )
            if housekeeping_next_run:
                logger.info(
                    f"[scheduler] Next Scheduled Run (housekeeping): "
                    f"{housekeeping_next_run.isoformat()}"
                )

    def start(self) -> None:
        self._register_job()
        self.scheduler.add_listener(self._on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self.log_next_run_time()
        logger.info("[scheduler] BlockingScheduler started. Waiting for next run...")
        self.scheduler.start()

    def shutdown(self, wait: bool = False) -> None:
        if self.scheduler.running:
            logger.info("[scheduler] Shutdown requested. Stopping scheduler...")
            self.scheduler.shutdown(wait=wait)
            logger.info("[scheduler] Scheduler stopped cleanly.")


NewsletterScheduler = WorkerScheduler


def current_utc_timestamp() -> str:
    """Return current UTC timestamp for diagnostics."""
    return datetime.utcnow().isoformat() + "Z"
