"""Persistent scheduler for autonomous newsletter execution."""

from __future__ import annotations

import os
from datetime import datetime
from functools import partial
from typing import Any, Callable

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger


class WorkerScheduler:
    """Run the newsletter pipeline on a fixed weekly schedule."""

    def __init__(
        self,
        job_func: Callable[..., None],
        housekeeping_func: Callable[[], None] | None = None,
        timezone: str | None = None,
        job_schedules: list[dict[str, Any]] | None = None,
    ) -> None:
        self.job_func = job_func
        self.housekeeping_func = housekeeping_func
        self.job_schedules = list(job_schedules or [])
        self.timezone = timezone or os.getenv("NEWSLETTER_TIMEZONE", "UTC")
        self.scheduler = BlockingScheduler(timezone=self.timezone)
        self.job_id = "newsletter_weekly_job"
        self.housekeeping_job_id = "checkpoint_housekeeping_job"
        self.triggers: list[tuple[str, str, CronTrigger]] = []
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
        self.triggers = []
        if self.job_schedules:
            for index, schedule in enumerate(self.job_schedules, start=1):
                schedule_id = str(schedule.get("id", f"newsletter_job_{index}")).strip() or f"newsletter_job_{index}"
                profile_path = str(schedule.get("profile_path", "config/profile.json")).strip()
                interval_days = schedule.get("interval_days")
                interval_hours = schedule.get("interval_hours")

                if interval_days is not None or interval_hours is not None:
                    days = int(interval_days or 0)
                    hours = int(interval_hours or 0)
                    if days <= 0 and hours <= 0:
                        raise ValueError(
                            f"Invalid interval config for schedule '{schedule_id}': interval_days/interval_hours must be > 0"
                        )
                    trigger = IntervalTrigger(days=days, hours=hours, timezone=self.timezone)
                    trigger_desc = f"interval days={days}, hours={hours}"
                else:
                    day = str(schedule.get("day", "*")).strip()
                    day_of_week = str(schedule.get("day_of_week", "*")).strip()
                    hour = int(schedule.get("hour", 8))
                    minute = int(schedule.get("minute", 0))
                    trigger = CronTrigger(
                        day=day,
                        day_of_week=day_of_week,
                        hour=hour,
                        minute=minute,
                        timezone=self.timezone,
                    )
                    trigger_desc = f"cron day={day}, day_of_week={day_of_week}, hour={hour}, minute={minute}"

                self.scheduler.add_job(
                    func=partial(self.job_func, profile_path=profile_path),
                    trigger=trigger,
                    id=schedule_id,
                    replace_existing=True,
                    max_instances=1,
                    coalesce=True,
                    misfire_grace_time=3600,
                )
                self.triggers.append((schedule_id, profile_path, trigger))
                logger.info(
                    "[scheduler] Registered configured job "
                    f"id={schedule_id}, profile={profile_path}, {trigger_desc} ({self.timezone})"
                )
        else:
            trigger = CronTrigger(day_of_week="mon", hour=8, minute=0, timezone=self.timezone)
            self.scheduler.add_job(
                func=self.job_func,
                trigger=trigger,
                id=self.job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=3600,
            )
            self.triggers.append((self.job_id, "config/profile.json", trigger))
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
        if self.triggers:
            now = datetime.now(self.scheduler.timezone)
            for job_id, profile_path, trigger in self.triggers:
                next_run = trigger.get_next_fire_time(previous_fire_time=None, now=now)
                if next_run:
                    logger.info(
                        f"[scheduler] Next Scheduled Run id={job_id}, "
                        f"profile={profile_path}: {next_run.isoformat()}"
                    )
                else:
                    logger.warning(f"[scheduler] Next Scheduled Run unavailable for id={job_id}.")

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
