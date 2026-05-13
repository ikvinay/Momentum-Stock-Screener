"""
APScheduler singleton — starts the daily refresh job once for the process lifetime.

Importing this module multiple times (e.g. from different page files) is safe:
the module-level singleton ensures the scheduler is only created and started once.
"""

import logging
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import pytz
from config import REFRESH_HOUR_IST, REFRESH_MINUTE_IST, IST_TIMEZONE

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None
_lock = threading.Lock()


def start_scheduler() -> None:
    """
    Ensure the daily 4 PM IST refresh job is running.
    Safe to call multiple times — no-op if already started.
    """
    global _scheduler
    with _lock:
        if _scheduler is not None:
            return

        from src.pipeline import run_full_pipeline
        IST = pytz.timezone(IST_TIMEZONE)
        _scheduler = BackgroundScheduler(timezone=IST)
        _scheduler.add_job(
            func=lambda: run_full_pipeline("scheduler"),
            trigger=CronTrigger(
                hour=REFRESH_HOUR_IST,
                minute=REFRESH_MINUTE_IST,
                timezone=IST,
            ),
            id="daily_refresh",
            name=f"Daily refresh at {REFRESH_HOUR_IST:02d}:{REFRESH_MINUTE_IST:02d} IST",
            replace_existing=True,
        )
        _scheduler.start()
        logger.info(
            "Scheduler started — daily refresh at %02d:%02d IST",
            REFRESH_HOUR_IST,
            REFRESH_MINUTE_IST,
        )


def stop_scheduler() -> None:
    """Gracefully stop the scheduler (called on app shutdown)."""
    global _scheduler
    with _lock:
        if _scheduler is not None and _scheduler.running:
            _scheduler.shutdown(wait=False)
            _scheduler = None
            logger.info("Scheduler stopped")
