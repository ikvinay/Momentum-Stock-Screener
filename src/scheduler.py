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
from config import (
    REFRESH_HOUR_IST, REFRESH_MINUTE_IST,
    COMMODITY_REFRESH_HOUR_IST, COMMODITY_REFRESH_MINUTE_IST,
    FREEFLOAT_REFRESH_DAY, FREEFLOAT_REFRESH_HOUR_IST, FREEFLOAT_REFRESH_MINUTE_IST,
    IST_TIMEZONE,
)

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

        from src.pipeline import run_full_pipeline, run_commodity_pipeline, run_freefloat_refresh
        IST = pytz.timezone(IST_TIMEZONE)
        _scheduler = BackgroundScheduler(timezone=IST)

        # Stocks + Indices: daily at 16:00 IST
        _scheduler.add_job(
            func=lambda: run_full_pipeline("scheduler"),
            trigger=CronTrigger(hour=REFRESH_HOUR_IST, minute=REFRESH_MINUTE_IST, timezone=IST),
            id="daily_refresh",
            name=f"Stocks/Indices refresh at {REFRESH_HOUR_IST:02d}:{REFRESH_MINUTE_IST:02d} IST",
            replace_existing=True,
        )

        # Commodities: daily at 23:45 IST (after MCX closes at 23:30)
        _scheduler.add_job(
            func=lambda: run_commodity_pipeline("scheduler"),
            trigger=CronTrigger(
                hour=COMMODITY_REFRESH_HOUR_IST,
                minute=COMMODITY_REFRESH_MINUTE_IST,
                timezone=IST,
            ),
            id="commodity_refresh",
            name=f"Commodity refresh at {COMMODITY_REFRESH_HOUR_IST:02d}:{COMMODITY_REFRESH_MINUTE_IST:02d} IST",
            replace_existing=True,
        )

        # Free float: full refresh once a week on Saturday at 10:00 IST
        _scheduler.add_job(
            func=lambda: run_freefloat_refresh("scheduler"),
            trigger=CronTrigger(
                day_of_week=FREEFLOAT_REFRESH_DAY,
                hour=FREEFLOAT_REFRESH_HOUR_IST,
                minute=FREEFLOAT_REFRESH_MINUTE_IST,
                timezone=IST,
            ),
            id="weekly_freefloat_refresh",
            name=(
                f"Free float full refresh "
                f"({FREEFLOAT_REFRESH_DAY} {FREEFLOAT_REFRESH_HOUR_IST:02d}:{FREEFLOAT_REFRESH_MINUTE_IST:02d} IST)"
            ),
            replace_existing=True,
        )

        _scheduler.start()
        logger.info(
            "Scheduler started — stocks/indices at %02d:%02d IST, "
            "commodities at %02d:%02d IST, "
            "free float full refresh: %s at %02d:%02d IST",
            REFRESH_HOUR_IST, REFRESH_MINUTE_IST,
            COMMODITY_REFRESH_HOUR_IST, COMMODITY_REFRESH_MINUTE_IST,
            FREEFLOAT_REFRESH_DAY, FREEFLOAT_REFRESH_HOUR_IST, FREEFLOAT_REFRESH_MINUTE_IST,
        )


def stop_scheduler() -> None:
    """Gracefully stop the scheduler (called on app shutdown)."""
    global _scheduler
    with _lock:
        if _scheduler is not None and _scheduler.running:
            _scheduler.shutdown(wait=False)
            _scheduler = None
            logger.info("Scheduler stopped")
