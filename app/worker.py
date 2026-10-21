import os
from typing import Any

import structlog
from arq import cron, func
from arq.connections import RedisSettings
from arq.cron import CronJob

from app.config import HOURS_PER_DAY, MINUTES_PER_HOUR, ScheduleSettings, Settings
from app.container import Container, build_container, close_container
from app.logging import configure_logging
from app.pipeline.process import PipelineDeps, process_article
from app.pipeline.reconcile import reconcile_characters
from app.pipeline.sweep import sweep_stuck_articles
from app.queue import RECONCILE_JOB, RECONCILE_JOB_ID


async def startup(ctx: dict[str, Any]) -> None:
    settings = Settings()
    configure_logging(settings.log_level, settings.log_json)
    container = await build_container(settings)
    ctx["container"] = container
    ctx["deps"] = container.pipeline_deps


async def shutdown(ctx: dict[str, Any]) -> None:
    container: Container | None = ctx.get("container")
    if container is not None:
        await close_container(container)


async def process_article_job(ctx: dict[str, Any], article_id: str) -> None:
    deps: PipelineDeps = ctx["deps"]
    with structlog.contextvars.bound_contextvars(job_id=ctx.get("job_id"), article_id=article_id):
        await process_article(article_id, deps)


async def reconcile_job(ctx: dict[str, Any]) -> dict[str, Any]:
    deps: PipelineDeps = ctx["deps"]
    with structlog.contextvars.bound_contextvars(job_id=ctx.get("job_id")):
        report = await reconcile_characters(deps)
    return report.model_dump()


async def sweep_stuck_articles_job(ctx: dict[str, Any]) -> int:
    deps: PipelineDeps = ctx["deps"]
    with structlog.contextvars.bound_contextvars(job_id=ctx.get("job_id")):
        return await sweep_stuck_articles(deps)


def cron_schedule(interval_minutes: int) -> tuple[set[int], set[int] | None]:
    if interval_minutes < MINUTES_PER_HOUR:
        return set(range(0, MINUTES_PER_HOUR, interval_minutes)), None
    hours = interval_minutes // MINUTES_PER_HOUR
    return {0}, None if hours == 1 else set(range(0, HOURS_PER_DAY, hours))


def build_cron_jobs(settings: ScheduleSettings) -> list[CronJob]:
    jobs = []
    if settings.stuck_sweep_enabled:
        minute, hour = cron_schedule(settings.stuck_sweep_interval_minutes)
        jobs.append(cron(sweep_stuck_articles_job, minute=minute, hour=hour, unique=True))
    if settings.reconcile_enabled:
        minute, hour = cron_schedule(settings.reconcile_interval_minutes)
        jobs.append(
            cron(
                reconcile_job,
                minute=minute,
                hour=hour,
                run_at_startup=False,
                unique=True,
                job_id=RECONCILE_JOB_ID,
            )
        )
    return jobs


class WorkerSettings:
    functions = (process_article_job, func(reconcile_job, name=RECONCILE_JOB, keep_result=0))
    cron_jobs = build_cron_jobs(ScheduleSettings())
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 4
    job_timeout = 3600
    max_tries = 1
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://redis:6379/0"))
