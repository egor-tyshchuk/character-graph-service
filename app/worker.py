import os
from typing import Any

import structlog
from arq.connections import RedisSettings

from app.config import Settings
from app.container import Container, build_container, close_container
from app.logging import configure_logging
from app.pipeline.process import PipelineDeps, process_article


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


class WorkerSettings:
    functions = (process_article_job,)
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 4
    job_timeout = 3600
    max_tries = 1
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://redis:6379/0"))
