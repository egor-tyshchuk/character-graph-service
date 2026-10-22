from collections.abc import AsyncIterator
from typing import Any

import pytest_asyncio
from arq import Worker, create_pool, func
from arq.connections import ArqRedis, RedisSettings
from arq.worker import Function

from app.queue import PROCESS_ARTICLE_JOB, ArqJobQueue
from app.worker import WorkerSettings


@pytest_asyncio.fixture
async def pool(redis_url: str) -> AsyncIterator[ArqRedis]:
    arq_pool = await create_pool(RedisSettings.from_dsn(redis_url))
    await arq_pool.flushdb()
    try:
        yield arq_pool
    finally:
        await arq_pool.flushdb()
        await arq_pool.aclose()


def registered_keep_result() -> float | None:
    [job] = [
        f
        for f in WorkerSettings.functions
        if isinstance(f, Function) and f.name == PROCESS_ARTICLE_JOB
    ]
    return job.keep_result_s


async def run_burst(pool: ArqRedis, processed: list[str]) -> None:
    async def stub_job(ctx: dict[str, Any], article_id: str) -> None:
        processed.append(article_id)

    job = func(stub_job, name=PROCESS_ARTICLE_JOB, keep_result=registered_keep_result())
    worker = Worker(
        functions=[job], redis_pool=pool, burst=True, handle_signals=False, poll_delay=0.05
    )
    await worker.main()


async def test_requeue_while_queued_adds_no_second_job_and_works_after_it_finished(
    pool: ArqRedis,
) -> None:
    queue = ArqJobQueue(pool)
    processed: list[str] = []

    await queue.enqueue_article("article-1")
    await queue.enqueue_article("article-1")
    await run_burst(pool, processed)
    await queue.enqueue_article("article-1")
    await run_burst(pool, processed)

    assert processed == ["article-1", "article-1"]
