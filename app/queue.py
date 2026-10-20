from typing import Protocol

from arq.connections import ArqRedis

from app.pipeline.process import PipelineDeps, process_article

PROCESS_ARTICLE_JOB = "process_article_job"


class JobQueue(Protocol):
    async def enqueue_article(self, article_id: str) -> None: ...


class ArqJobQueue:
    def __init__(self, pool: ArqRedis) -> None:
        self._pool = pool

    async def enqueue_article(self, article_id: str) -> None:
        await self._pool.enqueue_job(PROCESS_ARTICLE_JOB, article_id)


class InlineQueue:
    def __init__(self, deps: PipelineDeps) -> None:
        self._deps = deps

    async def enqueue_article(self, article_id: str) -> None:
        await process_article(article_id, self._deps)
