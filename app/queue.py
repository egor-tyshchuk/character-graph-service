from typing import Protocol

from arq.connections import ArqRedis

from app.pipeline.deps import PipelineDeps
from app.pipeline.process import process_article
from app.pipeline.reconcile import reconcile_characters

PROCESS_ARTICLE_JOB = "process_article_job"
RECONCILE_JOB = "reconcile_job"
RECONCILE_JOB_ID = "reconcile"


class JobQueue(Protocol):
    async def enqueue_article(self, article_id: str) -> None: ...

    async def enqueue_reconcile(self) -> None: ...


class ArqJobQueue:
    def __init__(self, pool: ArqRedis) -> None:
        self._pool = pool

    async def enqueue_article(self, article_id: str) -> None:
        await self._pool.enqueue_job(PROCESS_ARTICLE_JOB, article_id)

    async def enqueue_reconcile(self) -> None:
        await self._pool.enqueue_job(RECONCILE_JOB, _job_id=RECONCILE_JOB_ID)


class InlineQueue:
    def __init__(self, deps: PipelineDeps) -> None:
        self._deps = deps

    async def enqueue_article(self, article_id: str) -> None:
        await process_article(article_id, self._deps)

    async def enqueue_reconcile(self) -> None:
        await reconcile_characters(self._deps)
