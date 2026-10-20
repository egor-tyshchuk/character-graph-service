from typing import Any, cast

import pytest
from arq.connections import ArqRedis

from app import worker
from app.pipeline.process import PipelineDeps
from app.queue import PROCESS_ARTICLE_JOB, ArqJobQueue
from app.worker import WorkerSettings, process_article_job


class RecordingPool:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.kwargs: list[dict[str, Any]] = []

    async def enqueue_job(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(args)
        self.kwargs.append(kwargs)


def test_worker_settings_registers_job() -> None:
    assert process_article_job in WorkerSettings.functions
    assert process_article_job.__name__ == PROCESS_ARTICLE_JOB


def test_worker_settings_limits() -> None:
    assert WorkerSettings.job_timeout == 3600
    assert WorkerSettings.max_tries == 1
    assert WorkerSettings.max_jobs == 4
    assert WorkerSettings.on_startup is worker.startup
    assert WorkerSettings.on_shutdown is worker.shutdown


async def test_arq_queue_enqueues_job_name() -> None:
    pool = RecordingPool()

    await ArqJobQueue(cast(ArqRedis, pool)).enqueue_article("article-1")

    assert pool.calls == [("process_article_job", "article-1")]
    assert pool.kwargs == [{}]


async def test_process_article_job_runs_pipeline_with_worker_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = cast(PipelineDeps, object())
    processed: list[tuple[str, PipelineDeps]] = []

    async def fake_process(article_id: str, received: PipelineDeps) -> None:
        processed.append((article_id, received))

    monkeypatch.setattr(worker, "process_article", fake_process)

    await process_article_job({"deps": deps, "job_id": "job-1"}, "article-1")

    assert processed == [("article-1", deps)]
