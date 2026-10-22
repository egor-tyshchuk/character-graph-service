from types import SimpleNamespace
from typing import Any, cast

import pytest
from arq import Retry
from arq.connections import ArqRedis
from arq.cron import CronJob
from arq.worker import Function
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from pydantic import SecretStr
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from structlog.testing import capture_logs

from app import worker
from app.config import ScheduleSettings, Settings
from app.pipeline.deps import PipelineDeps
from app.pipeline.reconcile import ReconcileReport
from app.queue import PROCESS_ARTICLE_JOB, RECONCILE_JOB, RECONCILE_JOB_ID, ArqJobQueue
from app.worker import (
    WorkerSettings,
    build_cron_jobs,
    process_article_job,
    reconcile_job,
    sweep_stuck_articles_job,
)

SWEEP_CRON = "cron:sweep_stuck_articles_job"
RECONCILE_CRON = "cron:reconcile_job"


class RecordingPool:
    def __init__(self, *, accepts: bool = True) -> None:
        self.accepts = accepts
        self.calls: list[tuple[Any, ...]] = []
        self.kwargs: list[dict[str, Any]] = []

    async def enqueue_job(self, *args: Any, **kwargs: Any) -> object | None:
        self.calls.append(args)
        self.kwargs.append(kwargs)
        return object() if self.accepts else None


def test_worker_settings_registers_job() -> None:
    assert process_article_job in WorkerSettings.functions
    assert process_article_job.__name__ == PROCESS_ARTICLE_JOB


def test_worker_settings_limits() -> None:
    assert WorkerSettings.job_timeout == worker.SCHEDULE.job_timeout_seconds
    assert ScheduleSettings().job_timeout_seconds == 3600
    assert ScheduleSettings().job_max_tries == 3
    assert ScheduleSettings().job_retry_delay_seconds == 30
    assert WorkerSettings.max_tries == worker.SCHEDULE.job_max_tries
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


INFRASTRUCTURE_FAILURES = [
    ServiceUnavailable("neo4j is down"),
    SessionExpired("session expired"),
    TransientError("deadlock"),
    RedisConnectionError("connection refused"),
    RedisTimeoutError("timed out"),
]


def deps_with(settings: Settings) -> PipelineDeps:
    return cast(PipelineDeps, SimpleNamespace(settings=settings))


def failing_process(error: Exception) -> Any:
    async def process(article_id: str, received: PipelineDeps) -> None:
        raise error

    return process


@pytest.mark.parametrize("error", INFRASTRUCTURE_FAILURES, ids=lambda e: type(e).__name__)
async def test_infrastructure_error_retries_the_job(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    deps = deps_with(make_settings(job_max_tries=3, job_retry_delay_seconds=30))
    monkeypatch.setattr(worker, "process_article", failing_process(error))

    with pytest.raises(Retry) as retry:
        await process_article_job({"deps": deps, "job_id": "job-1", "job_try": 2}, "article-1")

    assert retry.value.defer_score == 30_000
    assert retry.value.__cause__ is error


async def test_last_try_logs_retry_exhausted_and_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = deps_with(make_settings(job_max_tries=3))
    error = ServiceUnavailable("neo4j is down")
    monkeypatch.setattr(worker, "process_article", failing_process(error))

    with capture_logs() as logs, pytest.raises(ServiceUnavailable):
        await process_article_job({"deps": deps, "job_id": "job-1", "job_try": 3}, "article-1")

    [exhausted] = [entry for entry in logs if entry["event"] == "article_retry_exhausted"]
    assert exhausted["attempts"] == 3


async def test_other_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = deps_with(make_settings())
    monkeypatch.setattr(worker, "process_article", failing_process(RuntimeError("bug")))

    with pytest.raises(RuntimeError, match="bug"):
        await process_article_job({"deps": deps, "job_id": "job-1", "job_try": 1}, "article-1")


def make_settings(**overrides: Any) -> Settings:
    return Settings.model_validate({"jwt_secret": SecretStr("x" * 32), **overrides})


def by_name(jobs: list[CronJob]) -> dict[str, CronJob]:
    return {job.name: job for job in jobs}


def test_reconcile_disabled_by_default() -> None:
    jobs = by_name(build_cron_jobs(make_settings()))

    assert RECONCILE_CRON not in jobs
    assert SWEEP_CRON in jobs
    assert set(by_name(build_cron_jobs(ScheduleSettings()))) == {SWEEP_CRON}
    assert set(by_name(list(WorkerSettings.cron_jobs))) == set(
        by_name(build_cron_jobs(worker.SCHEDULE))
    )


def test_cron_registered_when_enabled() -> None:
    jobs = by_name(build_cron_jobs(make_settings(reconcile_enabled=True)))

    assert set(jobs) == {SWEEP_CRON, RECONCILE_CRON}
    reconcile = jobs[RECONCILE_CRON]
    assert reconcile.coroutine is reconcile_job
    assert reconcile.job_id == RECONCILE_JOB_ID
    assert reconcile.unique is True
    assert reconcile.run_at_startup is False
    assert (reconcile.minute, reconcile.hour) == ({0}, None)
    assert jobs[SWEEP_CRON].coroutine is sweep_stuck_articles_job


def test_no_cron_when_disabled() -> None:
    settings = make_settings(reconcile_enabled=False, stuck_sweep_enabled=False)

    assert build_cron_jobs(settings) == []


@pytest.mark.parametrize(
    ("minutes", "minute", "hour"),
    [
        (10, {0, 10, 20, 30, 40, 50}, None),
        (15, {0, 15, 30, 45}, None),
        (60, {0}, None),
        (180, {0}, {0, 3, 6, 9, 12, 15, 18, 21}),
        (1440, {0}, {0}),
    ],
)
def test_cron_schedule_honors_interval(
    minutes: int, minute: set[int], hour: set[int] | None
) -> None:
    settings = make_settings(
        reconcile_enabled=True,
        reconcile_interval_minutes=minutes,
        stuck_sweep_interval_minutes=minutes,
    )

    jobs = build_cron_jobs(settings)

    assert len(jobs) == 2
    assert all((job.minute, job.hour, job.unique) == (minute, hour, True) for job in jobs)


def test_sweep_cron_follows_its_own_interval() -> None:
    jobs = by_name(build_cron_jobs(make_settings(stuck_sweep_interval_minutes=30)))

    assert jobs[SWEEP_CRON].minute == {0, 30}


def test_worker_registers_reconcile_job_without_kept_result() -> None:
    [reconcile] = [
        f for f in WorkerSettings.functions if isinstance(f, Function) and f.name == RECONCILE_JOB
    ]

    assert reconcile.coroutine is reconcile_job
    assert reconcile.keep_result_s == 0


async def test_arq_queue_enqueues_reconcile_with_fixed_job_id() -> None:
    pool = RecordingPool()

    assert await ArqJobQueue(cast(ArqRedis, pool)).enqueue_reconcile() is True

    assert pool.calls == [("reconcile_job",)]
    assert pool.kwargs == [{"_job_id": "reconcile"}]


async def test_arq_queue_reports_a_reconcile_run_already_queued() -> None:
    pool = RecordingPool(accepts=False)

    assert await ArqJobQueue(cast(ArqRedis, pool)).enqueue_reconcile() is False


async def test_reconcile_job_runs_pipeline_and_returns_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deps = cast(PipelineDeps, object())
    report = ReconcileReport(checked=3, merged=1, reembedded=0, merges=[("old", "new")])
    received: list[PipelineDeps] = []

    async def fake_reconcile(got: PipelineDeps) -> ReconcileReport:
        received.append(got)
        return report

    monkeypatch.setattr(worker, "reconcile_characters", fake_reconcile)

    result = await reconcile_job({"deps": deps, "job_id": "reconcile"})

    assert received == [deps]
    assert result == report.model_dump()


async def test_sweep_job_runs_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = cast(PipelineDeps, object())
    received: list[PipelineDeps] = []

    async def fake_sweep(got: PipelineDeps) -> int:
        received.append(got)
        return 2

    monkeypatch.setattr(worker, "sweep_stuck_articles", fake_sweep)

    assert await sweep_stuck_articles_job({"deps": deps, "job_id": "sweep"}) == 2
    assert received == [deps]
