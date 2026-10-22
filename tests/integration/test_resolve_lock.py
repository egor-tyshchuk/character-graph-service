import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from redis.asyncio import Redis
from redis.asyncio.lock import Lock
from redis.exceptions import ConnectionError as RedisConnectionError
from structlog.testing import capture_logs

from app.config import Settings
from app.pipeline.lock import (
    RESOLVE_LOCK_NAME,
    ResolveLockLostError,
    ResolveLockTimeoutError,
    resolve_lock,
)

LOCK_TIMEOUT = 1.0
LONGER_THAN_LOCK_TIMEOUT = LOCK_TIMEOUT * 1.5


@pytest_asyncio.fixture
async def redis(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.redis_url)
    await client.delete(RESOLVE_LOCK_NAME)
    try:
        yield client
    finally:
        await client.delete(RESOLVE_LOCK_NAME)
        await client.aclose()


@pytest.fixture
def fast_settings(settings: Settings) -> Settings:
    return Settings(
        **{
            **settings.model_dump(),
            "resolve_lock_timeout_seconds": LOCK_TIMEOUT,
            "resolve_lock_acquire_timeout_seconds": LOCK_TIMEOUT * 5,
            "llm_timeout_seconds": 0.01,
            "dedup_max_judge_calls_per_article": 1,
        }
    )


async def test_lock_is_extended_while_held(redis: Redis, fast_settings: Settings) -> None:
    async with resolve_lock(redis, fast_settings):
        await asyncio.sleep(LOCK_TIMEOUT * 2)
        assert await redis.exists(RESOLVE_LOCK_NAME) == 1
        assert 0 < await redis.pttl(RESOLVE_LOCK_NAME) <= LOCK_TIMEOUT * 1000

    assert await redis.exists(RESOLVE_LOCK_NAME) == 0


async def test_lock_serializes_holders(redis: Redis, fast_settings: Settings) -> None:
    events: list[str] = []

    async def hold(label: str) -> None:
        async with resolve_lock(redis, fast_settings):
            events.append(f"enter {label}")
            await asyncio.sleep(LONGER_THAN_LOCK_TIMEOUT)
            events.append(f"exit {label}")

    await asyncio.gather(hold("first"), hold("second"))

    assert [event.split()[0] for event in events] == ["enter", "exit", "enter", "exit"]
    assert events[0].split()[1] == events[1].split()[1]


async def test_lock_releases_when_body_raises(redis: Redis, fast_settings: Settings) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        async with resolve_lock(redis, fast_settings):
            raise RuntimeError("boom")

    assert await redis.exists(RESOLVE_LOCK_NAME) == 0


async def test_lock_acquire_timeout_raises(redis: Redis, fast_settings: Settings) -> None:
    await redis.set(RESOLVE_LOCK_NAME, b"someone-else", px=10_000)
    impatient = fast_settings.model_copy(update={"resolve_lock_acquire_timeout_seconds": 0.2})

    with pytest.raises(ResolveLockTimeoutError, match=r"within 0\.2s"):
        async with resolve_lock(redis, impatient):
            pytest.fail("lock must not be acquired")


async def test_lost_lock_is_logged_and_release_failure_not_raised(
    redis: Redis, fast_settings: Settings
) -> None:
    with capture_logs() as logs:
        async with resolve_lock(redis, fast_settings):
            await redis.delete(RESOLVE_LOCK_NAME)
            await asyncio.sleep(LOCK_TIMEOUT)

    events = [entry["event"] for entry in logs]
    assert "resolve_lock_lost" in events
    assert "resolve_lock_release_failed" in events


async def test_ensure_owned_passes_while_held(redis: Redis, fast_settings: Settings) -> None:
    async with resolve_lock(redis, fast_settings) as held:
        await asyncio.sleep(LOCK_TIMEOUT)
        await held.ensure_owned()


async def test_ensure_owned_raises_after_extension_failed(
    redis: Redis, fast_settings: Settings
) -> None:
    with capture_logs() as logs:
        async with resolve_lock(redis, fast_settings, article_id="a-1") as held:
            await redis.delete(RESOLVE_LOCK_NAME)
            await asyncio.sleep(LOCK_TIMEOUT)
            await redis.set(RESOLVE_LOCK_NAME, b"someone-else", px=10_000)
            with pytest.raises(ResolveLockLostError) as lost_error:
                await held.ensure_owned()

    assert lost_error.value.__cause__ is None
    lost = [entry for entry in logs if entry["event"] == "resolve_lock_lost"]
    assert [entry["article_id"] for entry in lost] == ["a-1"]


async def test_lock_lost_during_a_redis_outage_names_the_redis_error(
    redis: Redis, fast_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    outage = RedisConnectionError("connection refused")

    async def failing_extend(self: Lock, additional_time: float, replace_ttl: bool) -> bool:
        raise outage

    async with resolve_lock(redis, fast_settings) as held:
        monkeypatch.setattr(Lock, "extend", failing_extend)
        await redis.delete(RESOLVE_LOCK_NAME)
        await asyncio.sleep(LOCK_TIMEOUT / 2)
        with pytest.raises(ResolveLockLostError) as lost_error:
            await held.ensure_owned()

    assert lost_error.value.__cause__ is outage


async def test_ensure_owned_detects_loss_before_extender_notices(
    redis: Redis, fast_settings: Settings
) -> None:
    with capture_logs() as logs:
        async with resolve_lock(redis, fast_settings, article_id="a-2") as held:
            await redis.delete(RESOLVE_LOCK_NAME)
            with pytest.raises(ResolveLockLostError):
                await held.ensure_owned()

    lost = [entry for entry in logs if entry["event"] == "resolve_lock_lost"]
    assert [entry["article_id"] for entry in lost] == ["a-2"]
