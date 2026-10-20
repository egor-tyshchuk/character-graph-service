import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from redis.asyncio import Redis
from redis.asyncio.lock import Lock
from redis.exceptions import LockError, RedisError

from app.config import Settings

RESOLVE_LOCK_NAME = "lock:graph-resolve"
EXTENSIONS_PER_TIMEOUT = 3
ACQUIRE_TIMEOUT_FACTOR = 5

logger = structlog.get_logger(__name__)


class ResolveLockTimeoutError(TimeoutError):
    pass


@asynccontextmanager
async def resolve_lock(redis: Redis, settings: Settings) -> AsyncIterator[None]:
    timeout = settings.resolve_lock_timeout_seconds
    blocking_timeout = timeout * ACQUIRE_TIMEOUT_FACTOR
    lock = redis.lock(RESOLVE_LOCK_NAME, timeout=timeout, blocking_timeout=blocking_timeout)
    if not await lock.acquire():
        raise ResolveLockTimeoutError(
            f"could not acquire {RESOLVE_LOCK_NAME} within {blocking_timeout}s"
        )
    extender = asyncio.create_task(_keep_extended(lock, timeout))
    try:
        yield
    finally:
        extender.cancel()
        try:
            await asyncio.wait([extender])
        finally:
            await _release(lock)


async def _keep_extended(lock: Lock, ttl_seconds: float) -> None:
    while True:
        await asyncio.sleep(ttl_seconds / EXTENSIONS_PER_TIMEOUT)
        try:
            await lock.extend(ttl_seconds, replace_ttl=True)
        except LockError as exc:
            logger.error("resolve_lock_lost", lock=RESOLVE_LOCK_NAME, error=str(exc))
            return
        except RedisError as exc:
            logger.warning("resolve_lock_extend_failed", lock=RESOLVE_LOCK_NAME, error=str(exc))


async def _release(lock: Lock) -> None:
    try:
        await lock.release()
    except LockError as exc:
        logger.warning("resolve_lock_release_failed", lock=RESOLVE_LOCK_NAME, error=str(exc))
