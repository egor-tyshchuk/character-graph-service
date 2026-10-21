import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from redis.asyncio import Redis
from redis.asyncio.lock import Lock
from redis.exceptions import LockError, RedisError

from app.config import Settings

RESOLVE_LOCK_NAME = "lock:graph-resolve"
EXTENSIONS_PER_TIMEOUT = 3

logger = structlog.get_logger(__name__)


class ResolveLockTimeoutError(TimeoutError):
    pass


class ResolveLockLostError(RuntimeError):
    pass


class HeldLock:
    def __init__(self, lock: Lock, log_context: dict[str, Any]) -> None:
        self._lock = lock
        self._log_context = log_context
        self._lost = False

    async def ensure_owned(self) -> None:
        if not self._lost and await self._lock.owned():
            return
        if not self._lost:
            self.mark_lost("lock key no longer holds this owner's token")
        raise ResolveLockLostError(f"{RESOLVE_LOCK_NAME} ownership was lost")

    def mark_lost(self, error: str) -> None:
        self._lost = True
        logger.error("resolve_lock_lost", lock=RESOLVE_LOCK_NAME, error=error, **self._log_context)

    async def keep_extended(self, ttl_seconds: float) -> None:
        while True:
            await asyncio.sleep(ttl_seconds / EXTENSIONS_PER_TIMEOUT)
            try:
                await self._lock.extend(ttl_seconds, replace_ttl=True)
            except LockError as exc:
                self.mark_lost(str(exc))
                return
            except RedisError as exc:
                logger.warning(
                    "resolve_lock_extend_failed",
                    lock=RESOLVE_LOCK_NAME,
                    error=str(exc),
                    **self._log_context,
                )

    async def release(self) -> None:
        try:
            await self._lock.release()
        except RedisError as exc:
            logger.warning(
                "resolve_lock_release_failed",
                lock=RESOLVE_LOCK_NAME,
                error=str(exc),
                **self._log_context,
            )


@asynccontextmanager
async def resolve_lock(
    redis: Redis, settings: Settings, *, acquire_timeout: float | None = None, **log_context: Any
) -> AsyncIterator[HeldLock]:
    timeout = settings.resolve_lock_timeout_seconds
    blocking_timeout = (
        settings.resolve_lock_acquire_timeout_seconds
        if acquire_timeout is None
        else acquire_timeout
    )
    lock = redis.lock(RESOLVE_LOCK_NAME, timeout=timeout, blocking_timeout=blocking_timeout)
    if not await lock.acquire():
        raise ResolveLockTimeoutError(
            f"could not acquire {RESOLVE_LOCK_NAME} within {blocking_timeout}s"
        )
    held = HeldLock(lock, log_context)
    extender = asyncio.create_task(held.keep_extended(timeout))
    try:
        yield held
    finally:
        extender.cancel()
        try:
            await asyncio.wait([extender])
        finally:
            await held.release()
