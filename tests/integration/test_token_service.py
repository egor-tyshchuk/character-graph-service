from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from app.auth.tokens import TokenService
from app.config import Settings
from app.errors import AuthError
from tests.conftest import TEST_APP_ID, TEST_APP_TOKEN


@pytest_asyncio.fixture
async def redis(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.redis_url)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def service(settings: Settings, redis: Redis) -> TokenService:
    return TokenService(settings, redis)


async def test_issue_and_verify_access(service: TokenService) -> None:
    pair = await service.issue_pair(TEST_APP_ID)

    claims = service.verify_access(pair.access_token)

    assert claims.app_id == TEST_APP_ID
    assert claims.jti
    assert pair.token_type == "bearer"
    assert pair.expires_in == 900


async def test_refresh_rotates_and_old_refresh_is_rejected(service: TokenService) -> None:
    first = await service.issue_pair(TEST_APP_ID)

    second = await service.refresh(first.refresh_token)

    assert second.refresh_token != first.refresh_token
    assert service.verify_access(second.access_token).app_id == TEST_APP_ID
    with pytest.raises(AuthError):
        await service.refresh(first.refresh_token)


async def test_reusing_old_refresh_revokes_family(service: TokenService) -> None:
    r1 = (await service.issue_pair(TEST_APP_ID)).refresh_token
    r2 = (await service.refresh(r1)).refresh_token

    with pytest.raises(AuthError):
        await service.refresh(r1)
    with pytest.raises(AuthError):
        await service.refresh(r2)


async def test_families_are_independent(service: TokenService) -> None:
    a = await service.issue_pair(TEST_APP_ID)
    b = await service.issue_pair(TEST_APP_ID)

    with pytest.raises(AuthError):
        await service.refresh(a.refresh_token + "x")

    await service.refresh(b.refresh_token)


async def test_refresh_sets_ttl_on_family_key(
    service: TokenService, redis: Redis, settings: Settings
) -> None:
    pair = await service.issue_pair(TEST_APP_ID)
    await service.refresh(pair.refresh_token)

    keys = [key async for key in redis.scan_iter("refresh:family:*")]

    assert len(keys) == 1
    ttl = await redis.ttl(keys[0])
    assert 0 < ttl <= settings.refresh_token_ttl_seconds


async def test_access_token_cannot_refresh(service: TokenService) -> None:
    pair = await service.issue_pair(TEST_APP_ID)

    with pytest.raises(AuthError):
        await service.refresh(pair.access_token)


async def test_refresh_token_is_not_access(service: TokenService) -> None:
    pair = await service.issue_pair(TEST_APP_ID)

    with pytest.raises(AuthError):
        service.verify_access(pair.refresh_token)


async def test_expired_access_rejected(settings: Settings, redis: Redis) -> None:
    expired = TokenService(settings.model_copy(update={"access_token_ttl_seconds": -1}), redis)
    pair = await expired.issue_pair(TEST_APP_ID)

    with pytest.raises(AuthError):
        expired.verify_access(pair.access_token)


async def test_garbage_token_rejected(service: TokenService) -> None:
    with pytest.raises(AuthError):
        service.verify_access("not-a-jwt")


async def test_authenticate_app(service: TokenService) -> None:
    assert service.authenticate_app(TEST_APP_TOKEN) == TEST_APP_ID


async def test_wrong_app_token_rejected(service: TokenService) -> None:
    with pytest.raises(AuthError):
        service.authenticate_app("wrong")
