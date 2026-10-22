import pytest
from neo4j.exceptions import (
    ClientError,
    ConnectionAcquisitionTimeoutError,
    ServiceUnavailable,
    SessionExpired,
    TransientError,
)
from redis.exceptions import AuthenticationError, AuthorizationError, ResponseError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.pipeline.infrastructure import is_infrastructure_error
from app.pipeline.lock import ResolveLockLostError


def lock_lost(cause: BaseException | None) -> ResolveLockLostError:
    error = ResolveLockLostError("lock:graph-resolve ownership was lost")
    error.__cause__ = cause
    return error


@pytest.mark.parametrize(
    "error",
    [
        ServiceUnavailable("down"),
        SessionExpired("expired"),
        TransientError("deadlock"),
        ConnectionAcquisitionTimeoutError("pool exhausted"),
        RedisConnectionError("refused"),
        RedisTimeoutError("timed out"),
        lock_lost(RedisConnectionError("refused")),
    ],
    ids=repr,
)
def test_outages_are_infrastructure_errors(error: BaseException) -> None:
    assert is_infrastructure_error(error)


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError("invalid password"),
        AuthorizationError("no permission"),
        ResponseError("WRONGTYPE"),
        ClientError("syntax error"),
        RuntimeError("bug"),
        lock_lost(None),
        lock_lost(RuntimeError("bug")),
    ],
    ids=repr,
)
def test_configuration_and_logic_errors_are_final(error: BaseException) -> None:
    assert not is_infrastructure_error(error)
