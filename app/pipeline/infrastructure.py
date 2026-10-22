from neo4j.exceptions import (
    ConnectionAcquisitionTimeoutError,
    ServiceUnavailable,
    SessionExpired,
    TransientError,
)
from redis.exceptions import AuthenticationError as RedisAuthenticationError
from redis.exceptions import AuthorizationError as RedisAuthorizationError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.pipeline.lock import ResolveLockLostError

INFRASTRUCTURE_ERRORS: tuple[type[Exception], ...] = (
    ServiceUnavailable,
    SessionExpired,
    TransientError,
    ConnectionAcquisitionTimeoutError,
    RedisConnectionError,
    RedisTimeoutError,
)
CONFIGURATION_ERRORS: tuple[type[Exception], ...] = (
    RedisAuthenticationError,
    RedisAuthorizationError,
)


def is_infrastructure_error(error: BaseException) -> bool:
    if isinstance(error, CONFIGURATION_ERRORS):
        return False
    if isinstance(error, ResolveLockLostError):
        return error.__cause__ is not None and is_infrastructure_error(error.__cause__)
    return isinstance(error, INFRASTRUCTURE_ERRORS)
