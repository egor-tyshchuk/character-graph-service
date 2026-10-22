from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

INFRASTRUCTURE_ERRORS: tuple[type[Exception], ...] = (
    ServiceUnavailable,
    SessionExpired,
    TransientError,
    RedisConnectionError,
    RedisTimeoutError,
)
