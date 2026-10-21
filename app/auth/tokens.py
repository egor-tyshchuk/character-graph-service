import hashlib
import hmac
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import jwt
import structlog
from pydantic import BaseModel
from redis.asyncio import Redis

from app.config import Settings
from app.errors import AuthError

logger = structlog.get_logger(__name__)

TokenType = Literal["access", "refresh"]

FAMILY_KEY = "refresh:family:{family}"
FAMILY_STARTED_CLAIM = "fam_iat"
REVOKED_APP_ERROR = "App is no longer authorized"

_CAS_MISSING = 0
_CAS_ROTATED = 1
_CAS_REUSED = 2

_ROTATE_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if not current then
    return 0
end
if current == ARGV[1] then
    redis.call('SET', KEYS[1], ARGV[2], 'EX', tonumber(ARGV[3]))
    return 1
end
redis.call('DEL', KEYS[1])
return 2
"""


def hash_app_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class AccessClaims(BaseModel):
    app_id: str
    jti: str
    expires_at: datetime


class TokenService:
    def __init__(
        self,
        settings: Settings,
        redis: Redis,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._settings = settings
        self._redis = redis
        self._clock = clock
        self._rotate = redis.register_script(_ROTATE_SCRIPT)

    def authenticate_app(self, app_token: str) -> str:
        presented = hash_app_token(app_token)
        matched: str | None = None
        for app_id, digest in self._settings.app_tokens.items():
            if hmac.compare_digest(presented, digest) and matched is None:
                matched = app_id
        if matched is None:
            raise AuthError("Invalid app token")
        return matched

    async def issue_pair(self, app_id: str) -> TokenPair:
        family = str(uuid.uuid4())
        pair, refresh_jti = self._build_pair(app_id, family, int(self._clock().timestamp()))
        await self._redis.set(
            FAMILY_KEY.format(family=family),
            refresh_jti,
            ex=self._settings.refresh_token_ttl_seconds,
        )
        return pair

    async def refresh(self, refresh_token: str) -> TokenPair:
        claims = self._decode(refresh_token, "refresh")
        app_id, family, presented_jti = claims["sub"], claims["family"], claims["jti"]
        family_started = int(claims[FAMILY_STARTED_CLAIM])
        if app_id not in self._settings.app_tokens:
            await self._revoke(family, app_id, "app_removed")
            raise AuthError(REVOKED_APP_ERROR)
        if self._family_expired(family_started):
            await self._revoke(family, app_id, "family_max_lifetime_reached")
            raise AuthError("Refresh token family expired; request a new token pair")
        pair, new_jti = self._build_pair(app_id, family, family_started)
        outcome = await self._rotate(
            keys=[FAMILY_KEY.format(family=family)],
            args=[presented_jti, new_jti, self._settings.refresh_token_ttl_seconds],
        )
        if outcome == _CAS_ROTATED:
            return pair
        if outcome == _CAS_MISSING:
            raise AuthError("Refresh token revoked or expired")
        logger.warning("refresh_token_reuse_detected", app_id=app_id, family=family)
        raise AuthError("Refresh token revoked or expired")

    def verify_access(self, token: str) -> AccessClaims:
        claims = self._decode(token, "access")
        if claims["sub"] not in self._settings.app_tokens:
            raise AuthError(REVOKED_APP_ERROR)
        return AccessClaims(
            app_id=claims["sub"],
            jti=claims["jti"],
            expires_at=datetime.fromtimestamp(claims["exp"], UTC),
        )

    def _family_expired(self, family_started: int) -> bool:
        age = self._clock().timestamp() - family_started
        return age > self._settings.refresh_family_max_lifetime_seconds

    async def _revoke(self, family: str, app_id: str, reason: str) -> None:
        await self._redis.delete(FAMILY_KEY.format(family=family))
        logger.info("refresh_family_revoked", app_id=app_id, family=family, reason=reason)

    def _build_pair(self, app_id: str, family: str, family_started: int) -> tuple[TokenPair, str]:
        access_ttl = self._settings.access_token_ttl_seconds
        refresh_jti = str(uuid.uuid4())
        access = self._encode(app_id, "access", str(uuid.uuid4()), access_ttl)
        refresh = self._encode(
            app_id,
            "refresh",
            refresh_jti,
            self._settings.refresh_token_ttl_seconds,
            family=family,
            **{FAMILY_STARTED_CLAIM: family_started},
        )
        return TokenPair(access_token=access, refresh_token=refresh, expires_in=access_ttl), (
            refresh_jti
        )

    def _encode(
        self, app_id: str, token_type: TokenType, jti: str, ttl_seconds: int, **extra: str | int
    ) -> str:
        now = self._clock()
        payload: dict[str, Any] = {
            "sub": app_id,
            "type": token_type,
            "jti": jti,
            "iat": now,
            "exp": now + timedelta(seconds=ttl_seconds),
            **extra,
        }
        return jwt.encode(
            payload,
            self._settings.jwt_secret.get_secret_value(),
            algorithm=self._settings.jwt_algorithm,
        )

    def _decode(self, token: str, expected_type: TokenType) -> dict[str, Any]:
        required = ["sub", "type", "jti", "exp", "iat"]
        if expected_type == "refresh":
            required.extend(["family", FAMILY_STARTED_CLAIM])
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                self._settings.jwt_secret.get_secret_value(),
                algorithms=[self._settings.jwt_algorithm],
                options={"require": required},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("Token expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthError("Invalid token") from exc
        if claims["type"] != expected_type:
            raise AuthError("Invalid token type")
        return claims
