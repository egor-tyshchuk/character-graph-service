import re
import time
import uuid
from collections.abc import Collection

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import app_error_response, internal_error_response
from app.errors import PayloadTooLargeError

logger = structlog.get_logger(__name__)

REQUEST_ID_HEADER = "x-request-id"
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}")
QUIET_PATH_PREFIX = "/health/"
JSON_BODY_MAX_BYTES = 64 * 1024


def _request_id(scope: Scope) -> str:
    incoming = Headers(scope=scope).get(REQUEST_ID_HEADER, "")
    if REQUEST_ID_PATTERN.fullmatch(incoming):
        return incoming
    return uuid.uuid4().hex


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        status_code = 500
        response_started = False
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_started = True
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            try:
                await self.app(scope, receive, send_with_request_id)
            except Exception:
                logger.error("unhandled_exception", exc_info=True)
                if response_started:
                    raise
                await internal_error_response()(scope, receive, send_with_request_id)
        finally:
            log = logger.debug if scope["path"].startswith(QUIET_PATH_PREFIX) else logger.info
            log(
                "request",
                method=scope["method"],
                path=scope["path"],
                status=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            structlog.contextvars.clear_contextvars()


class BodyLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int = JSON_BODY_MAX_BYTES,
        exempt_paths: Collection[str] = (),
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.exempt_paths = frozenset(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self.exempt_paths:
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                pending = [message]
                break
            body += message.get("body", b"")
            if len(body) > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body", False):
                pending = [{"type": "http.request", "body": bytes(body), "more_body": False}]
                break

        async def replay() -> Message:
            return pending.pop() if pending else await receive()

        await self.app(scope, replay, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        error = PayloadTooLargeError(f"Request body exceeds the limit of {self.max_bytes} bytes")
        await app_error_response(error)(scope, receive, send)
