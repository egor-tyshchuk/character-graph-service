from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.types import Message, Receive, Scope, Send
from structlog.testing import capture_logs

from app.api.errors import install_exception_handlers
from app.api.middleware import RequestIdMiddleware
from app.api.routes import health
from app.errors import AuthError, NotFoundError, ResourceBusyError
from app.llm.client import LLMError, LLMNotConfiguredError


class Body(BaseModel):
    name: str


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    install_exception_handlers(app)

    @app.post("/validate")
    async def validate(body: Body) -> None: ...

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    @app.get("/missing")
    async def missing() -> None:
        raise NotFoundError("thing not found")

    @app.get("/auth")
    async def auth() -> None:
        raise AuthError("bad token")

    @app.get("/busy")
    async def busy() -> None:
        raise ResourceBusyError("graph is busy")

    @app.get("/llm-unconfigured")
    async def llm_unconfigured() -> None:
        raise LLMNotConfiguredError("OPENAI_API_KEY is not set")

    @app.get("/llm-failed")
    async def llm_failed() -> None:
        raise LLMError("embedding request failed: upstream secret detail")

    app.include_router(health.router)
    return app


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = _build_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        c.app = app  # type: ignore[attr-defined]
        yield c


async def test_validation_error_shape(client: httpx.AsyncClient) -> None:
    r = await client.post("/validate", json={})
    assert r.status_code == 422
    error = r.json()["error"]
    assert error["code"] == "validation_error"
    assert "name" in error["message"]


async def test_unhandled_exception_returns_500_with_request_id(client: httpx.AsyncClient) -> None:
    r = await client.get("/boom", headers={"X-Request-ID": "abc123"})
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "Internal server error"}}
    assert r.headers["x-request-id"] == "abc123"


async def test_app_error_maps_status_and_code(client: httpx.AsyncClient) -> None:
    r = await client.get("/missing")
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "not_found", "message": "thing not found"}}


async def test_auth_error_sets_www_authenticate(client: httpx.AsyncClient) -> None:
    r = await client.get("/auth")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"
    assert r.headers["www-authenticate"] == "Bearer"


async def test_resource_busy_maps_to_409_with_retry_after(client: httpx.AsyncClient) -> None:
    r = await client.get("/busy")
    assert r.status_code == 409
    assert r.json() == {"error": {"code": "resource_busy", "message": "graph is busy"}}
    assert r.headers["retry-after"] == "5"
    assert "www-authenticate" not in r.headers


async def test_llm_not_configured_maps_to_503_and_logs_without_traceback(
    client: httpx.AsyncClient,
) -> None:
    with capture_logs() as logs:
        r = await client.get("/llm-unconfigured")

    assert r.status_code == 503
    assert r.json() == {"error": {"code": "llm_unavailable", "message": "LLM is not configured"}}
    [entry] = [entry for entry in logs if entry["event"] == "llm_unavailable"]
    assert entry["log_level"] == "warning"
    assert "exc_info" not in entry


async def test_llm_error_maps_to_503_and_logs_with_traceback(client: httpx.AsyncClient) -> None:
    with capture_logs() as logs:
        r = await client.get("/llm-failed")

    assert r.status_code == 503
    assert r.json() == {
        "error": {"code": "llm_unavailable", "message": "LLM provider is unavailable"}
    }
    [entry] = [entry for entry in logs if entry["event"] == "llm_unavailable"]
    assert entry["log_level"] == "warning"
    assert isinstance(entry["exc_info"], LLMError)


class _FailingRedis:
    async def ping(self) -> None:
        raise ConnectionError("redis down")


class _HealthyDriver:
    async def verify_connectivity(self) -> None:
        return None


async def test_ready_returns_503_when_dependency_fails(client: httpx.AsyncClient) -> None:
    app: Any = client.app  # type: ignore[attr-defined]
    app.state.container = SimpleNamespace(driver=_HealthyDriver(), redis=_FailingRedis())
    r = await client.get("/health/ready")
    assert r.status_code == 503
    assert r.json() == {"status": "error", "neo4j": "ok", "redis": "error"}


@pytest.mark.parametrize("request_id", ["abc123", "A.b_c-9", "x" * 128])
async def test_well_formed_request_id_is_echoed(client: httpx.AsyncClient, request_id: str) -> None:
    r = await client.get("/missing", headers={"X-Request-ID": request_id})

    assert r.headers["x-request-id"] == request_id


@pytest.mark.parametrize(
    "request_id", ["", "has space", "x" * 129, "semi;colon", "line\tbreak", "ünïcode"]
)
async def test_malformed_request_id_is_replaced(client: httpx.AsyncClient, request_id: str) -> None:
    with capture_logs() as logs:
        r = await client.get("/missing", headers={b"X-Request-ID": request_id.encode()})

    generated = r.headers["x-request-id"]
    assert generated != request_id
    assert len(generated) == 32
    assert int(generated, 16) >= 0
    [entry] = [entry for entry in logs if entry["event"] == "request"]
    assert entry["status"] == 404


async def test_health_requests_are_logged_at_debug_and_others_at_info(
    client: httpx.AsyncClient,
) -> None:
    with capture_logs() as logs:
        await client.get("/health/live")
        await client.get("/missing")

    levels = {entry["path"]: entry["log_level"] for entry in logs if entry["event"] == "request"}
    assert levels == {"/health/live": "debug", "/missing": "info"}


async def test_failure_after_the_response_started_is_not_answered_twice() -> None:
    async def streams_then_fails(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        raise RuntimeError("stream broke")

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    scope: Scope = {"type": "http", "method": "GET", "path": "/stream", "headers": []}
    with pytest.raises(RuntimeError, match="stream broke"):
        await RequestIdMiddleware(streams_then_fails)(scope, receive, send)

    assert [message["type"] for message in sent] == ["http.response.start", "http.response.body"]
