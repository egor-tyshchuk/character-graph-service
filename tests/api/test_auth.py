import hashlib

import httpx
import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient

from app.auth.cli import main
from app.config import Settings
from app.main import create_app
from tests.conftest import TEST_APP_TOKEN
from tests.fakes import FakeLLMClient


async def test_token_endpoint_returns_pair(client: AsyncClient) -> None:
    r = await client.post("/auth/token", json={"app_token": TEST_APP_TOKEN})

    assert r.status_code == 200
    body = r.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 900


async def test_token_endpoint_wrong_token_401(client: AsyncClient) -> None:
    r = await client.post("/auth/token", json={"app_token": "nope"})

    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"
    assert r.headers["www-authenticate"] == "Bearer"


async def test_refresh_endpoint_rotates(client: AsyncClient) -> None:
    pair = (await client.post("/auth/token", json={"app_token": TEST_APP_TOKEN})).json()

    r = await client.post("/auth/refresh", json={"refresh_token": pair["refresh_token"]})

    assert r.status_code == 200
    assert r.json()["refresh_token"] != pair["refresh_token"]
    replay = await client.post("/auth/refresh", json={"refresh_token": pair["refresh_token"]})
    assert replay.status_code == 401


async def test_auth_headers_fixture_yields_bearer(auth_headers: dict[str, str]) -> None:
    assert auth_headers["Authorization"].startswith("Bearer ")


def test_cli_prints_token_and_entry(capsys: pytest.CaptureFixture[str]) -> None:
    main(["new-app-token", "--app-id", "x"])

    lines = capsys.readouterr().out.splitlines()
    token = next(line.removeprefix("token: ") for line in lines if line.startswith("token: "))
    entry = next(
        line.removeprefix("APP_TOKENS entry: ")
        for line in lines
        if line.startswith("APP_TOKENS entry: ")
    )
    assert entry == f"x:{hashlib.sha256(token.encode()).hexdigest()}"


async def test_removed_app_is_rejected_after_restart(
    client: AsyncClient, settings: Settings, fake_llm: FakeLLMClient
) -> None:
    pair = (await client.post("/auth/token", json={"app_token": TEST_APP_TOKEN})).json()
    restarted = create_app(settings.model_copy(update={"app_tokens": {}}), llm=fake_llm)

    async with LifespanManager(restarted):
        transport = httpx.ASGITransport(app=restarted)
        async with AsyncClient(transport=transport, base_url="http://test") as after_removal:
            refresh = await after_removal.post(
                "/auth/refresh", json={"refresh_token": pair["refresh_token"]}
            )
            read = await after_removal.get(
                "/characters", headers={"Authorization": f"Bearer {pair['access_token']}"}
            )

    assert refresh.status_code == 401
    assert read.status_code == 401
    assert read.json()["error"]["code"] == "unauthorized"
