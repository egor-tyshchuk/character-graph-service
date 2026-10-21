import httpx
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from app.config import Settings
from app.container import Container
from app.main import create_app
from tests.conftest import TEST_APP_TOKEN
from tests.fakes import FakeLLMClient, vec
from tests.integration.test_graph_repository import character, persist, seed_article


async def test_reconcile_requires_auth(client: httpx.AsyncClient) -> None:
    response = await client.post("/characters/reconcile")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_reconcile_endpoint_queues(
    app: FastAPI, client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    container: Container = app.state.container
    anna, anya = character("Anna Petrova", embedding=vec(1)), character("Anya", embedding=vec(1))
    await persist(container.repo, await seed_article(container.repo, "First"), [anna])
    await persist(container.repo, await seed_article(container.repo, "Second"), [anya])

    response = await client.post("/characters/reconcile", headers=auth_headers)

    assert response.status_code == 202
    assert response.json() == {"status": "queued"}
    assert await container.repo.get_character(anya.id) is None
    assert await container.repo.get_character(anna.id) is not None


class AlreadyQueued:
    async def enqueue_article(self, article_id: str) -> None:
        return None

    async def enqueue_reconcile(self) -> bool:
        return False


async def test_reconcile_endpoint_reports_a_run_already_queued(
    settings: Settings, fake_llm: FakeLLMClient, clean_stores: None
) -> None:
    application = create_app(settings, llm=fake_llm, queue=AlreadyQueued())
    async with LifespanManager(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            token = await client.post("/auth/token", json={"app_token": TEST_APP_TOKEN})
            headers = {"Authorization": f"Bearer {token.json()['access_token']}"}
            response = await client.post("/characters/reconcile", headers=headers)

    assert response.status_code == 202
    assert response.json() == {"status": "already_queued"}
