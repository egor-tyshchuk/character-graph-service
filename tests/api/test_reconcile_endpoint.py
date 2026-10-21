import httpx
from fastapi import FastAPI

from app.container import Container
from tests.fakes import vec
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
