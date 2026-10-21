import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from app.container import Container
from app.graph.repository import GraphRepository
from app.llm.client import UnconfiguredLLMClient
from tests.integration.test_graph_repository import character, persist, relation, seed_article


@pytest.fixture
def repo(app: FastAPI) -> GraphRepository:
    container: Container = app.state.container
    return container.repo


@pytest_asyncio.fixture
async def duplicates(repo: GraphRepository) -> tuple[str, str]:
    anna, anya, mark = character("Anna Petrova"), character("Anya"), character("Mark Petrov")
    await persist(repo, await seed_article(repo, "First"), [anna, mark])
    await persist(repo, await seed_article(repo, "Second"), [anya, mark], [relation(anya, mark)])
    return anna.id, anya.id


async def test_merge_requires_auth(client: httpx.AsyncClient) -> None:
    response = await client.post("/characters/merge", json={"target_id": "a", "source_ids": ["b"]})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_merge_unknown_id_404(
    client: httpx.AsyncClient, auth_headers: dict[str, str], duplicates: tuple[str, str]
) -> None:
    target_id, _ = duplicates

    response = await client.post(
        "/characters/merge",
        json={"target_id": target_id, "source_ids": ["missing-id"]},
        headers=auth_headers,
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert "missing-id" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "source_ids",
    [
        pytest.param(["target"], id="target-in-sources"),
        pytest.param(["target", "other"], id="target-among-sources"),
        pytest.param(["other", "other"], id="duplicates"),
        pytest.param([], id="empty"),
    ],
)
async def test_merge_target_in_sources_422(
    client: httpx.AsyncClient, auth_headers: dict[str, str], source_ids: list[str]
) -> None:
    response = await client.post(
        "/characters/merge",
        json={"target_id": "target", "source_ids": source_ids},
        headers=auth_headers,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_merge_returns_character(
    client: httpx.AsyncClient, auth_headers: dict[str, str], duplicates: tuple[str, str]
) -> None:
    target_id, source_id = duplicates

    response = await client.post(
        "/characters/merge",
        json={"target_id": target_id, "source_ids": [source_id]},
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "id": target_id,
        "name": "Anna Petrova",
        "aliases": ["Anya"],
        "description": "Anna Petrova from the tale.",
        "article_count": 2,
    }
    listed = await client.get("/characters", headers=auth_headers)
    assert [item["name"] for item in listed.json()["items"]] == ["Anna Petrova", "Mark Petrov"]


async def test_merge_without_llm_returns_503_and_keeps_sources(
    app: FastAPI,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    duplicates: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_id, source_id = duplicates
    container: Container = app.state.container
    monkeypatch.setattr(container, "llm", UnconfiguredLLMClient())

    response = await client.post(
        "/characters/merge",
        json={"target_id": target_id, "source_ids": [source_id]},
        headers=auth_headers,
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"
    assert await container.repo.get_character(source_id) is not None
