import math
import uuid
from typing import Any

import httpx
import pytest
import pytest_asyncio

from app.llm.schemas import ExtractedCharacter, ExtractedRelation, ExtractionResult
from tests.fakes import FakeLLMClient, vec

LIGHTHOUSE = "# The Lighthouse\n\nAnna Petrova and Mark Petrov kept the lighthouse together.\n"
HARBOUR = "# The Harbour\n\nMark Petrov and Lena Volkova were old friends.\n"
FRIEND_VECTOR = vec(1)
SIBLING_VECTOR = vec(0, 0, 1)
HALF_FRIEND_VECTOR = vec(0.5, math.sqrt(0.75))
UNRELATED_VECTOR = vec(0, 0, 0, 1)


def mark() -> ExtractedCharacter:
    return ExtractedCharacter(name="Mark Petrov", aliases=[], description="Lighthouse keeper.")


def lighthouse_extraction() -> ExtractionResult:
    return ExtractionResult(
        characters=[
            ExtractedCharacter(name="Anna Petrova", aliases=[], description="Mark's sister."),
            mark(),
        ],
        relations=[
            ExtractedRelation(
                source="Anna Petrova",
                target="Mark Petrov",
                type="SIBLING_OF",
                description="Anna is Mark's sister.",
                evidence="Anna Petrova and Mark Petrov kept the lighthouse together.",
            )
        ],
    )


def harbour_extraction() -> ExtractionResult:
    return ExtractionResult(
        characters=[
            mark(),
            ExtractedCharacter(name="Lena Volkova", aliases=[], description="Mark's friend."),
        ],
        relations=[
            ExtractedRelation(
                source="Mark Petrov",
                target="Lena Volkova",
                type="FRIEND_OF",
                description="Mark and Lena are friends.",
                evidence="Mark Petrov and Lena Volkova were old friends.",
            )
        ],
    )


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient(
        dim=8,
        extractions={"Anna Petrova": lighthouse_extraction(), "Lena Volkova": harbour_extraction()},
        embeddings={
            "Anna Petrova": vec(0, 0, 0, 0, 1),
            "Mark Petrov": vec(0, 0, 0, 0, 0, 1),
            "Lena Volkova": vec(0, 0, 0, 0, 0, 0, 1),
            "FRIEND_OF": FRIEND_VECTOR,
            "SIBLING_OF": SIBLING_VECTOR,
            "друг": FRIEND_VECTOR,
            "enemy": HALF_FRIEND_VECTOR,
            "spaceship": UNRELATED_VECTOR,
        },
    )


@pytest_asyncio.fixture
async def seeded(client: httpx.AsyncClient, auth_headers: dict[str, str]) -> list[str]:
    article_ids = []
    for text in (LIGHTHOUSE, HARBOUR):
        response = await client.post(
            "/articles/upload", params={"wait": "true"}, data={"text": text}, headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["status"] == "done"
        article_ids.append(response.json()["id"])
    return article_ids


async def get_json(
    client: httpx.AsyncClient, path: str, headers: dict[str, str], **params: Any
) -> Any:
    response = await client.get(path, params=params, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


async def character_id(client: httpx.AsyncClient, headers: dict[str, str], name: str) -> str:
    page = await get_json(client, "/characters", headers)
    ids: dict[str, str] = {item["name"]: item["id"] for item in page["items"]}
    return ids[name]


@pytest.mark.parametrize(
    "path",
    [
        "/characters",
        "/characters/by_article/some-id",
        "/relations/types",
        "/relations/connected/some-id?type=FRIEND_OF",
    ],
)
async def test_all_read_endpoints_require_auth(client: httpx.AsyncClient, path: str) -> None:
    response = await client.get(path)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_list_characters(
    client: httpx.AsyncClient, auth_headers: dict[str, str], seeded: list[str]
) -> None:
    page = await get_json(client, "/characters", auth_headers)

    assert page["total"] == 3
    assert page["limit"] == 50
    assert page["offset"] == 0
    assert [item["name"] for item in page["items"]] == [
        "Anna Petrova",
        "Lena Volkova",
        "Mark Petrov",
    ]
    counts = {item["name"]: item["article_count"] for item in page["items"]}
    assert counts == {"Anna Petrova": 1, "Lena Volkova": 1, "Mark Petrov": 2}


async def test_list_characters_pagination(
    client: httpx.AsyncClient, auth_headers: dict[str, str], seeded: list[str]
) -> None:
    page = await get_json(client, "/characters", auth_headers, limit=1, offset=1)

    assert [item["name"] for item in page["items"]] == ["Lena Volkova"]
    assert (page["total"], page["limit"], page["offset"]) == (3, 1, 1)


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"offset": -1}])
async def test_list_characters_rejects_invalid_paging(
    client: httpx.AsyncClient, auth_headers: dict[str, str], params: dict[str, int]
) -> None:
    response = await client.get("/characters", params=params, headers=auth_headers)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_characters_by_article(
    client: httpx.AsyncClient, auth_headers: dict[str, str], seeded: list[str]
) -> None:
    characters = await get_json(client, f"/characters/by_article/{seeded[0]}", auth_headers)

    assert [item["name"] for item in characters] == ["Anna Petrova", "Mark Petrov"]


async def test_characters_by_unknown_article_404(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(f"/characters/by_article/{uuid.uuid4()}", headers=auth_headers)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_relation_types(
    client: httpx.AsyncClient, auth_headers: dict[str, str], seeded: list[str]
) -> None:
    types = await get_json(client, "/relations/types", auth_headers)

    assert {(item["name"], item["usage_count"]) for item in types} == {
        ("FRIEND_OF", 1),
        ("SIBLING_OF", 1),
    }
    assert {item["name"]: item["description"] for item in types}["FRIEND_OF"] == "friend of"


async def test_connected_exact_type(
    client: httpx.AsyncClient, auth_headers: dict[str, str], seeded: list[str]
) -> None:
    mark_id = await character_id(client, auth_headers, "Mark Petrov")
    lena_id = await character_id(client, auth_headers, "Lena Volkova")

    from_mark = await get_json(
        client, f"/relations/connected/{mark_id}", auth_headers, type="FRIEND_OF"
    )
    from_lena = await get_json(
        client, f"/relations/connected/{lena_id}", auth_headers, type="FRIEND_OF"
    )

    assert from_mark["character_id"] == mark_id
    assert from_mark["requested_type"] == "FRIEND_OF"
    assert from_mark["resolved_type"] == "FRIEND_OF"
    [connection] = from_mark["items"]
    assert connection["character"]["name"] == "Lena Volkova"
    assert connection["relation_type"] == "FRIEND_OF"
    assert connection["direction"] == "outgoing"
    assert connection["evidence"] == [
        {
            "article_id": seeded[1],
            "description": "Mark and Lena are friends.",
            "evidence": "Mark Petrov and Lena Volkova were old friends.",
        }
    ]
    [reverse] = from_lena["items"]
    assert reverse["character"]["name"] == "Mark Petrov"
    assert reverse["direction"] == "incoming"


@pytest.mark.parametrize("requested", ["friend of", "friend-of", "Friend_Of"])
async def test_connected_type_normalization(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    seeded: list[str],
    fake_llm: FakeLLMClient,
    requested: str,
) -> None:
    mark_id = await character_id(client, auth_headers, "Mark Petrov")
    embed_calls_before = len(fake_llm.embed_calls)

    body = await get_json(client, f"/relations/connected/{mark_id}", auth_headers, type=requested)

    assert body["requested_type"] == requested
    assert body["resolved_type"] == "FRIEND_OF"
    assert [item["character"]["name"] for item in body["items"]] == ["Lena Volkova"]
    assert len(fake_llm.embed_calls) == embed_calls_before


async def test_connected_semantic_type(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    seeded: list[str],
    fake_llm: FakeLLMClient,
) -> None:
    mark_id = await character_id(client, auth_headers, "Mark Petrov")

    body = await get_json(client, f"/relations/connected/{mark_id}", auth_headers, type="друг")

    assert body["requested_type"] == "друг"
    assert body["resolved_type"] == "FRIEND_OF"
    assert [item["character"]["name"] for item in body["items"]] == ["Lena Volkova"]
    assert fake_llm.embed_calls[-1] == ["друг"]


async def test_connected_unresolvable_type_returns_empty(
    client: httpx.AsyncClient, auth_headers: dict[str, str], seeded: list[str]
) -> None:
    mark_id = await character_id(client, auth_headers, "Mark Petrov")

    response = await client.get(
        f"/relations/connected/{mark_id}", params={"type": "spaceship"}, headers=auth_headers
    )

    assert response.status_code == 200
    body = response.json()
    assert "resolved_type" in body
    assert body["resolved_type"] is None
    assert body["items"] == []


async def test_connected_enemy_is_not_friend(
    client: httpx.AsyncClient, auth_headers: dict[str, str], seeded: list[str]
) -> None:
    assert sum(a * b for a, b in zip(HALF_FRIEND_VECTOR, FRIEND_VECTOR, strict=True)) == (
        pytest.approx(0.5)
    )
    mark_id = await character_id(client, auth_headers, "Mark Petrov")

    body = await get_json(client, f"/relations/connected/{mark_id}", auth_headers, type="enemy")

    assert body["resolved_type"] is None
    assert body["items"] == []


async def test_connected_unknown_character_404(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    seeded: list[str],
    fake_llm: FakeLLMClient,
) -> None:
    embed_calls_before = len(fake_llm.embed_calls)

    response = await client.get(
        f"/relations/connected/{uuid.uuid4()}", params={"type": "друг"}, headers=auth_headers
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert len(fake_llm.embed_calls) == embed_calls_before


@pytest.mark.parametrize("params", [{}, {"type": ""}, {"type": "   "}])
async def test_connected_requires_non_empty_type(
    client: httpx.AsyncClient, auth_headers: dict[str, str], params: dict[str, str]
) -> None:
    response = await client.get("/relations/connected/some-id", params=params, headers=auth_headers)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
