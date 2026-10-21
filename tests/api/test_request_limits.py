from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

TOO_LARGE_BODY = 64 * 1024 + 1
TOKEN_MAX_CHARS = 4096
TITLE_MAX_CHARS = 300
MERGE_MAX_SOURCES = 50


def _padded_json(field: str, size: int) -> bytes:
    return b'{"' + field.encode() + b'": "' + b"x" * size + b'"}'


async def _chunks(body: bytes, size: int = 8192) -> AsyncIterator[bytes]:
    for start in range(0, len(body), size):
        yield body[start : start + size]


def _assert_too_large(response: httpx.Response) -> None:
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def _assert_invalid(response: httpx.Response, field: str) -> None:
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert field in error["message"]


@pytest.mark.parametrize(
    ("path", "field"), [("/auth/token", "app_token"), ("/auth/refresh", "refresh_token")]
)
async def test_oversized_auth_body_is_rejected_with_413(
    client: httpx.AsyncClient, path: str, field: str
) -> None:
    body = _padded_json(field, TOO_LARGE_BODY)

    declared = await client.post(path, content=body, headers={"Content-Type": "application/json"})
    streamed = await client.post(
        path, content=_chunks(body), headers={"Content-Type": "application/json"}
    )

    _assert_too_large(declared)
    _assert_too_large(streamed)
    assert "content-length" not in streamed.request.headers


@pytest.mark.parametrize(
    ("path", "field"), [("/auth/token", "app_token"), ("/auth/refresh", "refresh_token")]
)
async def test_overlong_auth_token_is_rejected_with_422(
    client: httpx.AsyncClient, path: str, field: str
) -> None:
    response = await client.post(path, json={field: "x" * (TOKEN_MAX_CHARS + 1)})

    _assert_invalid(response, field)


async def test_oversized_merge_body_is_rejected_with_413(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/characters/merge",
        content=_padded_json("target_id", TOO_LARGE_BODY),
        headers={**auth_headers, "Content-Type": "application/json"},
    )

    _assert_too_large(response)


async def test_merge_with_too_many_sources_is_rejected_with_422(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    sources = [f"source-{index}" for index in range(MERGE_MAX_SOURCES + 1)]

    response = await client.post(
        "/characters/merge",
        json={"target_id": "target", "source_ids": sources},
        headers=auth_headers,
    )

    _assert_invalid(response, "source_ids")


async def test_merge_with_the_maximum_number_of_sources_passes_validation(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    sources = [f"source-{index}" for index in range(MERGE_MAX_SOURCES)]

    response = await client.post(
        "/characters/merge",
        json={"target_id": "target", "source_ids": sources},
        headers=auth_headers,
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    "kwargs",
    [
        {"json": {"text": "Anna keeps the lighthouse.", "title": "t" * (TITLE_MAX_CHARS + 1)}},
        {"data": {"text": "Anna keeps the lighthouse.", "title": "t" * (TITLE_MAX_CHARS + 1)}},
        {
            "files": {
                "text": (None, "Anna keeps the lighthouse."),
                "title": (None, "t" * (TITLE_MAX_CHARS + 1)),
            }
        },
    ],
    ids=["json", "urlencoded", "multipart"],
)
async def test_overlong_upload_title_is_rejected_with_422(
    client: httpx.AsyncClient, auth_headers: dict[str, str], kwargs: dict[str, Any]
) -> None:
    response = await client.post("/articles/upload", headers=auth_headers, **kwargs)

    _assert_invalid(response, "title")


async def test_upload_title_at_the_limit_is_accepted(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    title = "t" * TITLE_MAX_CHARS

    response = await client.post(
        "/articles/upload",
        headers=auth_headers,
        data={"text": "Anna keeps the lighthouse.", "title": title},
    )

    assert response.status_code == 202
    assert response.json()["title"] == title


async def test_upload_is_not_bound_by_the_json_body_cap(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    text = "Anna keeps the lighthouse. " * 4000

    response = await client.post("/articles/upload", headers=auth_headers, json={"text": text})

    assert response.status_code == 202
