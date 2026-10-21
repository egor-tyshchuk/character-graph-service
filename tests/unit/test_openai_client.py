import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.llm.client import LLMError, LLMNotConfiguredError, OpenAILLMClient, UnconfiguredLLMClient
from app.llm.prompts import EXTRACTION_SYSTEM_PROMPT
from app.llm.schemas import (
    CharacterProfile,
    ExtractedCharacter,
    ExtractedRelation,
    ExtractionResult,
)

Handler = Callable[[httpx2.Request], httpx2.Response]


def _settings() -> Settings:
    return Settings(
        jwt_secret=SecretStr("test-secret-test-secret-test-secret"),
        openai_api_key=SecretStr("sk-test"),
        openai_base_url="http://llm.test/v1",
        embedding_dim=8,
        llm_max_retries=0,
    )


def _client(settings: Settings, handler: Handler) -> OpenAILLMClient:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return OpenAILLMClient(settings, http_client=http_client)


def _chat_completion(content: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content, "refusal": None},
                }
            ],
        },
    )


async def test_extract_sends_strict_json_schema_and_parses() -> None:
    expected = ExtractionResult(
        characters=[
            ExtractedCharacter(name="Anna Petrova", aliases=["Anna"], description="A doctor."),
            ExtractedCharacter(name="Mark Petrov", aliases=[], description="Her brother."),
        ],
        relations=[
            ExtractedRelation(
                source="Anna Petrova",
                target="Mark Petrov",
                type="SIBLING_OF",
                description="Siblings.",
                evidence="Anna and her brother Mark",
            )
        ],
    )
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return _chat_completion(expected.model_dump_json())

    result = await _client(_settings(), handler).extract("Anna and her brother Mark", ["FRIEND_OF"])

    assert result == expected
    body = bodies[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["temperature"] == 0
    assert body["messages"][0] == {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}
    assert "FRIEND_OF" in body["messages"][1]["content"]


async def test_extract_retries_once_on_invalid_json_then_raises() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return _chat_completion("{not json")

    with pytest.raises(LLMError):
        await _client(_settings(), handler).extract("text", [])

    assert calls == 2


async def test_judge_parses_verdict() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _chat_completion('{"same": false, "reason": "different families"}')

    profile = CharacterProfile(name="Ivan", aliases=[], description="A pilot.")
    verdict = await _client(_settings(), handler).judge_same_character(profile, profile)

    assert verdict.same is False
    assert verdict.reason == "different families"


async def test_describe_image_sends_data_url() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return _chat_completion('{"description": "Two people."}')

    settings = _settings()
    description = await _client(settings, handler).describe_image(b"\x89PNG", "caption")

    assert description == "Two people."
    body = bodies[0]
    assert body["model"] == settings.vision_model
    parts = body["messages"][0]["content"]
    image_part = next(part for part in parts if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_part["image_url"]["detail"] == "auto"


async def test_embed_preserves_order_and_requests_dimensions() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-test",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                ],
            },
        )

    settings = _settings()
    vectors = await _client(settings, handler).embed(["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert bodies[0]["dimensions"] == settings.embedding_dim
    assert bodies[0]["input"] == ["first", "second"]
    assert bodies[0]["model"] == settings.embedding_model


async def test_embed_empty_input_makes_no_request() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no request expected")

    assert await _client(_settings(), handler).embed([]) == []


async def test_unconfigured_client_raises_not_configured() -> None:
    with pytest.raises(LLMNotConfiguredError, match="OPENAI_API_KEY is not set"):
        await UnconfiguredLLMClient().embed(["text"])


async def test_blank_base_url_env_falls_back_to_official_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    urls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        urls.append(str(request.url))
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-test",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
            },
        )

    settings = _settings().model_copy(update={"openai_base_url": None})
    await _client(settings, handler).embed(["text"])

    assert urls == ["https://api.openai.com/v1/embeddings"]
