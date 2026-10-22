import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.domain.models import CharacterProfile
from app.llm.client import (
    LLM_NOT_CONFIGURED,
    LLMError,
    LLMNotConfiguredError,
    OpenAILLMClient,
    UnconfiguredLLMClient,
)
from app.llm.prompts import EXTRACTION_SYSTEM_PROMPT
from app.llm.schemas import ExtractedCharacter, ExtractedRelation, ExtractionResult

Handler = Callable[[httpx2.Request], httpx2.Response]


def _settings(**overrides: Any) -> Settings:
    return Settings(
        jwt_secret=SecretStr("test-secret-test-secret-test-secret"),
        openai_api_key=SecretStr("sk-test"),
        openai_base_url="http://llm.test/v1",
        **{"embedding_dim": 8, "llm_max_retries": 0, **overrides},
    )


def _client(settings: Settings, handler: Handler) -> OpenAILLMClient:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return OpenAILLMClient(settings, http_client=http_client)


def _chat_completion(content: str | None, *, refusal: str | None = None) -> httpx2.Response:
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
                    "message": {"role": "assistant", "content": content, "refusal": refusal},
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


VERDICT = '{"same": true, "reason": "same lighthouse keeper"}'
PROFILE = CharacterProfile(name="Anna", aliases=[], description="A lighthouse keeper.")


def _scripted(*responses: httpx2.Response) -> tuple[Handler, list[httpx2.Request]]:
    pending = list(responses)
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return pending.pop(0)

    return handler, requests


@pytest.mark.parametrize(
    "unusable",
    [_chat_completion(None, refusal="I can't help with that."), _chat_completion(None)],
    ids=["refusal", "no-parsed-output"],
)
async def test_unusable_structured_output_is_retried_once(unusable: httpx2.Response) -> None:
    handler, requests = _scripted(unusable, _chat_completion(VERDICT))

    verdict = await _client(_settings(), handler).judge_same_character(PROFILE, PROFILE)

    assert verdict.same is True
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("unusable", "reason"),
    [
        (lambda: _chat_completion(None, refusal="I can't help with that."), "model refused"),
        (lambda: _chat_completion(None), "model returned no parsed output"),
    ],
    ids=["refusal", "no-parsed-output"],
)
async def test_structured_output_failing_twice_raises_llm_error(
    unusable: Callable[[], httpx2.Response], reason: str
) -> None:
    handler, requests = _scripted(unusable(), unusable())

    with pytest.raises(LLMError, match=f"DedupVerdict structured output failed: {reason}"):
        await _client(_settings(), handler).judge_same_character(PROFILE, PROFILE)

    assert len(requests) == 2


@pytest.mark.parametrize("status", [400, 401, 429, 500])
async def test_openai_errors_become_llm_errors(status: int) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": {"message": "upstream says no"}})

    client = _client(_settings(), handler)

    with pytest.raises(LLMError, match="ExtractionResult request failed"):
        await client.extract("text", [])
    with pytest.raises(LLMError, match="embedding request failed"):
        await client.embed(["text"])


async def test_judge_parses_verdict() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _chat_completion('{"same": false, "reason": "different families"}')

    profile = CharacterProfile(name="Ivan", aliases=[], description="A pilot.")
    verdict = await _client(_settings(), handler).judge_same_character(profile, profile)

    assert verdict.same is False
    assert verdict.reason == "different families"


def _recording(content: str) -> tuple[Handler, list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return _chat_completion(content)

    return handler, bodies


@pytest.mark.parametrize(
    ("judge_model", "expected"),
    [("gpt-judge", "gpt-judge"), (None, "gpt-lite"), ("", "gpt-lite"), ("   ", "gpt-lite")],
    ids=["set", "unset", "empty", "whitespace"],
)
async def test_judge_uses_judge_model_or_falls_back_to_llm_model(
    judge_model: str | None, expected: str
) -> None:
    handler, bodies = _recording(VERDICT)
    settings = _settings(llm_model="gpt-lite", judge_model=judge_model)

    await _client(settings, handler).judge_same_character(PROFILE, PROFILE)

    assert bodies[0]["model"] == expected


async def test_judge_model_leaves_extract_and_vision_models_alone() -> None:
    settings = _settings(llm_model="gpt-lite", vision_model="gpt-vision", judge_model="gpt-judge")
    extract_handler, extract_bodies = _recording(
        ExtractionResult(characters=[], relations=[]).model_dump_json()
    )
    vision_handler, vision_bodies = _recording('{"description": "Two people."}')

    await _client(settings, extract_handler).extract("text", [])
    await _client(settings, vision_handler).describe_image(b"\x89PNG", "caption")

    assert extract_bodies[0]["model"] == "gpt-lite"
    assert vision_bodies[0]["model"] == "gpt-vision"


def _unavailable_recording() -> tuple[Handler, list[tuple[float, str]]]:
    attempts: list[tuple[float, str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        attempts.append(
            (request.extensions["timeout"]["read"], request.headers["x-stainless-retry-count"])
        )
        return httpx2.Response(
            503, headers={"retry-after-ms": "1"}, json={"error": {"message": "unavailable"}}
        )

    return handler, attempts


async def test_judge_requests_use_the_judge_timeout_and_retry_budget() -> None:
    settings = _settings(
        llm_timeout_seconds=60.0, llm_max_retries=0, judge_timeout_seconds=7.5, judge_max_retries=2
    )
    handler, attempts = _unavailable_recording()
    client = _client(settings, handler)

    with pytest.raises(LLMError):
        await client.judge_same_character(PROFILE, PROFILE)

    assert attempts == [(7.5, "0"), (7.5, "1"), (7.5, "2")]


async def test_extract_and_vision_keep_the_llm_timeout_and_retry_budget() -> None:
    settings = _settings(
        llm_timeout_seconds=45.0, llm_max_retries=1, judge_timeout_seconds=7.5, judge_max_retries=0
    )
    handler, attempts = _unavailable_recording()
    client = _client(settings, handler)

    with pytest.raises(LLMError):
        await client.extract("text", [])
    with pytest.raises(LLMError):
        await client.describe_image(b"\x89PNG", "caption")

    assert attempts == [(45.0, "0"), (45.0, "1"), (45.0, "0"), (45.0, "1")]


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


def _embeddings_response(data: list[dict[str, Any]]) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "object": "list",
            "model": "text-embedding-test",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
            "data": [{"object": "embedding", **item} for item in data],
        },
    )


async def test_embed_keeps_response_order_when_an_index_is_null() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _embeddings_response(
            [
                {"index": None, "embedding": [1.0, 0.0]},
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 2, "embedding": [0.5, 0.5]},
            ]
        )

    vectors = await _client(_settings(), handler).embed(["first", "second", "third"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]


async def test_embed_rejects_a_response_of_the_wrong_length() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _embeddings_response([{"index": 0, "embedding": [1.0, 0.0]}])

    with pytest.raises(LLMError, match="1 embeddings for 2 inputs"):
        await _client(_settings(), handler).embed(["first", "second"])


async def test_embed_empty_input_makes_no_request() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no request expected")

    assert await _client(_settings(), handler).embed([]) == []


async def test_unconfigured_client_raises_not_configured() -> None:
    with pytest.raises(LLMNotConfiguredError, match=LLM_NOT_CONFIGURED):
        await UnconfiguredLLMClient().embed(["text"])


def test_openai_client_requires_an_api_key() -> None:
    with pytest.raises(LLMNotConfiguredError, match=LLM_NOT_CONFIGURED):
        OpenAILLMClient(_settings().model_copy(update={"openai_api_key": None}))


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
