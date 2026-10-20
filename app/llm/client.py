import asyncio
import base64
from collections.abc import Sequence
from typing import Protocol

import httpx2
import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.llm.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    render_extraction_user,
    render_judge,
    render_vision,
)
from app.llm.schemas import CharacterProfile, DedupVerdict, ExtractionResult, ImageDescription

STRUCTURED_OUTPUT_ATTEMPTS = 2


class LLMError(Exception):
    pass


class LLMNotConfiguredError(LLMError):
    pass


class LLMClient(Protocol):
    async def extract(
        self, chunk: str, known_relation_types: Sequence[str]
    ) -> ExtractionResult: ...

    async def describe_image(self, image_png: bytes, page_context: str) -> str: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def judge_same_character(
        self, candidate: CharacterProfile, existing: CharacterProfile
    ) -> DedupVerdict: ...


class OpenAILLMClient:
    def __init__(
        self, settings: Settings, *, http_client: httpx2.AsyncClient | None = None
    ) -> None:
        if settings.openai_api_key is None:
            raise LLMNotConfiguredError("OPENAI_API_KEY is not set")
        self._settings = settings
        self._client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            http_client=http_client,
        )
        self._semaphore = asyncio.Semaphore(settings.llm_max_concurrency)

    async def extract(self, chunk: str, known_relation_types: Sequence[str]) -> ExtractionResult:
        return await self._parse(
            self._settings.llm_model,
            [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": render_extraction_user(chunk, known_relation_types)},
            ],
            ExtractionResult,
        )

    async def describe_image(self, image_png: bytes, page_context: str) -> str:
        data_url = f"data:image/png;base64,{base64.b64encode(image_png).decode('ascii')}"
        result = await self._parse(
            self._settings.vision_model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": render_vision(page_context)},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}},
                    ],
                }
            ],
            ImageDescription,
        )
        return result.description

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        async with self._semaphore:
            try:
                response = await self._client.embeddings.create(
                    model=self._settings.embedding_model,
                    input=list(texts),
                    dimensions=self._settings.embedding_dim,
                )
            except openai.OpenAIError as exc:
                raise LLMError(f"embedding request failed: {exc}") from exc
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    async def judge_same_character(
        self, candidate: CharacterProfile, existing: CharacterProfile
    ) -> DedupVerdict:
        return await self._parse(
            self._settings.llm_model,
            [{"role": "user", "content": render_judge(candidate, existing)}],
            DedupVerdict,
        )

    async def _parse[T: BaseModel](
        self, model: str, messages: list[ChatCompletionMessageParam], response_format: type[T]
    ) -> T:
        failure = "no attempts made"
        for _ in range(STRUCTURED_OUTPUT_ATTEMPTS):
            async with self._semaphore:
                try:
                    completion = await self._client.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        temperature=0,
                    )
                except ValidationError as exc:
                    failure = f"invalid structured output: {exc}"
                    continue
                except openai.OpenAIError as exc:
                    raise LLMError(f"{response_format.__name__} request failed: {exc}") from exc
            message = completion.choices[0].message
            if message.refusal:
                failure = f"model refused: {message.refusal}"
            elif message.parsed is None:
                failure = "model returned no parsed output"
            else:
                return message.parsed
        raise LLMError(f"{response_format.__name__} structured output failed: {failure}")


class UnconfiguredLLMClient:
    async def extract(self, chunk: str, known_relation_types: Sequence[str]) -> ExtractionResult:
        raise LLMNotConfiguredError("OPENAI_API_KEY is not set")

    async def describe_image(self, image_png: bytes, page_context: str) -> str:
        raise LLMNotConfiguredError("OPENAI_API_KEY is not set")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise LLMNotConfiguredError("OPENAI_API_KEY is not set")

    async def judge_same_character(
        self, candidate: CharacterProfile, existing: CharacterProfile
    ) -> DedupVerdict:
        raise LLMNotConfiguredError("OPENAI_API_KEY is not set")
