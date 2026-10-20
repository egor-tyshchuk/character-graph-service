from pydantic import SecretStr

from app.config import Settings
from app.container import build_container, close_container
from app.llm.client import OpenAILLMClient, UnconfiguredLLMClient
from tests.fakes import FakeLLMClient


async def test_container_without_api_key_uses_unconfigured_llm(settings: Settings) -> None:
    container = await build_container(settings)
    try:
        assert isinstance(container.llm, UnconfiguredLLMClient)
        deps = container.pipeline_deps
        assert (deps.repo, deps.llm, deps.blobs, deps.redis, deps.settings) == (
            container.repo,
            container.llm,
            container.blobs,
            container.redis,
            settings,
        )
    finally:
        await close_container(container)


async def test_container_with_api_key_uses_openai_llm(settings: Settings) -> None:
    configured = settings.model_copy(update={"openai_api_key": SecretStr("sk-test")})
    container = await build_container(configured)
    try:
        assert isinstance(container.llm, OpenAILLMClient)
    finally:
        await close_container(container)


async def test_container_prefers_injected_llm(settings: Settings) -> None:
    fake = FakeLLMClient()
    container = await build_container(settings, llm=fake)
    try:
        assert container.llm is fake
    finally:
        await close_container(container)
