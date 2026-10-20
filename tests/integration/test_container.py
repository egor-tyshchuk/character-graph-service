from pydantic import SecretStr

from app.config import Settings
from app.container import build_container, close_container
from app.llm.client import OpenAILLMClient, UnconfiguredLLMClient
from app.queue import ArqJobQueue, InlineQueue
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


class NullQueue:
    async def enqueue_article(self, article_id: str) -> None:
        return None


async def test_inline_job_queue_has_no_arq_pool(settings: Settings) -> None:
    container = await build_container(settings)
    try:
        assert isinstance(container.queue, InlineQueue)
        assert container.arq_pool is None
    finally:
        await close_container(container)


async def test_arq_job_queue_uses_redis_pool(settings: Settings) -> None:
    container = await build_container(settings.model_copy(update={"job_queue": "arq"}))
    try:
        assert isinstance(container.queue, ArqJobQueue)
        assert container.arq_pool is not None
        assert await container.arq_pool.ping()
    finally:
        await close_container(container)


async def test_container_prefers_injected_queue(settings: Settings) -> None:
    queue = NullQueue()
    container = await build_container(settings.model_copy(update={"job_queue": "arq"}), queue=queue)
    try:
        assert container.queue is queue
        assert container.arq_pool is None
    finally:
        await close_container(container)
