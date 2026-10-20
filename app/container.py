from dataclasses import dataclass

from neo4j import AsyncDriver, AsyncGraphDatabase
from redis.asyncio import Redis

from app.auth.tokens import TokenService
from app.config import Settings
from app.graph.repository import GraphRepository
from app.graph.schema import apply_schema
from app.llm.client import LLMClient, OpenAILLMClient, UnconfiguredLLMClient
from app.pipeline.process import PipelineDeps
from app.storage import BlobStore


@dataclass
class Container:
    settings: Settings
    driver: AsyncDriver
    redis: Redis
    repo: GraphRepository
    tokens: TokenService
    llm: LLMClient
    blobs: BlobStore

    @property
    def pipeline_deps(self) -> PipelineDeps:
        return PipelineDeps(
            repo=self.repo,
            llm=self.llm,
            blobs=self.blobs,
            redis=self.redis,
            settings=self.settings,
        )


def _default_llm(settings: Settings) -> LLMClient:
    if settings.openai_api_key is None:
        return UnconfiguredLLMClient()
    return OpenAILLMClient(settings)


async def build_container(settings: Settings, *, llm: LLMClient | None = None) -> Container:
    resolved_llm = llm if llm is not None else _default_llm(settings)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    redis: Redis = Redis.from_url(settings.redis_url)
    try:
        await apply_schema(driver, settings.embedding_dim)
    except BaseException:
        await redis.aclose()
        await driver.close()
        raise
    return Container(
        settings=settings,
        driver=driver,
        redis=redis,
        repo=GraphRepository(driver),
        tokens=TokenService(settings, redis),
        llm=resolved_llm,
        blobs=BlobStore(settings.upload_dir),
    )


async def close_container(c: Container) -> None:
    await c.redis.aclose()
    await c.driver.close()
