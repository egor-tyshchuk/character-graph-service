from dataclasses import dataclass

from arq.connections import ArqRedis, RedisSettings, create_pool
from neo4j import AsyncDriver, AsyncGraphDatabase
from redis.asyncio import Redis

from app.auth.tokens import TokenService
from app.config import Settings
from app.graph.repository import GraphRepository
from app.graph.schema import apply_schema
from app.llm.client import LLMClient, OpenAILLMClient, UnconfiguredLLMClient
from app.pipeline.process import PipelineDeps
from app.queue import ArqJobQueue, InlineQueue, JobQueue
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
    queue: JobQueue
    arq_pool: ArqRedis | None

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


async def _default_queue(
    settings: Settings, deps: PipelineDeps
) -> tuple[JobQueue, ArqRedis | None]:
    if settings.job_queue == "inline":
        return InlineQueue(deps), None
    pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return ArqJobQueue(pool), pool


async def build_container(
    settings: Settings, *, llm: LLMClient | None = None, queue: JobQueue | None = None
) -> Container:
    resolved_llm = llm if llm is not None else _default_llm(settings)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    redis: Redis = Redis.from_url(settings.redis_url)
    repo = GraphRepository(driver)
    blobs = BlobStore(settings.upload_dir)
    deps = PipelineDeps(repo=repo, llm=resolved_llm, blobs=blobs, redis=redis, settings=settings)
    try:
        await apply_schema(driver, settings.embedding_dim)
        resolved_queue, arq_pool = (
            (queue, None) if queue is not None else await _default_queue(settings, deps)
        )
    except BaseException:
        await redis.aclose()
        await driver.close()
        raise
    return Container(
        settings=settings,
        driver=driver,
        redis=redis,
        repo=repo,
        tokens=TokenService(settings, redis),
        llm=resolved_llm,
        blobs=blobs,
        queue=resolved_queue,
        arq_pool=arq_pool,
    )


async def close_container(c: Container) -> None:
    if c.arq_pool is not None:
        await c.arq_pool.aclose()
    await c.redis.aclose()
    await c.driver.close()
