from dataclasses import dataclass

from neo4j import AsyncDriver, AsyncGraphDatabase
from redis.asyncio import Redis

from app.config import Settings
from app.graph.repository import GraphRepository
from app.graph.schema import apply_schema


@dataclass
class Container:
    settings: Settings
    driver: AsyncDriver
    redis: Redis
    repo: GraphRepository


async def build_container(settings: Settings) -> Container:
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
    return Container(settings=settings, driver=driver, redis=redis, repo=GraphRepository(driver))


async def close_container(c: Container) -> None:
    await c.redis.aclose()
    await c.driver.close()
