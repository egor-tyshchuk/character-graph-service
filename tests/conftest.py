import hashlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from neo4j import AsyncDriver, AsyncGraphDatabase
from pydantic import SecretStr
from redis.asyncio import Redis
from testcontainers.neo4j import Neo4jContainer
from testcontainers.redis import RedisContainer

from app.config import Settings
from app.graph.repository import GraphRepository
from app.graph.schema import apply_schema
from app.main import create_app

NEO4J_PASSWORD = "testpassword"
TEST_APP_ID = "test-app"
TEST_APP_TOKEN = "test-app-token"


@pytest.fixture(scope="session")
def neo4j_container() -> Iterator[Neo4jContainer]:
    with Neo4jContainer("neo4j:5.26-community", password=NEO4J_PASSWORD) as container:
        yield container


@pytest.fixture(scope="session")
def redis_container() -> Iterator[RedisContainer]:
    with RedisContainer("redis:7-alpine") as container:
        yield container


@pytest.fixture
def fake_llm() -> None:
    return None


@pytest.fixture
def settings(
    tmp_path: Path, neo4j_container: Neo4jContainer, redis_container: RedisContainer
) -> Settings:
    redis_host = redis_container.get_container_host_ip()
    redis_port = redis_container.get_exposed_port(6379)
    return Settings(
        jwt_secret=SecretStr("test-secret-test-secret-test-secret"),
        app_tokens={TEST_APP_ID: hashlib.sha256(TEST_APP_TOKEN.encode()).hexdigest()},
        neo4j_uri=neo4j_container.get_connection_url(),
        neo4j_user="neo4j",
        neo4j_password=SecretStr(NEO4J_PASSWORD),
        redis_url=f"redis://{redis_host}:{redis_port}/0",
        openai_api_key=None,
        embedding_dim=8,
        job_queue="inline",
        upload_dir=tmp_path / "uploads",
        log_json=False,
    )


@pytest_asyncio.fixture(autouse=False)
async def clean_stores(settings: Settings) -> None:
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    try:
        await driver.execute_query("MATCH (n) DETACH DELETE n")
    finally:
        await driver.close()
    redis = Redis.from_url(settings.redis_url)
    try:
        await redis.flushdb()
    finally:
        await redis.aclose()


@pytest_asyncio.fixture
async def app(settings: Settings, fake_llm: None, clean_stores: None) -> AsyncIterator[FastAPI]:
    application = create_app(settings, llm=fake_llm)
    async with LifespanManager(application):
        yield application


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def driver(settings: Settings) -> AsyncIterator[AsyncDriver]:
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    try:
        await neo4j_driver.execute_query("MATCH (n) DETACH DELETE n")
        await apply_schema(neo4j_driver, settings.embedding_dim)
        yield neo4j_driver
    finally:
        await neo4j_driver.close()


@pytest_asyncio.fixture
async def repo(driver: AsyncDriver) -> GraphRepository:
    return GraphRepository(driver)
