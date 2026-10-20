import hashlib
import os
import shutil
import subprocess
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from neo4j import AsyncDriver, AsyncGraphDatabase
from pydantic import SecretStr
from redis.asyncio import Redis
from testcontainers.core.container import DockerContainer
from testcontainers.neo4j import Neo4jContainer
from testcontainers.redis import RedisContainer

from app.config import Settings
from app.graph.repository import GraphRepository
from app.graph.schema import apply_schema
from app.main import create_app
from tests.fakes import FakeLLMClient

NEO4J_IMAGE = "neo4j:5.26-community"
REDIS_IMAGE = "redis:7-alpine"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"
NEO4J_FAST_START_ENV = {
    "NEO4J_server_memory_heap_initial__size": "256m",
    "NEO4J_server_memory_heap_max__size": "512m",
    "NEO4J_server_memory_pagecache_size": "128m",
    "NEO4J_dbms_usage__report_enabled": "false",
}
EXTERNAL_SERVICE_ENV_VARS = ("TEST_NEO4J_URI", "TEST_REDIS_URL")
ALLOW_WIPE_ENV_VAR = "TEST_ALLOW_WIPE"
EXTERNAL_REDIS_DEFAULT_DB = 15
TEST_APP_ID = "test-app"
TEST_APP_TOKEN = "test-app-token"


@dataclass(frozen=True)
class Neo4jService:
    uri: str
    user: str
    password: str


def pytest_sessionstart(session: pytest.Session) -> None:
    external = [name for name in EXTERNAL_SERVICE_ENV_VARS if os.environ.get(name)]
    if external and os.environ.get(ALLOW_WIPE_ENV_VAR) != "1":
        pytest.exit(
            f"{', '.join(external)} set: the test suite deletes all data in these services. "
            f"Set {ALLOW_WIPE_ENV_VAR}=1 to allow it.",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )


def remove_without_waiting(container: DockerContainer) -> None:
    wrapped = container.get_wrapped_container()
    if wrapped is None or shutil.which("docker") is None:
        container.stop()
        return
    subprocess.run(
        ["sh", "-c", 'docker rm --force --volumes "$1" >/dev/null 2>&1 &', "sh", wrapped.id],
        check=True,
    )


@contextmanager
def running[T: DockerContainer](container: T) -> Iterator[T]:
    try:
        yield container.start()
    finally:
        remove_without_waiting(container)


def with_default_db(redis_url: str, db: int) -> str:
    parts = urlsplit(redis_url)
    if parts.path.strip("/"):
        return redis_url
    return urlunsplit(parts._replace(path=f"/{db}"))


def neo4j_container() -> Neo4jContainer:
    container = Neo4jContainer(NEO4J_IMAGE, password=NEO4J_PASSWORD)
    for name, value in NEO4J_FAST_START_ENV.items():
        container.with_env(name, value)
    return container


@pytest.fixture(scope="session")
def neo4j_service() -> Iterator[Neo4jService]:
    external_uri = os.environ.get("TEST_NEO4J_URI")
    if external_uri:
        yield Neo4jService(
            uri=external_uri,
            user=os.environ.get("TEST_NEO4J_USER", NEO4J_USER),
            password=os.environ.get("TEST_NEO4J_PASSWORD", NEO4J_PASSWORD),
        )
        return
    with running(neo4j_container()) as container:
        yield Neo4jService(container.get_connection_url(), NEO4J_USER, NEO4J_PASSWORD)


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    external_url = os.environ.get("TEST_REDIS_URL")
    if external_url:
        yield with_default_db(external_url, EXTERNAL_REDIS_DEFAULT_DB)
        return
    with running(RedisContainer(REDIS_IMAGE)) as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient(dim=8)


@pytest.fixture
def settings(tmp_path: Path, neo4j_service: Neo4jService, redis_url: str) -> Settings:
    return Settings(
        jwt_secret=SecretStr("test-secret-test-secret-test-secret"),
        app_tokens={TEST_APP_ID: hashlib.sha256(TEST_APP_TOKEN.encode()).hexdigest()},
        neo4j_uri=neo4j_service.uri,
        neo4j_user=neo4j_service.user,
        neo4j_password=SecretStr(neo4j_service.password),
        redis_url=redis_url,
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
async def app(
    settings: Settings, fake_llm: FakeLLMClient, clean_stores: None
) -> AsyncIterator[FastAPI]:
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


@pytest_asyncio.fixture
async def auth_headers(client: httpx.AsyncClient) -> dict[str, str]:
    response = await client.post("/auth/token", json={"app_token": TEST_APP_TOKEN})
    response.raise_for_status()
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
