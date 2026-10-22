import asyncio
import os
import re
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from dotenv import dotenv_values
from pydantic import SecretStr
from testcontainers.redis import RedisContainer

from app.api.uploads import file_article
from app.config import Settings
from app.container import Container, build_container, close_container
from app.domain.models import (
    Article,
    ArticleFormat,
    ArticleStats,
    ArticleStatus,
    Character,
)
from app.llm.client import OpenAILLMClient
from app.pipeline.process import process_article
from app.pipeline.relation_types import resolve_relation_type
from tests.conftest import (
    NEO4J_PASSWORD,
    NEO4J_USER,
    REDIS_IMAGE,
    Neo4jService,
    neo4j_container,
    running,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
EXAMPLES = REPO_ROOT / "examples"
RELATION_TYPE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
ALL_CHARACTERS_LIMIT = 1000
E2E_JWT_SECRET = SecretStr("e2e-secret-e2e-secret-e2e-secret")
LLM_CONFIG_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LLM_MODEL",
        "VISION_MODEL",
        "JUDGE_MODEL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIM",
        "LLM_MAX_CONCURRENCY",
        "LLM_MAX_RETRIES",
        "LLM_TIMEOUT_SECONDS",
        "DEDUP_AUTO_MERGE_THRESHOLD",
        "DEDUP_JUDGE_THRESHOLD",
        "RELATION_TYPE_QUERY_THRESHOLD",
        "DEDUP_CANDIDATES",
        "DEDUP_MAX_JUDGE_CALLS_PER_ARTICLE",
    }
)


def load_llm_config() -> tuple[dict[str, str], SecretStr | None]:
    if os.environ.get("RUN_E2E") != "1":
        return {}, None
    values = {
        key.lower(): value
        for key, value in dotenv_values(ENV_FILE).items()
        if value is not None and key.upper() in LLM_CONFIG_KEYS
    }
    api_key = values.pop("openai_api_key", "").strip()
    return values, SecretStr(api_key) if api_key else None


LLM_CONFIG, API_KEY = load_llm_config()

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        API_KEY is None,
        reason="set RUN_E2E=1 and OPENAI_API_KEY in .env to run against the real LLM",
    ),
]


@dataclass(frozen=True)
class IngestedSamples:
    en: Article
    ru: Article
    pdf: Article


@pytest.fixture(scope="module")
def e2e_neo4j() -> Iterator[Neo4jService]:
    with running(neo4j_container()) as container:
        yield Neo4jService(container.get_connection_url(), NEO4J_USER, NEO4J_PASSWORD)


@pytest.fixture(scope="module")
def e2e_redis_url() -> Iterator[str]:
    with running(RedisContainer(REDIS_IMAGE)) as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


@pytest.fixture(scope="module")
def e2e_settings(
    e2e_neo4j: Neo4jService, e2e_redis_url: str, tmp_path_factory: pytest.TempPathFactory
) -> Settings:
    values: dict[str, Any] = {
        **LLM_CONFIG,
        "openai_api_key": API_KEY,
        "jwt_secret": E2E_JWT_SECRET,
        "app_tokens": {},
        "neo4j_uri": e2e_neo4j.uri,
        "neo4j_user": e2e_neo4j.user,
        "neo4j_password": SecretStr(e2e_neo4j.password),
        "redis_url": e2e_redis_url,
        "job_queue": "inline",
        "upload_dir": tmp_path_factory.mktemp("e2e-uploads"),
        "log_json": False,
    }
    with pytest.MonkeyPatch.context() as patch:
        for name in list(os.environ):
            if name.lower() in Settings.model_fields:
                patch.delenv(name)
        return Settings(**values)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def e2e_container(e2e_settings: Settings) -> AsyncIterator[Container]:
    container = await build_container(e2e_settings)
    try:
        assert isinstance(container.llm, OpenAILLMClient)
        yield container
    finally:
        await close_container(container)


async def ingest(container: Container, path: Path, article_format: ArticleFormat) -> Article:
    data = await asyncio.to_thread(path.read_bytes)
    upload = file_article(data, path.name, None)
    assert upload.format is article_format
    now = datetime.now(UTC)
    article = Article(
        id=str(uuid.uuid4()),
        title=await upload.resolve_title(),
        format=upload.format,
        content_hash=upload.content_hash,
        status=ArticleStatus.PENDING,
        created_at=now,
        updated_at=now,
    )
    await container.repo.create_article(article)
    await container.blobs.save(article.id, data)
    await process_article(article.id, container.pipeline_deps)
    processed = await container.repo.get_article(article.id)
    assert processed is not None
    return processed


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def samples(e2e_container: Container) -> IngestedSamples:
    en = await ingest(e2e_container, EXAMPLES / "sample_en.md", ArticleFormat.MARKDOWN)
    ru = await ingest(e2e_container, EXAMPLES / "sample_ru.md", ArticleFormat.MARKDOWN)
    pdf = await ingest(e2e_container, EXAMPLES / "sample.pdf", ArticleFormat.PDF)
    return IngestedSamples(en=en, ru=ru, pdf=pdf)


def done_stats(article: Article) -> ArticleStats:
    assert article.status is ArticleStatus.DONE, (
        f"{article.title!r} is {article.status}: {error_type(article)}"
    )
    assert article.stats is not None
    return article.stats


def processed_stats(article: Article) -> ArticleStats:
    stats = done_stats(article)
    assert stats.unjudged_characters == 0, (
        f"judge unavailable or capped (quota or DEDUP_MAX_JUDGE_CALLS_PER_ARTICLE): "
        f"{stats.unjudged_characters} characters of "
        f"{article.title!r} were stored without a judge verdict"
    )
    return stats


def error_type(article: Article) -> str:
    return (article.error or "").partition(":")[0]


def surfaces(character: Character) -> list[str]:
    return [character.name, *character.aliases]


def named(characters: Sequence[Character], *prefixes: str) -> list[Character]:
    return [
        character
        for character in characters
        if any(surface.startswith(prefixes) for surface in surfaces(character))
    ]


async def all_characters(container: Container) -> list[Character]:
    characters, _ = await container.repo.list_characters(limit=ALL_CHARACTERS_LIMIT, offset=0)
    return characters


async def relation_type_names(container: Container) -> list[str]:
    return [relation_type.name for relation_type in await container.repo.list_relation_types()]


async def ranked_relation_types(
    container: Container, query: str, k: int
) -> list[tuple[str, float]]:
    [embedding] = await container.llm.embed([query])
    return await container.repo.find_similar_relation_types(embedding, k)


async def test_markdown_extraction_real(e2e_container: Container, samples: IngestedSamples) -> None:
    stats = processed_stats(samples.en)
    assert stats.characters >= 3
    assert stats.relations >= 2
    characters = await e2e_container.repo.list_characters_by_article(samples.en.id)
    assert named(characters, "Anna"), [c.name for c in characters]
    types = await relation_type_names(e2e_container)
    assert types
    assert all(RELATION_TYPE_PATTERN.match(name) for name in types), types


@pytest.mark.parametrize(("latin", "cyrillic"), [("Anna", "Анна"), ("Oleg", "Олег")])
async def test_cross_language_dedup_real(
    e2e_container: Container, samples: IngestedSamples, latin: str, cyrillic: str
) -> None:
    processed_stats(samples.en)
    processed_stats(samples.ru)
    matches = named(await all_characters(e2e_container), latin, cyrillic)
    assert len(matches) == 1, [surfaces(c) for c in matches]
    [character] = matches
    assert any(surface.startswith(latin) for surface in surfaces(character))
    assert any(surface.startswith(cyrillic) for surface in surfaces(character))
    assert character.article_count >= 2


async def test_pdf_image_relation_real(e2e_container: Container, samples: IngestedSamples) -> None:
    stats = done_stats(samples.pdf)
    assert stats.images_described >= 1, stats
    characters = await all_characters(e2e_container)
    marks = named(characters, "Mark", "Марк")
    assert marks, ([c.name for c in characters], stats)
    types = await relation_type_names(e2e_container)
    links = [
        (relation_type, connection)
        for mark in marks
        for relation_type in types
        for connection in await e2e_container.repo.find_connected(mark.id, relation_type)
        if named([connection.character], "Lena", "Лена")
    ]
    found = [(relation_type, connection.direction.value) for relation_type, connection in links]
    assert any(
        evidence.article_id == samples.pdf.id
        for _, connection in links
        for evidence in connection.evidence
    ), (
        "no Mark-Lena relation with PDF evidence; "
        f"Mark nodes: {[surfaces(mark) for mark in marks]}; Mark-Lena relations: {found}; "
        f"unjudged characters in the PDF: {stats.unjudged_characters}"
    )


async def test_pdf_vector_diagram_relation_real(
    e2e_container: Container, samples: IngestedSamples
) -> None:
    stats = done_stats(samples.pdf)
    assert stats.images_described >= 2, stats
    characters = await all_characters(e2e_container)
    igors = named(characters, "Igor", "Игорь")
    assert igors, ([c.name for c in characters], stats)
    types = await relation_type_names(e2e_container)
    links = [
        (igor, relation_type, connection)
        for igor in igors
        for relation_type in types
        for connection in await e2e_container.repo.find_connected(igor.id, relation_type)
        if named([connection.character], "Lena", "Лена")
    ]
    found = [
        (igor.name, relation_type, connection.direction.value, connection.character.name)
        for igor, relation_type, connection in links
    ]
    assert any(
        evidence.article_id == samples.pdf.id
        for _, _, connection in links
        for evidence in connection.evidence
    ), (
        "no Igor-Lena relation with PDF evidence; "
        f"Igor nodes: {[surfaces(igor) for igor in igors]}; Igor-Lena relations: {found}; "
        f"unjudged characters in the PDF: {stats.unjudged_characters}"
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [("sibling", "SIBLING_OF"), ("brother", "SIBLING_OF"), ("друг", "FRIEND_OF")],
)
async def test_relation_type_query_threshold_resolves_paraphrase_real(
    e2e_container: Container, samples: IngestedSamples, query: str, expected: str
) -> None:
    types = await relation_type_names(e2e_container)
    assert expected in types, types
    threshold = e2e_container.settings.relation_type_query_threshold
    top = await ranked_relation_types(e2e_container, query, 1)
    assert await resolve_relation_type(query, e2e_container.pipeline_deps) == expected, (
        f"{query!r}: top match {top}, threshold {threshold}"
    )


async def test_relation_type_query_threshold_keeps_opposite_apart_real(
    e2e_container: Container, samples: IngestedSamples
) -> None:
    types = await relation_type_names(e2e_container)
    assert "FRIEND_OF" in types, types
    threshold = e2e_container.settings.relation_type_query_threshold
    ranked = dict(await ranked_relation_types(e2e_container, "enemy", len(types)))
    assert "FRIEND_OF" in ranked, ranked
    assert ranked["FRIEND_OF"] < threshold, (
        f"'enemy' ~ FRIEND_OF similarity {ranked['FRIEND_OF']:.4f} >= threshold {threshold}"
    )
    assert await resolve_relation_type("enemy", e2e_container.pipeline_deps) != "FRIEND_OF"
