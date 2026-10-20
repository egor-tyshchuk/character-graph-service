import uuid
from datetime import UTC, datetime

import pytest
from neo4j import AsyncDriver

from app.domain.models import Article, ArticleFormat, ArticleStatus
from app.graph.repository import DuplicateContentError, GraphRepository
from app.graph.schema import CHARACTER_VECTOR_INDEX, RELATION_TYPE_VECTOR_INDEX, apply_schema


def make_article(content_hash: str = "hash-1") -> Article:
    now = datetime.now(UTC).replace(microsecond=0)
    return Article(
        id=str(uuid.uuid4()),
        title="A tale",
        format=ArticleFormat.MARKDOWN,
        content_hash=content_hash,
        status=ArticleStatus.PENDING,
        created_at=now,
        updated_at=now,
    )


async def test_create_and_get_article_roundtrip(repo: GraphRepository) -> None:
    article = make_article()

    created = await repo.create_article(article)

    assert created == article
    assert await repo.get_article(article.id) == article


async def test_get_missing_article_returns_none(repo: GraphRepository) -> None:
    assert await repo.get_article("missing") is None
    assert await repo.get_article_by_hash("missing") is None


async def test_get_article_by_hash(repo: GraphRepository) -> None:
    article = await repo.create_article(make_article("hash-lookup"))

    assert await repo.get_article_by_hash("hash-lookup") == article


async def test_duplicate_content_hash_raises(repo: GraphRepository) -> None:
    first = await repo.create_article(make_article("same-hash"))

    with pytest.raises(DuplicateContentError) as excinfo:
        await repo.create_article(make_article("same-hash"))

    assert excinfo.value.existing_id == first.id


async def test_set_status_failed_then_pending_clears_error(repo: GraphRepository) -> None:
    article = await repo.create_article(make_article())

    await repo.set_article_status(article.id, ArticleStatus.FAILED, error="boom")
    failed = await repo.get_article(article.id)
    assert failed is not None
    assert failed.status is ArticleStatus.FAILED
    assert failed.error == "boom"
    assert failed.updated_at > article.updated_at

    await repo.set_article_status(article.id, ArticleStatus.PENDING)
    pending = await repo.get_article(article.id)
    assert pending is not None
    assert pending.status is ArticleStatus.PENDING
    assert pending.error is None
    assert pending.updated_at >= failed.updated_at


async def test_apply_schema_is_idempotent_and_creates_vector_indexes(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    await apply_schema(driver, 8)
    await apply_schema(driver, 8)

    records, _, _ = await driver.execute_query(
        "SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name"
    )

    assert {r["name"] for r in records} == {CHARACTER_VECTOR_INDEX, RELATION_TYPE_VECTOR_INDEX}
