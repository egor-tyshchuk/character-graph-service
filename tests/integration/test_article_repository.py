import uuid
from datetime import UTC, datetime

import pytest
from neo4j import AsyncDriver

from app.domain.models import Article, ArticleFormat, ArticleStats, ArticleStatus
from app.graph.articles import DuplicateContentError
from app.graph.repository import GraphRepository
from app.graph.schema import (
    CHARACTER_VECTOR_INDEX,
    RELATION_TYPE_VECTOR_INDEX,
    SchemaMismatchError,
    apply_schema,
)


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


async def test_article_stored_before_unjudged_counter_loads_with_zero(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    article = await repo.create_article(make_article("hash-legacy-stats"))
    await driver.execute_query(
        "MATCH (a:Article {id: $id}) "
        "SET a.status = 'done', a.stats_characters = 3, a.stats_relations = 2, "
        "a.stats_images_described = 1",
        id=article.id,
    )

    loaded = await repo.get_article(article.id)

    assert loaded is not None
    assert loaded.stats == ArticleStats(
        characters=3,
        relations=2,
        images_described=1,
        images_failed=0,
        images_skipped=0,
        unjudged_characters=0,
    )


async def test_image_loss_counters_round_trip(repo: GraphRepository, driver: AsyncDriver) -> None:
    article = await repo.create_article(make_article("hash-image-losses"))
    stats = ArticleStats(
        characters=1, relations=0, images_described=2, images_failed=3, images_skipped=4
    )

    await repo.persist_article(
        article.id, characters=[], relations=[], relation_types=[], stats=stats
    )

    loaded = await repo.get_article(article.id)
    assert loaded is not None
    assert loaded.stats == stats
    records, _, _ = await driver.execute_query(
        "MATCH (a:Article {id: $id}) RETURN a.stats_images_failed AS failed, "
        "a.stats_images_skipped AS skipped",
        id=article.id,
    )
    assert (records[0]["failed"], records[0]["skipped"]) == (3, 4)


async def test_set_status_failed_then_pending_clears_error(repo: GraphRepository) -> None:
    article = await repo.create_article(make_article())

    await repo.mark_article_failed(article.id, "boom")
    failed = await repo.get_article(article.id)
    assert failed is not None
    assert failed.status is ArticleStatus.FAILED
    assert failed.error == "boom"
    assert failed.updated_at > article.updated_at

    assert await repo.set_article_status(article.id, ArticleStatus.PENDING) is True
    pending = await repo.get_article(article.id)
    assert pending is not None
    assert pending.status is ArticleStatus.PENDING
    assert pending.error is None
    assert pending.updated_at >= failed.updated_at


async def test_set_status_never_overwrites_done(repo: GraphRepository) -> None:
    article = await repo.create_article(make_article())
    await repo.set_article_status(article.id, ArticleStatus.DONE)

    assert await repo.set_article_status(article.id, ArticleStatus.PROCESSING) is False
    assert await repo.set_article_status("missing", ArticleStatus.PROCESSING) is False
    unchanged = await repo.get_article(article.id)
    assert unchanged is not None
    assert unchanged.status is ArticleStatus.DONE


async def test_reset_failed_article_only_resets_failed(repo: GraphRepository) -> None:
    article = await repo.create_article(make_article())
    await repo.mark_article_failed(article.id, "boom")

    assert await repo.reset_failed_article(article.id) is True
    assert await repo.reset_failed_article(article.id) is False

    reset = await repo.get_article(article.id)
    assert reset is not None
    assert reset.status is ArticleStatus.PENDING
    assert reset.error is None


@pytest.mark.parametrize(
    "status", [ArticleStatus.PENDING, ArticleStatus.PROCESSING, ArticleStatus.DONE]
)
async def test_reset_failed_article_ignores_other_statuses(
    repo: GraphRepository, status: ArticleStatus
) -> None:
    article = await repo.create_article(make_article())
    await repo.set_article_status(article.id, status)

    assert await repo.reset_failed_article(article.id) is False
    unchanged = await repo.get_article(article.id)
    assert unchanged is not None
    assert unchanged.status is status


async def test_apply_schema_is_idempotent_and_creates_vector_indexes(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    await apply_schema(driver, 8)
    await apply_schema(driver, 8)

    records, _, _ = await driver.execute_query(
        "SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name"
    )

    assert {r["name"] for r in records} == {CHARACTER_VECTOR_INDEX, RELATION_TYPE_VECTOR_INDEX}


async def test_apply_schema_awaits_and_rejects_dimension_mismatch(driver: AsyncDriver) -> None:
    await apply_schema(driver, 8)

    with pytest.raises(SchemaMismatchError) as excinfo:
        await apply_schema(driver, 16)

    message = str(excinfo.value)
    assert "dimension 8" in message
    assert "embedding_dim is 16" in message
    assert CHARACTER_VECTOR_INDEX in message
