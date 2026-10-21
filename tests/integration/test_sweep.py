import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from neo4j import AsyncDriver
from redis.asyncio import Redis

from app.config import Settings
from app.domain.models import Article, ArticleFormat, ArticleStatus
from app.graph.repository import GraphRepository
from app.pipeline.process import PipelineDeps
from app.pipeline.sweep import STUCK_ERROR, sweep_stuck_articles
from tests.fakes import FakeLLMClient
from tests.integration.test_merge_characters import make_deps

STUCK_AFTER_SECONDS = 4200


async def article_aged(
    repo: GraphRepository,
    status: ArticleStatus,
    *,
    age_seconds: int,
    error: str | None = None,
) -> str:
    now = datetime.now(UTC)
    then = now - timedelta(seconds=age_seconds)
    article = Article(
        id=str(uuid.uuid4()),
        title=f"{status} article",
        format=ArticleFormat.MARKDOWN,
        content_hash=str(uuid.uuid4()),
        status=status,
        error=error,
        created_at=then,
        updated_at=then,
    )
    await repo.create_article(article)
    return article.id


@pytest_asyncio.fixture
async def redis(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.redis_url)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def deps(
    repo: GraphRepository, redis: Redis, settings: Settings, fake_llm: FakeLLMClient
) -> PipelineDeps:
    return make_deps(repo, redis, settings, fake_llm)


async def test_sweep_fails_articles_stuck_past_the_job_timeout(
    repo: GraphRepository, deps: PipelineDeps, fake_llm: FakeLLMClient
) -> None:
    old = STUCK_AFTER_SECONDS + 60
    stuck_pending = await article_aged(repo, ArticleStatus.PENDING, age_seconds=old)
    stuck_processing = await article_aged(repo, ArticleStatus.PROCESSING, age_seconds=old)

    assert await sweep_stuck_articles(deps) == 2

    for article_id in (stuck_pending, stuck_processing):
        article = await repo.get_article(article_id)
        assert article is not None
        assert (article.status, article.error) == (ArticleStatus.FAILED, STUCK_ERROR)
    assert fake_llm.embed_calls == []
    assert fake_llm.judge_calls == []


async def test_sweep_leaves_done_failed_and_recent_articles_alone(
    repo: GraphRepository, deps: PipelineDeps
) -> None:
    old = STUCK_AFTER_SECONDS + 60
    recent = STUCK_AFTER_SECONDS - 60
    untouched = {
        await article_aged(repo, ArticleStatus.DONE, age_seconds=old): (
            ArticleStatus.DONE,
            None,
        ),
        await article_aged(repo, ArticleStatus.FAILED, age_seconds=old, error="boom"): (
            ArticleStatus.FAILED,
            "boom",
        ),
        await article_aged(repo, ArticleStatus.PENDING, age_seconds=recent): (
            ArticleStatus.PENDING,
            None,
        ),
        await article_aged(repo, ArticleStatus.PROCESSING, age_seconds=recent): (
            ArticleStatus.PROCESSING,
            None,
        ),
    }

    assert await sweep_stuck_articles(deps) == 0

    for article_id, expected in untouched.items():
        article = await repo.get_article(article_id)
        assert article is not None
        assert (article.status, article.error) == expected


async def test_sweep_fails_article_whose_worker_died_mid_processing(
    repo: GraphRepository, driver: AsyncDriver, deps: PipelineDeps
) -> None:
    article_id = await article_aged(repo, ArticleStatus.PENDING, age_seconds=0)
    await repo.set_article_status(article_id, ArticleStatus.PROCESSING)
    await driver.execute_query(
        "MATCH (a:Article {id: $id}) SET a.updated_at = a.updated_at - duration({hours: 2})",
        id=article_id,
    )

    assert await sweep_stuck_articles(deps) == 1

    article = await repo.get_article(article_id)
    assert article is not None
    assert (article.status, article.error) == (ArticleStatus.FAILED, STUCK_ERROR)


async def test_sweep_threshold_comes_from_settings(
    repo: GraphRepository,
    redis: Redis,
    settings: Settings,
    fake_llm: FakeLLMClient,
) -> None:
    article_id = await article_aged(repo, ArticleStatus.PROCESSING, age_seconds=120)
    impatient = settings.model_copy(update={"stuck_article_after_seconds": 60})

    assert await sweep_stuck_articles(make_deps(repo, redis, impatient, fake_llm)) == 1

    article = await repo.get_article(article_id)
    assert article is not None
    assert article.status is ArticleStatus.FAILED
