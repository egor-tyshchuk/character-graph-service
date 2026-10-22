from collections.abc import Sequence
from typing import Any

from neo4j.exceptions import ConstraintError

from app.domain.models import Article, ArticleFormat, ArticleStats, ArticleStatus
from app.graph.records import Neo4jStore, to_native

_SET_ARTICLE_STATUS = """
MATCH (a:Article {id: $id})
WHERE a.status <> 'done'
SET a.status = $status, a.error = null, a.updated_at = datetime()
RETURN a.id AS id
"""

_MARK_ARTICLE_FAILED = """
MATCH (a:Article {id: $id})
WHERE a.status <> 'done'
SET a.status = $status, a.error = $error, a.updated_at = datetime()
RETURN a.id AS id
"""

_LIST_STUCK_ARTICLES = """
MATCH (a:Article)
WHERE a.status IN $stuck AND a.updated_at < datetime() - duration({seconds: $older_than})
RETURN a.id AS id
ORDER BY id
"""

_FAIL_STUCK_ARTICLES = """
MATCH (a:Article)
WHERE a.id IN $ids
  AND a.status IN $stuck
  AND a.updated_at < datetime() - duration({seconds: $older_than})
SET a.status = $failed, a.error = $error, a.updated_at = datetime()
RETURN a.id AS id
ORDER BY id
"""

_RESET_FAILED_ARTICLE = """
MATCH (a:Article {id: $id})
WHERE a.status = $failed
SET a.status = $pending, a.error = null, a.updated_at = datetime()
RETURN a.id AS id
"""


_STUCK_STATUSES = [ArticleStatus.PENDING.value, ArticleStatus.PROCESSING.value]


class DuplicateContentError(Exception):
    def __init__(self, existing_id: str) -> None:
        super().__init__(f"article with identical content already exists: {existing_id}")
        self.existing_id = existing_id


def _article_from_node(node: Any) -> Article:
    stats = None
    if node.get("stats_characters") is not None:
        stats = ArticleStats(
            characters=node["stats_characters"],
            relations=node["stats_relations"],
            images_described=node["stats_images_described"],
            images_failed=node.get("stats_images_failed", 0),
            images_skipped=node.get("stats_images_skipped", 0),
            unjudged_characters=node.get("stats_unjudged_characters", 0),
        )
    return Article(
        id=node["id"],
        title=node["title"],
        format=ArticleFormat(node["format"]),
        content_hash=node["content_hash"],
        status=ArticleStatus(node["status"]),
        error=node.get("error"),
        stats=stats,
        created_at=to_native(node["created_at"]),
        updated_at=to_native(node["updated_at"]),
    )


def _article_props(article: Article) -> dict[str, Any]:
    props: dict[str, Any] = {
        "id": article.id,
        "title": article.title,
        "format": article.format.value,
        "content_hash": article.content_hash,
        "status": article.status.value,
        "error": article.error,
        "created_at": article.created_at,
        "updated_at": article.updated_at,
    }
    if article.stats is not None:
        props["stats_characters"] = article.stats.characters
        props["stats_relations"] = article.stats.relations
        props["stats_images_described"] = article.stats.images_described
        props["stats_images_failed"] = article.stats.images_failed
        props["stats_images_skipped"] = article.stats.images_skipped
        props["stats_unjudged_characters"] = article.stats.unjudged_characters
    return props


class ArticleRepository(Neo4jStore):
    async def create_article(self, article: Article) -> Article:
        try:
            await self._driver.execute_query(
                "CREATE (a:Article) SET a = $props", props=_article_props(article)
            )
        except ConstraintError:
            existing = await self.get_article_by_hash(article.content_hash)
            if existing is None:
                raise
            raise DuplicateContentError(existing.id) from None
        return article

    async def get_article(self, article_id: str) -> Article | None:
        records, _, _ = await self._driver.execute_query(
            "MATCH (a:Article {id: $id}) RETURN a", id=article_id
        )
        return _article_from_node(records[0]["a"]) if records else None

    async def get_article_by_hash(self, content_hash: str) -> Article | None:
        records, _, _ = await self._driver.execute_query(
            "MATCH (a:Article {content_hash: $hash}) RETURN a", hash=content_hash
        )
        return _article_from_node(records[0]["a"]) if records else None

    async def set_article_status(self, article_id: str, status: ArticleStatus) -> bool:
        records, _, _ = await self._driver.execute_query(
            _SET_ARTICLE_STATUS, id=article_id, status=status.value
        )
        return bool(records)

    async def mark_article_failed(self, article_id: str, error: str) -> bool:
        records, _, _ = await self._driver.execute_query(
            _MARK_ARTICLE_FAILED, id=article_id, status=ArticleStatus.FAILED.value, error=error
        )
        return bool(records)

    async def list_stuck_articles(self, *, older_than_seconds: float) -> list[str]:
        records, _, _ = await self._driver.execute_query(
            _LIST_STUCK_ARTICLES, stuck=_STUCK_STATUSES, older_than=older_than_seconds
        )
        return [record["id"] for record in records]

    async def fail_stuck_articles(
        self, article_ids: Sequence[str], *, older_than_seconds: float, error: str
    ) -> list[str]:
        if not article_ids:
            return []
        records, _, _ = await self._driver.execute_query(
            _FAIL_STUCK_ARTICLES,
            ids=list(article_ids),
            stuck=_STUCK_STATUSES,
            older_than=older_than_seconds,
            failed=ArticleStatus.FAILED.value,
            error=error,
        )
        return [record["id"] for record in records]

    async def reset_failed_article(self, article_id: str) -> bool:
        records, _, _ = await self._driver.execute_query(
            _RESET_FAILED_ARTICLE,
            id=article_id,
            failed=ArticleStatus.FAILED.value,
            pending=ArticleStatus.PENDING.value,
        )
        return bool(records)
