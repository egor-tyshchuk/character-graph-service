from collections.abc import Sequence
from typing import Any

from neo4j import AsyncManagedTransaction

from app.domain.models import ArticleStats, ArticleStatus, CharacterProfile
from app.domain.profiles import profile_hash
from app.errors import CharactersNotFoundError
from app.graph.records import MENTIONS, Neo4jStore
from app.graph.upserts import (
    CharacterUpsert,
    EmbeddingRefresh,
    NotSameUpsert,
    RelationTypeUpsert,
    RelationUpsert,
)

_MARK_ARTICLE_DONE = """
MATCH (a:Article {id: $article_id})
SET a.status = $status,
    a.error = null,
    a.stats_characters = $stats.characters,
    a.stats_relations = $stats.relations,
    a.stats_images_described = $stats.images_described,
    a.stats_unjudged_characters = $stats.unjudged_characters,
    a.updated_at = datetime()
RETURN a.id AS id
"""

_MERGE_RELATION_TYPES = """
UNWIND $relation_types AS row
MERGE (t:RelationType {name: row.name})
ON CREATE SET t.description = row.description, t.embedding = row.embedding
"""

_MERGE_CHARACTERS = """
MATCH (a:Article {id: $article_id})
UNWIND $characters AS row
MERGE (c:Character {id: row.id})
ON CREATE SET c.created_at = datetime()
SET c.name = row.name,
    c.aliases = row.aliases,
    c.description = row.description,
    c.embedding = coalesce(row.embedding, c.embedding),
    c.embedding_stale = row.embedding IS NULL,
    c.needs_reconcile = row.needs_reconcile OR coalesce(c.needs_reconcile, false),
    c.updated_at = datetime()
MERGE (c)-[m:APPEARS_IN]->(a)
SET m.mentions_json = row.mentions_json, m.match = row.match, m.similarity = row.similarity
"""

_MERGE_RELATIONS = """
UNWIND $rows AS row
MATCH (s:Character {id: row.source_id})
MATCH (t:Character {id: row.target_id})
MERGE (s)-[r:RELATED_TO {type: row.type, article_id: $article_id}]->(t)
SET r.description = row.description,
    r.evidence = row.evidence,
    r.created_at = coalesce(r.created_at, datetime())
RETURN count(r) AS stored
"""

_MERGE_NOT_SAME = """
UNWIND $rows AS row
MATCH (low:Character {id: row.low})
MATCH (high:Character {id: row.high})
MERGE (low)-[n:NOT_SAME]->(high)
SET n.reason = row.reason,
    n.judged_at = datetime(),
    n.hash_low = row.hash_low,
    n.hash_high = row.hash_high
RETURN count(n) AS stored
"""

_REFRESH_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (c:Character {id: row.id})
WHERE c.name = row.name AND c.aliases = row.aliases AND c.description = row.description
SET c.embedding = row.embedding, c.embedding_stale = false
RETURN c.id AS id
"""

_CLEAR_NEEDS_RECONCILE = """
MATCH (c:Character {id: $id})
SET c.needs_reconcile = false
"""

_NOT_SAME_PARTNERS = """
UNWIND $ids AS id
MATCH (c:Character {id: id})-[n:NOT_SAME]-(o:Character)
WITH c, o, n, startNode(n) = c AS is_low
RETURN c.id AS id,
       c {.name, .aliases, .description} AS own,
       o {.name, .aliases, .description} AS other,
       CASE WHEN is_low THEN n.hash_low ELSE n.hash_high END AS own_hash,
       CASE WHEN is_low THEN n.hash_high ELSE n.hash_low END AS other_hash
"""

_FRESH_NOT_SAME = """
UNWIND $pairs AS pair
MATCH (:Character {id: pair.low})-[n:NOT_SAME]->(:Character {id: pair.high})
WHERE n.hash_low = pair.hash_low AND n.hash_high = pair.hash_high
RETURN pair.low AS low, pair.high AS high
"""


def _character_row(character: CharacterUpsert) -> dict[str, Any]:
    first = character.mentions[0] if character.mentions else None
    return {
        "id": character.id,
        "name": character.name,
        "aliases": character.aliases,
        "description": character.description,
        "embedding": character.embedding,
        "needs_reconcile": character.needs_reconcile,
        "mentions_json": MENTIONS.dump_json(character.mentions).decode(),
        "match": first.match.value if first else None,
        "similarity": first.similarity if first else None,
    }


def _ordered_pair(a_id: str, b_id: str, hash_a: str, hash_b: str) -> dict[str, str]:
    (low, hash_low), (high, hash_high) = sorted([(a_id, hash_a), (b_id, hash_b)])
    return {"low": low, "high": high, "hash_low": hash_low, "hash_high": hash_high}


def _not_same_row(verdict: NotSameUpsert) -> dict[str, str]:
    if verdict.a_id == verdict.b_id:
        raise ValueError(f"a character cannot be NOT_SAME as itself: {verdict.a_id}")
    pair = _ordered_pair(verdict.a_id, verdict.b_id, verdict.hash_a, verdict.hash_b)
    return {**pair, "reason": verdict.reason}


def _check_stored(stored: int, expected: int, what: str) -> None:
    if stored != expected:
        raise CharactersNotFoundError(
            f"stored {stored} of {expected} {what}: an endpoint character does not exist"
        )


async def _write_counted(
    tx: AsyncManagedTransaction, query: str, rows: list[dict[str, Any]], what: str, **params: Any
) -> None:
    record = await (await tx.run(query, params, rows=rows)).single(strict=True)
    _check_stored(record["stored"], len(rows), what)


async def _write_article_graph(
    tx: AsyncManagedTransaction,
    article_id: str,
    characters: Sequence[CharacterUpsert],
    relations: Sequence[RelationUpsert],
    relation_types: Sequence[RelationTypeUpsert],
    not_same: Sequence[NotSameUpsert],
    stats: ArticleStats,
) -> None:
    result = await tx.run(
        _MARK_ARTICLE_DONE,
        article_id=article_id,
        status=ArticleStatus.DONE.value,
        stats=stats.model_dump(),
    )
    if await result.single() is None:
        raise LookupError(f"article not found: {article_id}")
    statements = (
        (_MERGE_RELATION_TYPES, {"relation_types": [t.model_dump() for t in relation_types]}),
        (_MERGE_CHARACTERS, {"characters": [_character_row(c) for c in characters]}),
    )
    for query, params in statements:
        await (await tx.run(query, params, article_id=article_id)).consume()
    await _write_counted(
        tx,
        _MERGE_RELATIONS,
        [r.model_dump() for r in relations],
        "relations",
        article_id=article_id,
    )
    await _write_counted(tx, _MERGE_NOT_SAME, [_not_same_row(v) for v in not_same], "NOT_SAME")


class PersistRepository(Neo4jStore):
    async def persist_article(
        self,
        article_id: str,
        *,
        characters: Sequence[CharacterUpsert],
        relations: Sequence[RelationUpsert],
        relation_types: Sequence[RelationTypeUpsert],
        not_same: Sequence[NotSameUpsert] = (),
        stats: ArticleStats,
    ) -> None:
        async with self._driver.session() as session:
            await session.execute_write(
                _write_article_graph,
                article_id,
                characters,
                relations,
                relation_types,
                not_same,
                stats,
            )

    async def refresh_character_embeddings(self, refreshes: Sequence[EmbeddingRefresh]) -> set[str]:
        records, _, _ = await self._driver.execute_query(
            _REFRESH_EMBEDDINGS, rows=[refresh.model_dump() for refresh in refreshes]
        )
        return {record["id"] for record in records}

    async def clear_needs_reconcile(self, character_id: str) -> None:
        await self._driver.execute_query(_CLEAR_NEEDS_RECONCILE, id=character_id)

    async def record_not_same(
        self, a_id: str, b_id: str, *, reason: str, hash_a: str, hash_b: str
    ) -> None:
        verdict = NotSameUpsert(a_id=a_id, b_id=b_id, reason=reason, hash_a=hash_a, hash_b=hash_b)
        records, _, _ = await self._driver.execute_query(
            _MERGE_NOT_SAME, rows=[_not_same_row(verdict)]
        )
        _check_stored(records[0]["stored"], 1, "NOT_SAME")

    async def fresh_not_same_hashes(self, character_ids: Sequence[str]) -> dict[str, set[str]]:
        if not character_ids:
            return {}
        records, _, _ = await self._driver.execute_query(
            _NOT_SAME_PARTNERS, ids=list(character_ids)
        )
        partners: dict[str, set[str]] = {}
        for record in records:
            own_hash = profile_hash(CharacterProfile.model_validate(record["own"]))
            other_hash = profile_hash(CharacterProfile.model_validate(record["other"]))
            if (record["own_hash"], record["other_hash"]) == (own_hash, other_hash):
                partners.setdefault(record["id"], set()).add(other_hash)
        return partners

    async def fresh_not_same(
        self, pairs: Sequence[tuple[str, str, str, str]]
    ) -> set[frozenset[str]]:
        if not pairs:
            return set()
        records, _, _ = await self._driver.execute_query(
            _FRESH_NOT_SAME, pairs=[_ordered_pair(*pair) for pair in pairs]
        )
        return {frozenset((record["low"], record["high"])) for record in records}
