from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from neo4j import AsyncDriver, AsyncManagedTransaction
from neo4j.exceptions import ConstraintError
from pydantic import TypeAdapter

from app.domain.models import (
    Article,
    ArticleFormat,
    ArticleStats,
    ArticleStatus,
    Character,
    Connection,
    Direction,
    Evidence,
    RelationTypeInfo,
)
from app.domain.profiles import profile_hash
from app.graph.schema import CHARACTER_VECTOR_INDEX, RELATION_TYPE_VECTOR_INDEX
from app.graph.upserts import (
    MATCH_STRENGTH,
    CharacterUpsert,
    EmbeddingRefresh,
    Mention,
    NotSameUpsert,
    RelationTypeUpsert,
    RelationUpsert,
    SimilarCharacter,
)
from app.llm.schemas import CharacterProfile

JUDGE_CONTEXT_MAX_RELATIONS = 20
JUDGE_CONTEXT_MAX_ARTICLES = 5

_MENTIONS = TypeAdapter(list[Mention])


def _character_projection(var: str) -> str:
    return (
        f"{var} {{.id, .name, .aliases, .description, "
        f"article_count: COUNT {{ ({var})-[:APPEARS_IN]->(:Article) }}}}"
    )


_MARK_ARTICLE_DONE = """
MATCH (a:Article {id: $article_id})
SET a.status = $status,
    a.error = null,
    a.stats_characters = $stats.characters,
    a.stats_relations = $stats.relations,
    a.stats_images_described = $stats.images_described,
    a.updated_at = datetime()
RETURN a.id AS id
"""

_MARK_ARTICLE_FAILED = """
MATCH (a:Article {id: $id})
WHERE a.status <> 'done'
SET a.status = $status, a.error = $error, a.updated_at = datetime()
RETURN a.id AS id
"""

_RESET_FAILED_ARTICLE = """
MATCH (a:Article {id: $id})
WHERE a.status = $failed
SET a.status = $pending, a.error = null, a.updated_at = datetime()
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

_EXISTING_RELATION_TYPES = """
MATCH (t:RelationType) WHERE t.name IN $names
RETURN t.name AS name
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

_FIND_SIMILAR_CHARACTERS = """
CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score
RETURN node.id AS id,
       node.name AS name,
       node.aliases AS aliases,
       node.description AS description,
       node.created_at AS created_at,
       2 * score - 1 AS similarity
ORDER BY similarity DESC, id
"""

_FIND_SIMILAR_RELATION_TYPES = """
CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score
RETURN node.name AS name, 2 * score - 1 AS similarity
ORDER BY similarity DESC, name
"""

_JUDGE_CONTEXT = """
UNWIND $ids AS id
MATCH (c:Character {id: id})
RETURN c.id AS id,
       COLLECT {
           MATCH (c)-[r:RELATED_TO]-(o:Character)
           WITH DISTINCT CASE
               WHEN startNode(r) = c THEN r.type + ' → ' + o.name
               ELSE '← ' + r.type + ' ' + o.name
           END AS line
           RETURN line ORDER BY line LIMIT $max_relations
       } AS relations,
       COLLECT {
           MATCH (c)-[:APPEARS_IN]->(a:Article)
           WITH DISTINCT a.title AS title
           RETURN title ORDER BY title LIMIT $max_articles
       } AS titles
"""

_LIST_CHARACTERS = f"""
MATCH (c:Character)
WITH count(c) AS total
RETURN total,
       COLLECT {{
           MATCH (c:Character)
           RETURN {_character_projection("c")} AS character
           ORDER BY toLower(c.name), c.id
           SKIP $offset LIMIT $limit
       }} AS characters
"""

_LIST_CHARACTERS_BY_ARTICLE = f"""
MATCH (c:Character)-[:APPEARS_IN]->(:Article {{id: $article_id}})
RETURN {_character_projection("c")} AS character
ORDER BY toLower(c.name), c.id
"""

_GET_CHARACTER = f"""
MATCH (c:Character {{id: $id}})
RETURN {_character_projection("c")} AS character
"""

_LIST_RELATION_TYPES = """
MATCH (:Character)-[r:RELATED_TO]->(:Character)
WITH r.type AS name, count(r) AS usage_count
MATCH (t:RelationType {name: name})
RETURN t.name AS name, t.description AS description, usage_count
ORDER BY usage_count DESC, name
"""

_FIND_CONNECTED = f"""
MATCH (c:Character {{id: $id}})-[r:RELATED_TO {{type: $type}}]-(o:Character)
WITH o, r, startNode(r) = c AS outgoing
ORDER BY r.article_id
WITH o, outgoing, collect(r {{.article_id, .description, .evidence}}) AS evidence
RETURN {_character_projection("o")} AS character, outgoing, evidence
ORDER BY toLower(o.name), o.id, outgoing DESC
"""

_FRESH_NOT_SAME = """
UNWIND $pairs AS pair
MATCH (:Character {id: pair.low})-[n:NOT_SAME]->(:Character {id: pair.high})
WHERE n.hash_low = pair.hash_low AND n.hash_high = pair.hash_high
RETURN pair.low AS low, pair.high AS high
"""

_CHARACTER_PROFILES = """
MATCH (c:Character) WHERE c.id IN $ids
RETURN c.id AS id, c {.name, .aliases, .description} AS profile
"""

_COUNT_CHARACTERS = """
MATCH (c:Character) WHERE c.id IN $merged_ids
RETURN count(c) AS found
"""

_MERGED_APPEARANCES = """
UNWIND range(0, size($merged_ids) - 1) AS position
MATCH (:Character {id: $merged_ids[position]})-[m:APPEARS_IN]->(a:Article)
RETURN a.id AS article_id,
       m.mentions_json AS mentions_json,
       m.match AS match,
       m.similarity AS similarity
ORDER BY position, article_id
"""

_WRITE_MERGED_APPEARANCES = """
MATCH (t:Character {id: $target_id})
UNWIND $rows AS row
MATCH (a:Article {id: row.article_id})
MERGE (t)-[m:APPEARS_IN]->(a)
SET m.mentions_json = row.mentions_json, m.match = row.match, m.similarity = row.similarity
"""

_KEEP_LONGER_EVIDENCE = """
ON CREATE SET m = properties(r)
ON MATCH SET m += CASE
    WHEN size(r.evidence) > size(m.evidence) THEN r {.description, .evidence}
    ELSE {}
END
"""

_MOVE_OUTGOING_RELATIONS = f"""
MATCH (t:Character {{id: $target_id}})
MATCH (s:Character)-[r:RELATED_TO]->(o:Character)
WHERE s.id IN $source_ids AND NOT o.id IN $merged_ids
MERGE (t)-[m:RELATED_TO {{type: r.type, article_id: r.article_id}}]->(o)
{_KEEP_LONGER_EVIDENCE}
"""

_MOVE_INCOMING_RELATIONS = f"""
MATCH (t:Character {{id: $target_id}})
MATCH (o:Character)-[r:RELATED_TO]->(s:Character)
WHERE s.id IN $source_ids AND NOT o.id IN $merged_ids
MERGE (o)-[m:RELATED_TO {{type: r.type, article_id: r.article_id}}]->(t)
{_KEEP_LONGER_EVIDENCE}
"""

_DELETE_SOURCES = """
MATCH (s:Character) WHERE s.id IN $source_ids
DETACH DELETE s
"""

_UPDATE_MERGED_TARGET = """
MATCH (t:Character {id: $target_id})
SET t.name = $profile.name,
    t.aliases = $profile.aliases,
    t.description = $profile.description,
    t.embedding = $embedding,
    t.embedding_stale = false,
    t.updated_at = datetime()
"""


class DuplicateContentError(Exception):
    def __init__(self, existing_id: str) -> None:
        super().__init__(f"article with identical content already exists: {existing_id}")
        self.existing_id = existing_id


def _to_native(value: Any) -> datetime:
    result: datetime = value.to_native()
    return result


def _article_from_node(node: Any) -> Article:
    stats = None
    if node.get("stats_characters") is not None:
        stats = ArticleStats(
            characters=node["stats_characters"],
            relations=node["stats_relations"],
            images_described=node["stats_images_described"],
        )
    return Article(
        id=node["id"],
        title=node["title"],
        format=ArticleFormat(node["format"]),
        content_hash=node["content_hash"],
        status=ArticleStatus(node["status"]),
        error=node.get("error"),
        stats=stats,
        created_at=_to_native(node["created_at"]),
        updated_at=_to_native(node["updated_at"]),
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
    return props


def _character_row(character: CharacterUpsert) -> dict[str, Any]:
    first = character.mentions[0] if character.mentions else None
    return {
        "id": character.id,
        "name": character.name,
        "aliases": character.aliases,
        "description": character.description,
        "embedding": character.embedding,
        "mentions_json": _MENTIONS.dump_json(character.mentions).decode(),
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


def _stored_profile(props: Mapping[str, Any]) -> CharacterProfile:
    return CharacterProfile(
        name=props["name"], aliases=props["aliases"], description=props["description"]
    )


def _check_stored(stored: int, expected: int, what: str) -> None:
    if stored != expected:
        raise LookupError(
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


def _check_merge_ids(target_id: str, source_ids: Sequence[str]) -> None:
    if not source_ids:
        raise ValueError("at least one source character is required")
    if target_id in source_ids:
        raise ValueError(f"target character is among the sources: {target_id}")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source characters must be distinct")


def _appearance_strength(edge: Mapping[str, Any]) -> tuple[int, float]:
    kind = edge["match"]
    rank = MATCH_STRENGTH.index(kind) if kind in MATCH_STRENGTH else len(MATCH_STRENGTH)
    return rank, -(edge["similarity"] or 0.0)


def _combined_appearances(edges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_article: dict[str, list[Mapping[str, Any]]] = {}
    for edge in edges:
        by_article.setdefault(edge["article_id"], []).append(edge)
    rows = []
    for article_id, group in by_article.items():
        strongest = min(group, key=_appearance_strength)
        mentions = [
            mention for edge in group for mention in _MENTIONS.validate_json(edge["mentions_json"])
        ]
        rows.append(
            {
                "article_id": article_id,
                "mentions_json": _MENTIONS.dump_json(mentions).decode(),
                "match": strongest["match"],
                "similarity": strongest["similarity"],
            }
        )
    return rows


async def _merge_into_target(
    tx: AsyncManagedTransaction,
    target_id: str,
    source_ids: list[str],
    profile: CharacterProfile,
    embedding: list[float],
) -> None:
    ids = {"target_id": target_id, "source_ids": source_ids, "merged_ids": [target_id, *source_ids]}
    record = await (await tx.run(_COUNT_CHARACTERS, ids)).single(strict=True)
    if record["found"] != len(ids["merged_ids"]):
        raise LookupError(f"a merged character does not exist: {ids['merged_ids']}")
    edges = await (await tx.run(_MERGED_APPEARANCES, ids)).data()
    statements: tuple[tuple[str, dict[str, Any]], ...] = (
        (_WRITE_MERGED_APPEARANCES, {"rows": _combined_appearances(edges)}),
        (_MOVE_OUTGOING_RELATIONS, {}),
        (_MOVE_INCOMING_RELATIONS, {}),
        (_DELETE_SOURCES, {}),
        (
            _UPDATE_MERGED_TARGET,
            {
                "profile": profile.model_dump(include={"name", "aliases", "description"}),
                "embedding": embedding,
            },
        ),
    )
    for query, params in statements:
        await (await tx.run(query, {**ids, **params})).consume()


class GraphRepository:
    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

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

    async def set_article_status(
        self, article_id: str, status: ArticleStatus, *, error: str | None = None
    ) -> None:
        await self._driver.execute_query(
            "MATCH (a:Article {id: $id}) "
            "SET a.status = $status, a.error = $error, a.updated_at = datetime()",
            id=article_id,
            status=status.value,
            error=error,
        )

    async def mark_article_failed(self, article_id: str, error: str) -> bool:
        records, _, _ = await self._driver.execute_query(
            _MARK_ARTICLE_FAILED, id=article_id, status=ArticleStatus.FAILED.value, error=error
        )
        return bool(records)

    async def reset_failed_article(self, article_id: str) -> bool:
        records, _, _ = await self._driver.execute_query(
            _RESET_FAILED_ARTICLE,
            id=article_id,
            failed=ArticleStatus.FAILED.value,
            pending=ArticleStatus.PENDING.value,
        )
        return bool(records)

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

    async def existing_relation_types(self, names: Sequence[str]) -> set[str]:
        if not names:
            return set()
        records, _, _ = await self._driver.execute_query(
            _EXISTING_RELATION_TYPES, names=list(names)
        )
        return {record["name"] for record in records}

    async def find_similar_characters(
        self, embedding: Sequence[float], k: int
    ) -> list[SimilarCharacter]:
        records, _, _ = await self._driver.execute_query(
            _FIND_SIMILAR_CHARACTERS,
            index=CHARACTER_VECTOR_INDEX,
            k=k,
            embedding=list(embedding),
        )
        return [
            SimilarCharacter(
                id=record["id"],
                name=record["name"],
                aliases=record["aliases"],
                description=record["description"],
                similarity=record["similarity"],
                created_at=_to_native(record["created_at"]),
            )
            for record in records
        ]

    async def find_similar_relation_types(
        self, embedding: Sequence[float], k: int
    ) -> list[tuple[str, float]]:
        records, _, _ = await self._driver.execute_query(
            _FIND_SIMILAR_RELATION_TYPES,
            index=RELATION_TYPE_VECTOR_INDEX,
            k=k,
            embedding=list(embedding),
        )
        return [(record["name"], record["similarity"]) for record in records]

    async def get_judge_context(
        self, character_ids: Sequence[str]
    ) -> dict[str, tuple[list[str], list[str]]]:
        records, _, _ = await self._driver.execute_query(
            _JUDGE_CONTEXT,
            ids=list(character_ids),
            max_relations=JUDGE_CONTEXT_MAX_RELATIONS,
            max_articles=JUDGE_CONTEXT_MAX_ARTICLES,
        )
        return {record["id"]: (record["relations"], record["titles"]) for record in records}

    async def list_characters(self, *, limit: int, offset: int) -> tuple[list[Character], int]:
        records, _, _ = await self._driver.execute_query(
            _LIST_CHARACTERS, limit=limit, offset=offset
        )
        record = records[0]
        return [Character.model_validate(c) for c in record["characters"]], record["total"]

    async def list_characters_by_article(self, article_id: str) -> list[Character]:
        records, _, _ = await self._driver.execute_query(
            _LIST_CHARACTERS_BY_ARTICLE, article_id=article_id
        )
        return [Character.model_validate(record["character"]) for record in records]

    async def get_character(self, character_id: str) -> Character | None:
        records, _, _ = await self._driver.execute_query(_GET_CHARACTER, id=character_id)
        return Character.model_validate(records[0]["character"]) if records else None

    async def get_character_profiles(self, ids: Sequence[str]) -> dict[str, CharacterProfile]:
        records, _, _ = await self._driver.execute_query(_CHARACTER_PROFILES, ids=list(ids))
        return {record["id"]: _stored_profile(record["profile"]) for record in records}

    async def merge_characters(
        self,
        target_id: str,
        source_ids: Sequence[str],
        *,
        profile: CharacterProfile,
        embedding: Sequence[float],
    ) -> None:
        _check_merge_ids(target_id, source_ids)
        async with self._driver.session() as session:
            await session.execute_write(
                _merge_into_target, target_id, list(source_ids), profile, list(embedding)
            )

    async def list_relation_types(self) -> list[RelationTypeInfo]:
        records, _, _ = await self._driver.execute_query(_LIST_RELATION_TYPES)
        return [RelationTypeInfo.model_validate(dict(record)) for record in records]

    async def find_connected(self, character_id: str, relation_type: str) -> list[Connection]:
        records, _, _ = await self._driver.execute_query(
            _FIND_CONNECTED, id=character_id, type=relation_type
        )
        return [
            Connection(
                character=Character.model_validate(record["character"]),
                relation_type=relation_type,
                direction=Direction.OUTGOING if record["outgoing"] else Direction.INCOMING,
                evidence=[Evidence.model_validate(item) for item in record["evidence"]],
            )
            for record in records
        ]

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
            own_hash = profile_hash(_stored_profile(record["own"]))
            other_hash = profile_hash(_stored_profile(record["other"]))
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
