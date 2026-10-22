from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.domain.models import (
    Character,
    CharacterProfile,
    Connection,
    Direction,
    Evidence,
    RelationTypeInfo,
)
from app.domain.profiles import JUDGE_MAX_ARTICLES, JUDGE_MAX_RELATIONS
from app.graph.records import Neo4jStore, character_projection, to_native
from app.graph.schema import CHARACTER_VECTOR_INDEX, RELATION_TYPE_VECTOR_INDEX
from app.graph.upserts import SimilarCharacter, StoredCharacter

_EXISTING_RELATION_TYPES = """
MATCH (t:RelationType) WHERE t.name IN $names
RETURN t.name AS name
"""

_FIND_SIMILAR_CHARACTERS = """
CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node
WITH node, 2 * vector.similarity.cosine(node.embedding, $embedding) - 1 AS similarity
RETURN node.id AS id,
       node.name AS name,
       node.aliases AS aliases,
       node.description AS description,
       node.created_at AS created_at,
       similarity
ORDER BY similarity DESC, id
"""

_FIND_SIMILAR_RELATION_TYPES = """
CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node
WITH node, 2 * vector.similarity.cosine(node.embedding, $embedding) - 1 AS similarity
RETURN node.name AS name, similarity
ORDER BY similarity DESC, name
"""

_JUDGE_CONTEXT = """
UNWIND $ids AS id
MATCH (c:Character {id: id})
RETURN c.id AS id,
       COLLECT {
           MATCH (c)-[r:RELATED_TO]-(o:Character)
           WITH DISTINCT c.name AS own, startNode(r) = c AS outgoing,
                         r.type AS type, o.name AS other
           RETURN CASE
               WHEN outgoing THEN own + ' ' + type + ' ' + other
               ELSE other + ' ' + type + ' ' + own
           END AS line
           ORDER BY outgoing DESC, type, other LIMIT $max_relations
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
           RETURN {character_projection("c")} AS character
           ORDER BY toLower(c.name), c.id
           SKIP $offset LIMIT $limit
       }} AS characters
"""

_LIST_CHARACTERS_BY_ARTICLE = f"""
MATCH (c:Character)-[:APPEARS_IN]->(:Article {{id: $article_id}})
RETURN {character_projection("c")} AS character
ORDER BY toLower(c.name), c.id
"""

_GET_CHARACTER = f"""
MATCH (c:Character {{id: $id}})
RETURN {character_projection("c")} AS character
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
RETURN {character_projection("o")} AS character, outgoing, evidence
ORDER BY toLower(o.name), o.id, outgoing DESC
"""

_CHARACTER_IDS_PAGE = """
MATCH (c:Character)
WITH c, c.created_at.epochSeconds AS seconds, c.created_at.nanosecond AS nanos
WHERE $after IS NULL
   OR seconds > $after.seconds
   OR (seconds = $after.seconds AND nanos > $after.nanos)
   OR (seconds = $after.seconds AND nanos = $after.nanos AND c.id > $after.id)
RETURN c.id AS id, seconds, nanos
ORDER BY seconds, nanos, id
LIMIT $limit
"""

_STALE_CHARACTERS_PAGE = """
MATCH (c:Character)
WHERE c.embedding_stale = true AND ($after IS NULL OR c.id > $after)
RETURN c.id AS id, c {.name, .aliases, .description} AS profile
ORDER BY c.id
LIMIT $limit
"""

_FLAGGED_CHARACTER_IDS_PAGE = """
MATCH (c:Character)
WHERE c.needs_reconcile = true AND ($after IS NULL OR c.id > $after)
RETURN c.id AS id
ORDER BY c.id
LIMIT $limit
"""

_STORED_CHARACTERS = """
MATCH (c:Character) WHERE c.id IN $ids AND c.embedding IS NOT NULL
RETURN c {
    .id, .name, .aliases, .description, .embedding, .created_at,
    needs_reconcile: coalesce(c.needs_reconcile, false)
} AS character
"""

_CHARACTER_PROFILES = """
MATCH (c:Character) WHERE c.id IN $ids
RETURN c.id AS id, c {.name, .aliases, .description} AS profile
"""


class CharacterRepository(Neo4jStore):
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
                created_at=to_native(record["created_at"]),
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
            max_relations=JUDGE_MAX_RELATIONS,
            max_articles=JUDGE_MAX_ARTICLES,
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

    async def iter_character_ids(self, *, batch_size: int) -> AsyncIterator[list[str]]:
        after: dict[str, Any] | None = None
        while True:
            records, _, _ = await self._driver.execute_query(
                _CHARACTER_IDS_PAGE, after=after, limit=batch_size
            )
            if not records:
                return
            yield [record["id"] for record in records]
            last = records[-1]
            after = {"id": last["id"], "seconds": last["seconds"], "nanos": last["nanos"]}

    async def iter_flagged_character_ids(self, *, batch_size: int) -> AsyncIterator[list[str]]:
        after: str | None = None
        while True:
            records, _, _ = await self._driver.execute_query(
                _FLAGGED_CHARACTER_IDS_PAGE, after=after, limit=batch_size
            )
            if not records:
                return
            yield [record["id"] for record in records]
            after = records[-1]["id"]

    async def iter_stale_characters(
        self, *, batch_size: int
    ) -> AsyncIterator[list[tuple[str, CharacterProfile]]]:
        after: str | None = None
        while True:
            records, _, _ = await self._driver.execute_query(
                _STALE_CHARACTERS_PAGE, after=after, limit=batch_size
            )
            if not records:
                return
            yield [
                (record["id"], CharacterProfile.model_validate(record["profile"]))
                for record in records
            ]
            after = records[-1]["id"]

    async def get_stored_characters(self, ids: Sequence[str]) -> dict[str, StoredCharacter]:
        records, _, _ = await self._driver.execute_query(_STORED_CHARACTERS, ids=list(ids))
        return {
            record["character"]["id"]: StoredCharacter.model_validate(
                {
                    **record["character"],
                    "created_at": to_native(record["character"]["created_at"]),
                }
            )
            for record in records
        }

    async def get_character_profiles(self, ids: Sequence[str]) -> dict[str, CharacterProfile]:
        records, _, _ = await self._driver.execute_query(_CHARACTER_PROFILES, ids=list(ids))
        return {
            record["id"]: CharacterProfile.model_validate(record["profile"]) for record in records
        }

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
