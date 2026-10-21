from collections.abc import Mapping, Sequence
from typing import Any

from neo4j import AsyncManagedTransaction

from app.domain.models import Character, CharacterProfile
from app.errors import CharactersNotFoundError
from app.graph.records import MENTIONS, Neo4jStore, character_projection
from app.graph.upserts import match_strength

_EXISTING_CHARACTER_IDS = """
MATCH (c:Character) WHERE c.id IN $merged_ids
RETURN collect(c.id) AS found
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

_UPDATE_MERGED_TARGET = f"""
MATCH (t:Character {{id: $target_id}})
SET t.name = $profile.name,
    t.aliases = $profile.aliases,
    t.description = $profile.description,
    t.embedding = $embedding,
    t.embedding_stale = false,
    t.updated_at = datetime()
RETURN {character_projection("t")} AS character
"""


def _check_merge_ids(target_id: str, source_ids: Sequence[str]) -> None:
    if not source_ids:
        raise ValueError("at least one source character is required")
    if target_id in source_ids:
        raise ValueError(f"target character is among the sources: {target_id}")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source characters must be distinct")


def _combined_appearances(edges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_article: dict[str, list[Mapping[str, Any]]] = {}
    for edge in edges:
        by_article.setdefault(edge["article_id"], []).append(edge)
    rows = []
    for article_id, group in by_article.items():
        strongest = min(group, key=lambda edge: match_strength(edge["match"], edge["similarity"]))
        mentions = [
            mention for edge in group for mention in MENTIONS.validate_json(edge["mentions_json"])
        ]
        rows.append(
            {
                "article_id": article_id,
                "mentions_json": MENTIONS.dump_json(mentions).decode(),
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
) -> Character:
    ids = {"target_id": target_id, "source_ids": source_ids, "merged_ids": [target_id, *source_ids]}
    record = await (await tx.run(_EXISTING_CHARACTER_IDS, ids)).single(strict=True)
    missing = [i for i in ids["merged_ids"] if i not in record["found"]]
    if missing:
        raise CharactersNotFoundError(f"Characters not found: {', '.join(missing)}")
    edges = await (await tx.run(_MERGED_APPEARANCES, ids)).data()
    statements: tuple[tuple[str, dict[str, Any]], ...] = (
        (_WRITE_MERGED_APPEARANCES, {"rows": _combined_appearances(edges)}),
        (_MOVE_OUTGOING_RELATIONS, {}),
        (_MOVE_INCOMING_RELATIONS, {}),
        (_DELETE_SOURCES, {}),
    )
    for query, params in statements:
        await (await tx.run(query, {**ids, **params})).consume()
    target = {
        "profile": profile.model_dump(include={"name", "aliases", "description"}),
        "embedding": embedding,
    }
    merged = await (await tx.run(_UPDATE_MERGED_TARGET, {**ids, **target})).single(strict=True)
    return Character.model_validate(merged["character"])


class MergeRepository(Neo4jStore):
    async def merge_characters(
        self,
        target_id: str,
        source_ids: Sequence[str],
        *,
        profile: CharacterProfile,
        embedding: Sequence[float],
    ) -> Character:
        _check_merge_ids(target_id, source_ids)
        async with self._driver.session() as session:
            return await session.execute_write(
                _merge_into_target, target_id, list(source_ids), profile, list(embedding)
            )
