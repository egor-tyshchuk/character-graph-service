import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
import pytest_asyncio
from neo4j import AsyncDriver
from redis.asyncio import Redis

from app.config import Settings
from app.domain.models import Character, CharacterProfile, Direction
from app.domain.profiles import profile_embedding_text, profile_hash
from app.errors import CharactersNotFoundError
from app.graph.repository import GraphRepository
from app.graph.upserts import CharacterUpsert, MatchKind, Mention, RelationUpsert
from app.llm.client import LLMClient, LLMError, UnconfiguredLLMClient
from app.pipeline.deps import PipelineDeps
from app.pipeline.lock import RESOLVE_LOCK_NAME, ResolveLockLostError, ResolveLockTimeoutError
from app.pipeline.merge_characters import merge_characters
from app.storage import BlobStore
from tests.fakes import FakeLLMClient
from tests.integration.test_graph_repository import (
    character,
    count,
    persist,
    relation,
    seed_article,
)


@pytest_asyncio.fixture
async def redis(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.redis_url)
    await client.delete(RESOLVE_LOCK_NAME)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def deps(
    repo: GraphRepository, redis: Redis, settings: Settings, fake_llm: FakeLLMClient
) -> PipelineDeps:
    return make_deps(repo, redis, settings, fake_llm)


def make_deps(
    repo: GraphRepository, redis: Redis, settings: Settings, llm: LLMClient
) -> PipelineDeps:
    return PipelineDeps(
        repo=repo, llm=llm, blobs=BlobStore(settings.upload_dir), redis=redis, settings=settings
    )


def mention(name: str, match: MatchKind, similarity: float | None) -> Mention:
    return Mention(surface_name=name, description="", match=match, similarity=similarity)


def profile_of(upsert: CharacterUpsert) -> CharacterProfile:
    return CharacterProfile(
        name=upsert.name, aliases=upsert.aliases, description=upsert.description
    )


async def scalar(driver: AsyncDriver, query: str, **params: Any) -> Any:
    records, _, _ = await driver.execute_query(query, params)
    return records[0][0]


async def appearances(driver: AsyncDriver, character_id: str) -> list[dict[str, Any]]:
    records, _, _ = await driver.execute_query(
        "MATCH (:Character {id: $id})-[m:APPEARS_IN]->(a:Article) "
        "RETURN a.id AS article_id, m.match AS match, m.similarity AS similarity, "
        "m.mentions_json AS mentions ORDER BY a.id",
        id=character_id,
    )
    return [dict(record) for record in records]


async def relation_rows(driver: AsyncDriver) -> set[tuple[str, str, str, str]]:
    records, _, _ = await driver.execute_query(
        "MATCH (s:Character)-[r:RELATED_TO]->(t:Character) "
        "RETURN s.id AS source, r.type AS type, t.id AS target, r.evidence AS evidence"
    )
    return {
        (record["source"], record["type"], record["target"], record["evidence"])
        for record in records
    }


async def test_merge_moves_edges_and_deletes_sources(
    repo: GraphRepository, driver: AsyncDriver, deps: PipelineDeps
) -> None:
    anna, duplicate = character("Anna Petrova"), character("Anya Petrova")
    mark, pyotr = character("Mark Petrov"), character("Pyotr")
    await persist(repo, await seed_article(repo, "First"), [anna, mark])
    await persist(
        repo,
        await seed_article(repo, "Second"),
        [duplicate, mark, pyotr],
        [relation(duplicate, mark, "FRIEND_OF"), relation(pyotr, duplicate, "ENEMY_OF")],
    )

    merged = await merge_characters(anna.id, [duplicate.id], deps)

    assert merged.id == anna.id
    assert merged.article_count == 2
    assert await repo.get_character(duplicate.id) is None
    assert await count(driver, "(:Character)") == 3
    [friend] = await repo.find_connected(anna.id, "FRIEND_OF")
    assert (friend.character.id, friend.direction) == (mark.id, Direction.OUTGOING)
    [enemy] = await repo.find_connected(anna.id, "ENEMY_OF")
    assert (enemy.character.id, enemy.direction) == (pyotr.id, Direction.INCOMING)
    assert enemy.evidence[0].evidence == "quote about Pyotr and Anya Petrova"


async def test_merge_drops_relations_between_merged_nodes(
    repo: GraphRepository, driver: AsyncDriver, deps: PipelineDeps
) -> None:
    anna, anya, annushka = character("Anna"), character("Anya"), character("Annushka")
    mark = character("Mark")
    await persist(
        repo,
        await seed_article(repo),
        [anna, anya, annushka, mark],
        [
            relation(anya, anna, "SIBLING_OF"),
            relation(anna, annushka, "SIBLING_OF"),
            relation(annushka, anya, "FRIEND_OF"),
            relation(annushka, mark, "FRIEND_OF"),
        ],
    )

    await merge_characters(anna.id, [anya.id, annushka.id], deps)

    assert {(s, t, o) for s, t, o, _ in await relation_rows(driver)} == {
        (anna.id, "FRIEND_OF", mark.id)
    }
    assert await count(driver, "(c:Character)-[:RELATED_TO]->(c)") == 0


async def test_merge_unions_aliases_and_reembeds(
    repo: GraphRepository, driver: AsyncDriver, deps: PipelineDeps, fake_llm: FakeLLMClient
) -> None:
    anna = character("Anna Petrova", aliases=["Anya"])
    russian = character("Анна Петрова", aliases=["Annushka", "anya"]).model_copy(
        update={"description": "Anna Petrova, the lighthouse keeper's sister."}
    )
    await persist(repo, await seed_article(repo), [anna, russian])

    merged = await merge_characters(anna.id, [russian.id], deps)

    expected = CharacterProfile(
        name="Anna Petrova",
        aliases=["Annushka", "Anya", "Анна Петрова"],
        description=russian.description,
    )
    assert merged == Character(
        id=anna.id,
        name=expected.name,
        aliases=expected.aliases,
        description=expected.description,
        article_count=1,
    )
    assert fake_llm.embed_calls[-1] == [profile_embedding_text(expected)]
    [vector] = await fake_llm.embed([profile_embedding_text(expected)])
    records, _, _ = await driver.execute_query(
        "MATCH (c:Character {id: $id}) RETURN c.embedding AS embedding, "
        "c.embedding_stale AS stale, c.updated_at > c.created_at AS touched",
        id=anna.id,
    )
    assert records[0]["embedding"] == pytest.approx(vector)
    assert records[0]["stale"] is False
    assert records[0]["touched"] is True


async def test_merge_removes_not_same_between_merged_nodes(
    repo: GraphRepository, driver: AsyncDriver, deps: PipelineDeps
) -> None:
    anna, anya = character("Anna"), character("Anya")
    elena, olga = character("Elena"), character("Olga")
    await persist(repo, await seed_article(repo), [anna, anya, elena, olga])
    for a, b in ((anna, anya), (anya, elena), (anna, olga)):
        await repo.record_not_same(
            a.id,
            b.id,
            reason="different people",
            hash_a=profile_hash(profile_of(a)),
            hash_b=profile_hash(profile_of(b)),
        )

    await merge_characters(anna.id, [anya.id], deps)

    records, _, _ = await driver.execute_query(
        "MATCH (a:Character)-[:NOT_SAME]->(b:Character) RETURN a.id AS a, b.id AS b"
    )
    assert [{record["a"], record["b"]} for record in records] == [{anna.id, olga.id}]


async def test_merge_combines_appearances_in_a_shared_article(
    repo: GraphRepository, driver: AsyncDriver, deps: PipelineDeps
) -> None:
    article_id = await seed_article(repo)
    anna_mentions = [mention("Anna", MatchKind.NEW, 0.41)]
    anya_mentions = [
        mention("Anya", MatchKind.VECTOR, 0.93),
        mention("Anyuta", MatchKind.JUDGE, 0.88),
    ]
    anna = character("Anna", mentions=anna_mentions)
    anya = character("Anya", mentions=anya_mentions)
    await persist(repo, article_id, [anna, anya])

    merged = await merge_characters(anna.id, [anya.id], deps)

    assert merged.article_count == 1
    [edge] = await appearances(driver, anna.id)
    assert (edge["article_id"], edge["match"]) == (article_id, "vector")
    assert edge["similarity"] == pytest.approx(0.93)
    stored = [Mention.model_validate(item) for item in json.loads(edge["mentions"])]
    assert stored == [*anna_mentions, *anya_mentions]


async def test_merge_keeps_stronger_appearance_with_higher_similarity(
    repo: GraphRepository, driver: AsyncDriver, deps: PipelineDeps
) -> None:
    article_id = await seed_article(repo)
    anna = character("Anna", mentions=[mention("Anna", MatchKind.JUDGE, 0.81)])
    anya = character("Anya", mentions=[mention("Anya", MatchKind.JUDGE, 0.86)])
    await persist(repo, article_id, [anna, anya])

    await merge_characters(anna.id, [anya.id], deps)

    [edge] = await appearances(driver, anna.id)
    assert edge["match"] == "judge"
    assert edge["similarity"] == pytest.approx(0.86)


@pytest.mark.parametrize("incoming", [False, True], ids=["outgoing", "incoming"])
@pytest.mark.parametrize("source_has_longer", [True, False], ids=["source-longer", "target-longer"])
async def test_merge_keeps_longer_evidence_on_relation_collision(
    repo: GraphRepository,
    driver: AsyncDriver,
    deps: PipelineDeps,
    source_has_longer: bool,
    incoming: bool,
) -> None:
    anna, anya, mark = character("Anna"), character("Anya"), character("Mark")
    short, long = "Short quote.", "A much longer quote about the friendship."
    target_evidence, source_evidence = (short, long) if source_has_longer else (long, short)

    def friend_of(merged: CharacterUpsert, evidence: str) -> RelationUpsert:
        source, target = (mark, merged) if incoming else (merged, mark)
        return relation(source, target, "FRIEND_OF").model_copy(update={"evidence": evidence})

    await persist(
        repo,
        await seed_article(repo),
        [anna, anya, mark],
        [friend_of(anna, target_evidence), friend_of(anya, source_evidence)],
    )

    await merge_characters(anna.id, [anya.id], deps)

    source, target = (mark.id, anna.id) if incoming else (anna.id, mark.id)
    assert await relation_rows(driver) == {(source, "FRIEND_OF", target, long)}


async def test_merge_unknown_id_raises_not_found_and_writes_nothing(
    repo: GraphRepository, driver: AsyncDriver, deps: PipelineDeps
) -> None:
    anna = character("Anna")
    await persist(repo, await seed_article(repo), [anna])

    with pytest.raises(CharactersNotFoundError, match="missing-id"):
        await merge_characters(anna.id, ["missing-id"], deps)

    assert await count(driver, "(:Character)") == 1


async def test_merge_maps_character_vanished_in_transaction_to_not_found(
    repo: GraphRepository, driver: AsyncDriver, deps: PipelineDeps, monkeypatch: pytest.MonkeyPatch
) -> None:
    anna = character("Anna")
    await persist(repo, await seed_article(repo), [anna])
    ghost = CharacterProfile(name="Ghost", aliases=[], description="")
    loaded = await repo.get_character_profiles([anna.id])

    async def profiles_with_ghost(ids: Sequence[str]) -> dict[str, CharacterProfile]:
        return {**loaded, "ghost-id": ghost}

    monkeypatch.setattr(repo, "get_character_profiles", profiles_with_ghost)

    with pytest.raises(CharactersNotFoundError, match="ghost-id"):
        await merge_characters(anna.id, ["ghost-id"], deps)

    assert await count(driver, "(:Character)") == 1


@pytest.mark.parametrize(
    "source_ids",
    [
        pytest.param(["target"], id="target-in-sources"),
        pytest.param(["a", "a"], id="duplicates"),
        pytest.param([], id="empty"),
    ],
)
async def test_repository_rejects_invalid_source_ids(
    repo: GraphRepository, source_ids: Sequence[str]
) -> None:
    profile = CharacterProfile(name="Anna", aliases=[], description="")

    with pytest.raises(ValueError):
        await repo.merge_characters("target", source_ids, profile=profile, embedding=[1.0])


async def test_merge_llm_failure_leaves_graph_untouched(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    anna, anya = character("Anna"), character("Anya")
    await persist(repo, await seed_article(repo), [anna, anya])
    deps = make_deps(repo, redis, settings, UnconfiguredLLMClient())

    with pytest.raises(LLMError):
        await merge_characters(anna.id, [anya.id], deps)

    assert await count(driver, "(:Character)") == 2
    assert await redis.exists(RESOLVE_LOCK_NAME) == 0


async def test_merge_waits_for_the_resolve_lock_up_to_the_merge_timeout(
    repo: GraphRepository, redis: Redis, settings: Settings, fake_llm: FakeLLMClient
) -> None:
    anna, anya = character("Anna"), character("Anya")
    await persist(repo, await seed_article(repo), [anna, anya])
    impatient = Settings(**{**settings.model_dump(), "merge_lock_acquire_timeout_seconds": 0.3})
    deps = make_deps(repo, redis, impatient, fake_llm)
    await redis.set(RESOLVE_LOCK_NAME, b"another-worker", px=10_000)

    with pytest.raises(ResolveLockTimeoutError):
        await asyncio.wait_for(merge_characters(anna.id, [anya.id], deps), timeout=5)

    assert await repo.get_character(anya.id) is not None
    assert fake_llm.embed_calls == []


class LockStealingEmbedLLM(FakeLLMClient):
    def __init__(self, redis: Redis) -> None:
        super().__init__(dim=8)
        self.redis = redis

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        await self.redis.delete(RESOLVE_LOCK_NAME)
        return await super().embed(texts)


async def test_merge_aborts_when_lock_is_lost_before_write(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    anna, anya = character("Anna"), character("Anya")
    await persist(repo, await seed_article(repo), [anna, anya])
    deps = make_deps(repo, redis, settings, LockStealingEmbedLLM(redis))

    with pytest.raises(ResolveLockLostError):
        await merge_characters(anna.id, [anya.id], deps)

    assert await count(driver, "(:Character)") == 2
    assert await scalar(driver, "MATCH (c:Character {id: $id}) RETURN c.aliases", id=anna.id) == []
