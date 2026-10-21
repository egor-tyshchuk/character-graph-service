import math
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from neo4j import AsyncDriver
from redis.asyncio import Redis
from structlog.testing import capture_logs

from app.config import Settings
from app.domain.models import CharacterProfile
from app.domain.profiles import profile_embedding_text, profile_hash
from app.graph.repository import GraphRepository
from app.graph.upserts import CharacterUpsert
from app.llm.client import LLMError
from app.pipeline.deps import PipelineDeps
from app.pipeline.lock import RESOLVE_LOCK_NAME
from app.pipeline.reconcile import ReconcileReport, reconcile_characters
from tests.fakes import FakeLLMClient, vec
from tests.integration.test_graph_repository import character, count, persist, seed_article
from tests.integration.test_merge_characters import make_deps, profile_of

JUDGE_BAND = vec(0.85, math.sqrt(1 - 0.85**2))


@pytest_asyncio.fixture
async def redis(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.redis_url)
    await client.delete(RESOLVE_LOCK_NAME)
    try:
        yield client
    finally:
        await client.aclose()


def reconcile_deps(
    repo: GraphRepository,
    redis: Redis,
    settings: Settings,
    llm: FakeLLMClient,
    **overrides: Any,
) -> PipelineDeps:
    return make_deps(repo, redis, settings.model_copy(update=overrides), llm)


async def persist_in_order(repo: GraphRepository, *characters: CharacterUpsert) -> None:
    for position, upsert in enumerate(characters):
        await persist(repo, await seed_article(repo, f"Article {position}"), [upsert])


async def character_ids(driver: AsyncDriver) -> set[str]:
    records, _, _ = await driver.execute_query("MATCH (c:Character) RETURN c.id AS id")
    return {record["id"] for record in records}


async def record_not_same(
    repo: GraphRepository, a: CharacterUpsert, b: CharacterUpsert, *, hash_a: str | None = None
) -> None:
    await repo.record_not_same(
        a.id,
        b.id,
        reason="different people",
        hash_a=hash_a or profile_hash(profile_of(a)),
        hash_b=profile_hash(profile_of(b)),
    )


async def mark_stale(driver: AsyncDriver, upsert: CharacterUpsert, old: list[float]) -> None:
    await driver.execute_query(
        "MATCH (c:Character {id: $id}) SET c.embedding = $old, c.embedding_stale = true",
        id=upsert.id,
        old=old,
    )


async def test_iter_character_ids_pages_in_creation_order(repo: GraphRepository) -> None:
    created = [character(name) for name in ("Anna", "Mark", "Olga", "Pyotr", "Elena")]
    await persist_in_order(repo, *created[:3])
    await persist(repo, await seed_article(repo, "Shared"), created[3:])
    shared = sorted(upsert.id for upsert in created[3:])

    batches = [batch async for batch in repo.iter_character_ids(batch_size=2)]

    assert batches == [[created[0].id, created[1].id], [created[2].id, shared[0]], [shared[1]]]


async def test_reconcile_merges_high_similarity_duplicates(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient()
    anna, anya = character("Anna Petrova"), character("Anya", embedding=vec(1))
    await persist_in_order(repo, anna, anya)

    report = await reconcile_characters(reconcile_deps(repo, redis, settings, llm))

    assert report == ReconcileReport(checked=1, merged=1, reembedded=0, merges=[(anna.id, anya.id)])
    assert await character_ids(driver) == {anna.id}
    merged = await repo.get_character(anna.id)
    assert merged is not None
    assert (merged.aliases, merged.article_count) == (["Anya"], 2)
    assert llm.judge_calls == []


async def test_reconcile_merges_the_newer_node_into_the_older_one(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(same_pairs=[("Anna Petrova", "Anya")])
    angle = math.radians(28)
    anna = character("Anna Petrova", embedding=vec(1, 0))
    anya = character("Anya", embedding=vec(math.cos(angle), -math.sin(angle)))
    annushka = character("Annushka", embedding=vec(math.cos(math.radians(25)), 0.42))
    await persist_in_order(repo, anna, anya, annushka)
    deps = reconcile_deps(repo, redis, settings, llm, dedup_candidates=1)

    report = await reconcile_characters(deps)

    assert report.merges == [(anna.id, anya.id)]
    assert await character_ids(driver) == {anna.id, annushka.id}
    assert [(c.name, e.name) for c, e in llm.judge_calls][:2] == [
        ("Anna Petrova", "Annushka"),
        ("Anya", "Anna Petrova"),
    ]


@pytest.mark.parametrize("same", [True, False], ids=["judged-same", "judged-different"])
async def test_reconcile_uses_judge_in_band(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings, same: bool
) -> None:
    llm = FakeLLMClient(same_pairs=[("Anna Petrova", "Anya")] if same else [])
    anna, anya = (
        character("Anna Petrova", embedding=vec(1)),
        character("Anya", embedding=JUDGE_BAND),
    )
    await persist_in_order(repo, anna, anya)

    report = await reconcile_characters(reconcile_deps(repo, redis, settings, llm))

    assert report.merged == (1 if same else 0)
    assert len(llm.judge_calls) == 1
    candidate, existing = llm.judge_calls[0]
    assert (candidate.name, candidate.articles, existing.name) == (
        "Anna Petrova",
        ["Article 0"],
        "Anya",
    )
    assert await character_ids(driver) == ({anna.id} if same else {anna.id, anya.id})
    assert await count(driver, "(:Character)-[:NOT_SAME]->(:Character)") == (0 if same else 1)


async def test_reconcile_records_not_same_with_current_hashes(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    anna, anya = (
        character("Anna Petrova", embedding=vec(1)),
        character("Anya", embedding=JUDGE_BAND),
    )
    await persist_in_order(repo, anna, anya)

    await reconcile_characters(reconcile_deps(repo, redis, settings, FakeLLMClient()))

    fresh = await repo.fresh_not_same(
        [(anna.id, anya.id, profile_hash(profile_of(anna)), profile_hash(profile_of(anya)))]
    )
    assert fresh == {frozenset((anna.id, anya.id))}


async def test_reconcile_is_noop_on_clean_graph(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient()
    created = [
        character("Anna", embedding=vec(1)),
        character("Mark", embedding=vec(0, 1)),
        character("Olga", embedding=vec(0, 0, 1)),
    ]
    await persist_in_order(repo, *created)

    report = await reconcile_characters(reconcile_deps(repo, redis, settings, llm))

    assert report == ReconcileReport(checked=3, merged=0, reembedded=0, merges=[])
    assert await character_ids(driver) == {upsert.id for upsert in created}
    assert (llm.embed_calls, llm.judge_calls) == ([], [])


@pytest.mark.parametrize("embedding", [vec(1), JUDGE_BAND], ids=["auto-merge-band", "judge-band"])
async def test_reconcile_skips_fresh_not_same(
    repo: GraphRepository,
    driver: AsyncDriver,
    redis: Redis,
    settings: Settings,
    embedding: list[float],
) -> None:
    llm = FakeLLMClient(same_pairs=[("Anna Petrova", "Anya")])
    anna, anya = character("Anna Petrova", embedding=vec(1)), character("Anya", embedding=embedding)
    await persist_in_order(repo, anna, anya)
    await record_not_same(repo, anna, anya)

    report = await reconcile_characters(reconcile_deps(repo, redis, settings, llm))

    assert report.merged == 0
    assert llm.judge_calls == []
    assert await character_ids(driver) == {anna.id, anya.id}


async def test_reconcile_rejudges_when_profile_changed(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(same_pairs=[("Anna Petrova", "Anya")])
    anna, anya = (
        character("Anna Petrova", embedding=vec(1)),
        character("Anya", embedding=JUDGE_BAND),
    )
    await persist_in_order(repo, anna, anya)
    earlier = CharacterProfile(name="Anna Petrova", aliases=[], description="A stranger.")
    await record_not_same(repo, anna, anya, hash_a=profile_hash(earlier))

    report = await reconcile_characters(reconcile_deps(repo, redis, settings, llm))

    assert report.merges == [(anna.id, anya.id)]
    assert len(llm.judge_calls) == 1
    assert await character_ids(driver) == {anna.id}


async def test_reconcile_repairs_stale_embeddings(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    anna = character("Anna Petrova", embedding=vec(1))
    anya = character("Anya", embedding=vec(1))
    olga = character("Olga", embedding=vec(0, 0, 1))
    llm = FakeLLMClient(embeddings={"Anna Petrova;": vec(1), "Olga;": vec(0, 0, 0, 1)})
    await persist_in_order(repo, anna, anya, olga)
    await mark_stale(driver, anna, vec(0, 1))
    await mark_stale(driver, olga, vec(0, 0, 1))

    report = await reconcile_characters(reconcile_deps(repo, redis, settings, llm))

    assert report.reembedded == 2
    assert report.merges == [(anna.id, anya.id)]
    assert sorted(llm.embed_calls[0]) == sorted(
        [profile_embedding_text(profile_of(anna)), profile_embedding_text(profile_of(olga))]
    )
    records, _, _ = await driver.execute_query(
        "MATCH (c:Character {id: $id}) RETURN c.embedding AS embedding, c.embedding_stale AS stale",
        id=olga.id,
    )
    assert records[0]["stale"] is False
    assert records[0]["embedding"] == pytest.approx(vec(0, 0, 0, 1))


async def test_reconcile_keeps_auto_merging_after_judge_budget_is_spent(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(same_pairs=[("Anna Petrova", "Anya")])
    anna, anya = (
        character("Anna Petrova", embedding=vec(1)),
        character("Anya", embedding=JUDGE_BAND),
    )
    mark, markus = character("Mark", embedding=vec(0, 1)), character("Markus", embedding=vec(0, 1))
    await persist_in_order(repo, anna, anya, mark, markus)
    deps = reconcile_deps(repo, redis, settings, llm, reconcile_max_judge_calls=0)

    report = await reconcile_characters(deps)

    assert report.merges == [(mark.id, markus.id)]
    assert llm.judge_calls == []
    assert await character_ids(driver) == {anna.id, anya.id, mark.id}


async def test_reconcile_skips_pair_when_graph_is_busy(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    anna, anya = character("Anna Petrova", embedding=vec(1)), character("Anya", embedding=vec(1))
    mark, markus = character("Mark", embedding=vec(0, 1)), character("Markus", embedding=vec(0, 1))
    await persist_in_order(repo, anna, anya, mark, markus)
    await redis.set(RESOLVE_LOCK_NAME, b"ingestion-worker", px=10_000)
    deps = reconcile_deps(
        repo, redis, settings, FakeLLMClient(), merge_lock_acquire_timeout_seconds=0.2
    )

    with capture_logs() as logs:
        report = await reconcile_characters(deps)

    assert report.merged == 0
    assert await character_ids(driver) == {anna.id, anya.id, mark.id, markus.id}
    skipped = [entry for entry in logs if entry["event"] == "reconcile_merge_skipped"]
    assert [(entry["target_id"], entry["source_id"]) for entry in skipped] == [
        (anna.id, anya.id),
        (anna.id, anya.id),
        (mark.id, markus.id),
        (mark.id, markus.id),
    ]
    assert {entry["reason"] for entry in skipped} == {"resolve_lock_busy"}


async def test_reconcile_handles_many_duplicates_across_batches(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    annas = [character(f"Anna {n}", embedding=vec(1)) for n in range(4)]
    marks = [character(f"Mark {n}", embedding=vec(0, 1)) for n in range(3)]
    await persist_in_order(
        repo, annas[0], marks[0], annas[1], marks[1], annas[2], marks[2], annas[3]
    )
    llm = FakeLLMClient(embeddings={"Anna": vec(1), "Mark": vec(0, 1)})
    deps = reconcile_deps(repo, redis, settings, llm, reconcile_batch_size=2)

    report = await reconcile_characters(deps)

    assert (report.checked, report.merged) == (2, 5)
    assert await character_ids(driver) == {annas[0].id, marks[0].id}
    assert {target for target, _ in report.merges} == {annas[0].id, marks[0].id}


def two_judged_pairs() -> tuple[CharacterUpsert, ...]:
    return (
        character("Anna Petrova", embedding=vec(1)),
        character("Anya", embedding=JUDGE_BAND),
        character("Olga", embedding=vec(0, 0, 1)),
        character("Olya", embedding=vec(0, 0, 0.85, math.sqrt(1 - 0.85**2))),
    )


async def flag(driver: AsyncDriver, *upserts: CharacterUpsert) -> None:
    await driver.execute_query(
        "MATCH (c:Character) WHERE c.id IN $ids SET c.needs_reconcile = true",
        ids=[upsert.id for upsert in upserts],
    )


async def flagged_ids(driver: AsyncDriver) -> set[str]:
    records, _, _ = await driver.execute_query(
        "MATCH (c:Character) WHERE c.needs_reconcile = true RETURN c.id AS id"
    )
    return {record["id"] for record in records}


@pytest.mark.parametrize(
    ("same_pairs", "embedding", "merged"),
    [
        ([("Anna Petrova", "Anya")], JUDGE_BAND, True),
        ([], JUDGE_BAND, False),
        ([], vec(0, 1), False),
    ],
    ids=["judged-same", "judged-different", "no-band-candidates"],
)
async def test_reconcile_clears_flag_with_working_judge(
    repo: GraphRepository,
    driver: AsyncDriver,
    redis: Redis,
    settings: Settings,
    same_pairs: list[tuple[str, str]],
    embedding: list[float],
    merged: bool,
) -> None:
    llm = FakeLLMClient(same_pairs=same_pairs)
    anna, anya = character("Anna Petrova", embedding=vec(1)), character("Anya", embedding=embedding)
    await persist_in_order(repo, anna, anya)
    await flag(driver, anya)

    report = await reconcile_characters(reconcile_deps(repo, redis, settings, llm))

    assert report.merges == ([(anna.id, anya.id)] if merged else [])
    assert await character_ids(driver) == ({anna.id} if merged else {anna.id, anya.id})
    assert await flagged_ids(driver) == set()


async def test_reconcile_checks_flagged_nodes_first(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(same_pairs=[("Anna Petrova", "Anya"), ("Olga", "Olya")])
    anna, anya, olga, olya = two_judged_pairs()
    await persist_in_order(repo, anna, anya, olga, olya)
    await flag(driver, olya)
    deps = reconcile_deps(repo, redis, settings, llm, reconcile_max_judge_calls=1)

    report = await reconcile_characters(deps)

    assert [(c.name, e.name) for c, e in llm.judge_calls] == [("Olya", "Olga")]
    assert report.merges == [(olga.id, olya.id)]
    assert report.checked == 4
    assert await character_ids(driver) == {anna.id, anya.id, olga.id}
    assert await flagged_ids(driver) == set()


async def test_reconcile_judge_failure_keeps_flags_and_completes(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(
        same_pairs=[("Anna Petrova", "Anya"), ("Olga", "Olya")],
        judge_errors=[LLMError("DedupVerdict request failed: Error code: 429")],
    )
    anna, anya, olga, olya = two_judged_pairs()
    mark, markus = character("Mark", embedding=vec(0, 1)), character("Markus", embedding=vec(0, 1))
    await persist_in_order(repo, anna, anya, olga, olya, mark, markus)
    await flag(driver, anya, olya)

    with capture_logs() as logs:
        report = await reconcile_characters(reconcile_deps(repo, redis, settings, llm))

    assert len(llm.judge_calls) == 1
    assert report.merges == [(mark.id, markus.id)]
    assert await character_ids(driver) == {anna.id, anya.id, olga.id, olya.id, mark.id}
    assert await flagged_ids(driver) == {anya.id, olya.id}
    assert await count(driver, "(:Character)-[:NOT_SAME]->(:Character)") == 0
    [warning] = [entry for entry in logs if entry["event"] == "judge_unavailable"]
    assert warning["log_level"] == "warning"
    assert (warning["error_type"], warning["error_message"], warning["unjudged_characters"]) == (
        "LLMError",
        "DedupVerdict request failed: Error code: 429",
        4,
    )


async def test_reconcile_keeps_flag_when_judge_budget_is_spent(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(same_pairs=[("Anna Petrova", "Anya")])
    anna, anya = (
        character("Anna Petrova", embedding=vec(1)),
        character("Anya", embedding=JUDGE_BAND),
    )
    await persist_in_order(repo, anna, anya)
    await flag(driver, anya)
    deps = reconcile_deps(repo, redis, settings, llm, reconcile_max_judge_calls=0)

    report = await reconcile_characters(deps)

    assert (report.merges, llm.judge_calls) == ([], [])
    assert await flagged_ids(driver) == {anya.id}


async def test_reconcile_keeps_flag_when_graph_is_busy(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(same_pairs=[("Anna Petrova", "Anya")])
    anna, anya = (
        character("Anna Petrova", embedding=vec(1)),
        character("Anya", embedding=JUDGE_BAND),
    )
    await persist_in_order(repo, anna, anya)
    await flag(driver, anya)
    await redis.set(RESOLVE_LOCK_NAME, b"ingestion-worker", px=10_000)
    deps = reconcile_deps(repo, redis, settings, llm, merge_lock_acquire_timeout_seconds=0.2)

    report = await reconcile_characters(deps)

    assert report.merges == []
    assert await character_ids(driver) == {anna.id, anya.id}
    assert await flagged_ids(driver) == {anya.id}


async def test_reconcile_keeps_flag_when_known_different_blocks_auto_merge(
    repo: GraphRepository, driver: AsyncDriver, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(same_pairs=[("Anna Petrova", "Anya")])
    anna, anya = character("Anna Petrova", embedding=vec(1)), character("Anya", embedding=vec(1))
    await persist_in_order(repo, anna, anya)
    await record_not_same(repo, anna, anya)
    await flag(driver, anya)

    report = await reconcile_characters(reconcile_deps(repo, redis, settings, llm))

    assert (report.merges, llm.judge_calls) == ([], [])
    assert await flagged_ids(driver) == {anya.id}
