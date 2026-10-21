import asyncio
import hashlib
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
from neo4j import AsyncDriver
from redis.asyncio import Redis
from redis.asyncio.lock import Lock
from redis.exceptions import ConnectionError as RedisConnectionError
from structlog.testing import capture_logs

from app.config import Settings
from app.domain.models import (
    Article,
    ArticleFormat,
    ArticleStats,
    ArticleStatus,
    CharacterProfile,
    Direction,
)
from app.domain.profiles import profile_hash
from app.graph.repository import GraphRepository
from app.llm.client import (
    LLM_NOT_CONFIGURED,
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    UnconfiguredLLMClient,
)
from app.llm.schemas import DedupVerdict, ExtractedCharacter, ExtractedRelation, ExtractionResult
from app.pipeline.deps import PipelineDeps
from app.pipeline.lock import RESOLVE_LOCK_NAME, HeldLock
from app.pipeline.process import process_article
from app.storage import BlobStore
from tests.fakes import FakeLLMClient, vec

BORDERLINE = vec(0.85, (1 - 0.85**2) ** 0.5)
IVAN_A = vec(1, 0, 0)
IVAN_B = vec(0.866, 0.5, 0)
IVAN_BETWEEN = vec(1, 0.2, 0.5)
TEACHER = "Учитель из Твери."
SAILOR = "Моряк из Мурманска."


@pytest_asyncio.fixture
async def redis(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.redis_url)
    await client.delete(RESOLVE_LOCK_NAME)
    try:
        yield client
    finally:
        await client.aclose()


def tuned(settings: Settings, **changes: Any) -> Settings:
    return Settings(**{**settings.model_dump(), **changes})


def make_deps(
    repo: GraphRepository, redis: Redis, settings: Settings, llm: LLMClient
) -> PipelineDeps:
    return PipelineDeps(
        repo=repo, llm=llm, blobs=BlobStore(settings.upload_dir), redis=redis, settings=settings
    )


async def seed_blob(
    deps: PipelineDeps, data: bytes, *, article_format: ArticleFormat, title: str
) -> str:
    now = datetime.now(UTC)
    article = Article(
        id=str(uuid.uuid4()),
        title=title,
        format=article_format,
        content_hash=hashlib.sha256(data).hexdigest(),
        status=ArticleStatus.PENDING,
        created_at=now,
        updated_at=now,
    )
    await deps.repo.create_article(article)
    await deps.blobs.save(article.id, data)
    return article.id


async def seed_article(deps: PipelineDeps, text: str, *, title: str = "Untitled") -> str:
    return await seed_blob(deps, text.encode(), article_format=ArticleFormat.MARKDOWN, title=title)


async def ingest(deps: PipelineDeps, text: str, *, title: str = "Untitled") -> str:
    article_id = await seed_article(deps, text, title=title)
    await process_article(article_id, deps)
    return article_id


def person(name: str, description: str = "", aliases: Sequence[str] = ()) -> ExtractedCharacter:
    return ExtractedCharacter(
        name=name, aliases=list(aliases), description=description or f"{name} from the story."
    )


def link(source: str, target: str, type_: str) -> ExtractedRelation:
    return ExtractedRelation(
        source=source,
        target=target,
        type=type_,
        description=f"{source} {type_} {target}",
        evidence=f"A quote about {source} and {target}.",
    )


def extraction(
    *characters: ExtractedCharacter, relations: Sequence[ExtractedRelation] = ()
) -> ExtractionResult:
    return ExtractionResult(characters=list(characters), relations=list(relations))


async def scalar(driver: AsyncDriver, query: str, **params: Any) -> Any:
    records, _, _ = await driver.execute_query(query, **params)
    return records[0][0]


async def character_count(driver: AsyncDriver) -> int:
    return int(await scalar(driver, "MATCH (c:Character) RETURN count(c)"))


async def appearances(driver: AsyncDriver, article_id: str) -> dict[str, tuple[str, float | None]]:
    records, _, _ = await driver.execute_query(
        "MATCH (c:Character)-[m:APPEARS_IN]->(:Article {id: $id}) "
        "RETURN c.id AS id, m.match AS match, m.similarity AS similarity",
        id=article_id,
    )
    return {record["id"]: (record["match"], record["similarity"]) for record in records}


async def status_of(repo: GraphRepository, article_id: str) -> Article:
    article = await repo.get_article(article_id)
    assert article is not None
    return article


async def stored_profile(repo: GraphRepository, character_id: str) -> CharacterProfile:
    character = await repo.get_character(character_id)
    assert character is not None
    return CharacterProfile(
        name=character.name, aliases=character.aliases, description=character.description
    )


async def test_markdown_article_persisted(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(
        extractions={
            "lighthouse": extraction(
                person("Anna Petrova"),
                person("Mark Petrov"),
                relations=[link("Anna Petrova", "Mark Petrov", "SIBLING_OF")],
            )
        },
        embeddings={"Anna Petrova": vec(1), "Mark Petrov": vec(0, 1)},
    )
    deps = make_deps(repo, redis, settings, llm)
    article_id = await seed_article(deps, "# The Lighthouse\n\nAnna and Mark kept the lighthouse.")

    await process_article(article_id, deps)

    article = await status_of(repo, article_id)
    assert article.status is ArticleStatus.DONE
    assert article.error is None
    assert article.stats == ArticleStats(characters=2, relations=1, images_described=0)
    characters = await repo.list_characters_by_article(article_id)
    assert [c.name for c in characters] == ["Anna Petrova", "Mark Petrov"]
    assert [(t.name, t.description, t.usage_count) for t in await repo.list_relation_types()] == [
        ("SIBLING_OF", "sibling of", 1)
    ]
    with pytest.raises(FileNotFoundError):
        await deps.blobs.load(article_id)


async def test_existing_relation_type_is_reused_without_embedding(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(
                person("Anna"), person("Mark"), relations=[link("Anna", "Mark", "FRIEND_OF")]
            ),
            "second": extraction(
                person("Lena"), person("Oleg"), relations=[link("Lena", "Oleg", "friend of")]
            ),
        },
        embeddings={
            "Anna": vec(1),
            "Mark": vec(0, 1),
            "Lena": vec(0, 0, 1),
            "Oleg": vec(0, 0, 0, 1),
        },
    )
    deps = make_deps(repo, redis, settings, llm)
    await ingest(deps, "first")
    llm.embed_calls.clear()

    await ingest(deps, "second")

    assert [(t.name, t.usage_count) for t in await repo.list_relation_types()] == [("FRIEND_OF", 2)]
    assert not any(text.startswith("FRIEND_OF") for call in llm.embed_calls for text in call)


async def test_character_deduplicated_across_articles_by_vector(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(person("Anna Petrova", "A lighthouse keeper.")),
            "second": extraction(person("Анна Петрова", "Смотритель маяка.")),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Анна Петрова": vec(1, 0)},
    )
    deps = make_deps(repo, redis, settings, llm)

    await ingest(deps, "first")
    second_id = await ingest(deps, "second")

    assert await character_count(driver) == 1
    [anna] = await repo.list_characters_by_article(second_id)
    assert anna.name == "Anna Petrova"
    assert "Анна Петрова" in anna.aliases
    assert anna.article_count == 2
    assert llm.judge_calls == []
    [(match, similarity)] = (await appearances(driver, second_id)).values()
    assert match == "vector"
    assert similarity == pytest.approx(1.0, abs=1e-3)
    assert await scalar(driver, "MATCH (c:Character) RETURN c.embedding_stale") is False


@pytest.mark.parametrize(
    ("same_pairs", "expected_nodes"), [([("Anna Petrova", "Anya")], 1), ([], 2)]
)
async def test_borderline_similarity_asks_judge(
    repo: GraphRepository,
    redis: Redis,
    settings: Settings,
    driver: AsyncDriver,
    same_pairs: list[tuple[str, str]],
    expected_nodes: int,
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(person("Anna Petrova")),
            "second": extraction(person("Anya")),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Anya": BORDERLINE},
        same_pairs=same_pairs,
    )
    deps = make_deps(repo, redis, settings, llm)

    await ingest(deps, "first")
    second_id = await ingest(deps, "second")

    assert await character_count(driver) == expected_nodes
    assert len(llm.judge_calls) == 1
    [(match, similarity)] = (await appearances(driver, second_id)).values()
    assert match == ("judge" if expected_nodes == 1 else "new")
    assert similarity == pytest.approx(0.85, abs=1e-3)


async def test_judge_receives_graph_context(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(
                person("Anna Petrova"),
                person("Mark Petrov"),
                relations=[link("Anna Petrova", "Mark Petrov", "SIBLING_OF")],
            ),
            "second": extraction(
                person("Anya"), person("Pyotr"), relations=[link("Pyotr", "Anya", "EMPLOYS")]
            ),
        },
        embeddings={
            "Anna Petrova": vec(1, 0),
            "Mark Petrov": vec(0, 1),
            "Anya": BORDERLINE,
            "Pyotr": vec(0, 0, 1),
        },
    )
    deps = make_deps(repo, redis, settings, llm)
    await ingest(deps, "first", title="The Lighthouse")

    await ingest(deps, "second", title="The Harbour")

    [(candidate, existing)] = llm.judge_calls
    assert (candidate.name, candidate.relations, candidate.articles) == (
        "Anya",
        ["← EMPLOYS Pyotr"],
        ["The Harbour"],
    )
    assert (existing.name, existing.relations, existing.articles) == (
        "Anna Petrova",
        ["SIBLING_OF → Mark Petrov"],
        ["The Lighthouse"],
    )


async def test_same_first_name_different_people_kept_apart(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(person("Иван", TEACHER)),
            "second": extraction(person("Иван", SAILOR)),
        },
        embeddings={"Иван; aliases: none; Учитель": IVAN_A, "Иван; aliases: none; Моряк": IVAN_B},
    )
    deps = make_deps(repo, redis, settings, llm)

    await ingest(deps, "first")
    second_id = await ingest(deps, "second")

    assert await character_count(driver) == 2
    assert len(llm.judge_calls) == 1
    [(match, _)] = (await appearances(driver, second_id)).values()
    assert match == "new"


async def test_low_similarity_creates_new_character(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FakeLLMClient(
        extractions={"first": extraction(person("Anna")), "second": extraction(person("Boris"))},
        embeddings={"Anna": vec(1, 0), "Boris": vec(0, 1)},
    )
    deps = make_deps(repo, redis, settings, llm)

    await ingest(deps, "first")
    await ingest(deps, "second")

    assert await character_count(driver) == 2
    assert llm.judge_calls == []


async def test_opposite_types_never_merged(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(
                person("Anna"), person("Mark"), relations=[link("Anna", "Mark", "PARENT_OF")]
            ),
            "second": extraction(
                person("Lena"), person("Oleg"), relations=[link("Lena", "Oleg", "CHILD_OF")]
            ),
        },
        embeddings={
            "Anna": vec(1),
            "Mark": vec(0, 1),
            "Lena": vec(0, 0, 1),
            "Oleg": vec(0, 0, 0, 1),
            "PARENT_OF": vec(0, 0, 0, 0, 1),
            "CHILD_OF": vec(0, 0, 0, 0, 1),
        },
    )
    deps = make_deps(repo, redis, settings, llm)

    await ingest(deps, "first")
    await ingest(deps, "second")

    stored_types = await scalar(driver, "MATCH ()-[r:RELATED_TO]->() RETURN collect(r.type)")
    assert sorted(stored_types) == ["CHILD_OF", "PARENT_OF"]
    assert sorted(t.name for t in await repo.list_relation_types()) == ["CHILD_OF", "PARENT_OF"]


async def test_same_chunk_characters_never_fold_into_one_existing_node(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(person("Anna Petrova", "A keeper.")),
            "second": extraction(
                person("Anya", "Anna's nickname in the harbour."),
                person("Анна Петрова", "Смотритель маяка на севере."),
                relations=[link("Anya", "Анна Петрова", "FRIEND_OF")],
            ),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Анна Петрова": vec(1, 0), "Anya": BORDERLINE},
        same_pairs=[("Anya", "Anna Petrova")],
    )
    deps = make_deps(repo, redis, settings, llm)
    await ingest(deps, "first")

    second_id = await ingest(deps, "second")

    assert await character_count(driver) == 2
    anna, russian = await repo.list_characters_by_article(second_id)
    assert (anna.name, anna.aliases, anna.article_count) == ("Anna Petrova", ["Anya"], 2)
    assert (russian.name, russian.article_count) == ("Анна Петрова", 1)
    matches = await appearances(driver, second_id)
    assert (matches[anna.id][0], matches[russian.id]) == ("judge", ("new", None))
    assert (await status_of(repo, second_id)).stats == ArticleStats(
        characters=2, relations=1, images_described=0
    )
    [connection] = await repo.find_connected(anna.id, "FRIEND_OF")
    assert (connection.character.id, connection.direction) == (russian.id, Direction.OUTGOING)


async def test_characters_from_different_chunks_still_fold_into_one_existing_node(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(person("Anna Petrova", "A keeper.")),
            "harbour chapter": extraction(person("Anya", "Anna's nickname in the harbour.")),
            "russian chapter": extraction(person("Анна Петрова", "Смотритель маяка.")),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Анна Петрова": vec(1, 0), "Anya": BORDERLINE},
        same_pairs=[("Anya", "Anna Petrova")],
    )
    two_chunks = tuned(settings, chunk_max_chars=20, chunk_overlap_chars=0)
    deps = make_deps(repo, redis, two_chunks, llm)
    await ingest(deps, "first")

    second_id = await ingest(deps, "harbour chapter.\n\nrussian chapter.")

    assert len(llm.extract_calls) == 3
    assert await character_count(driver) == 1
    [anna] = await repo.list_characters_by_article(second_id)
    assert (anna.name, anna.aliases) == ("Anna Petrova", ["Anya", "Анна Петрова"])
    assert sorted(match for match, _ in (await appearances(driver, second_id)).values()) == [
        "vector"
    ]


async def test_concurrent_articles_do_not_duplicate_new_character(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FakeLLMClient(
        extractions={"lena": extraction(person("Lena Ivanova"))},
        embeddings={"Lena Ivanova": vec(0, 0, 1)},
    )
    deps = make_deps(repo, redis, settings, llm)
    first_id = await seed_article(deps, "lena at the market")
    second_id = await seed_article(deps, "lena at the station")

    await asyncio.gather(process_article(first_id, deps), process_article(second_id, deps))

    assert await character_count(driver) == 1
    [lena] = await repo.list_characters_by_article(first_id)
    assert lena.article_count == 2


class SlowJudgeLLM(FakeLLMClient):
    def __init__(self, delay: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.delay = delay

    async def judge_same_character(
        self, candidate: CharacterProfile, existing: CharacterProfile
    ) -> DedupVerdict:
        await asyncio.sleep(self.delay)
        return await super().judge_same_character(candidate, existing)


async def test_slow_judge_keeps_lock_and_serializes(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    lock_timeout = 0.6
    fast = tuned(
        settings,
        resolve_lock_timeout_seconds=lock_timeout,
        llm_timeout_seconds=0.1,
        dedup_max_judge_calls_per_article=2,
    )
    llm = SlowJudgeLLM(
        delay=lock_timeout * 1.5,
        extractions={
            "anna": extraction(person("Anna Petrova")),
            "lena": extraction(person("Lena")),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Lena": BORDERLINE},
    )
    deps = make_deps(repo, redis, fast, llm)
    await ingest(deps, "anna")
    first_id = await seed_article(deps, "lena at the market")
    second_id = await seed_article(deps, "lena at the station")

    await asyncio.gather(process_article(first_id, deps), process_article(second_id, deps))

    assert [(await status_of(repo, i)).status for i in (first_id, second_id)] == [
        ArticleStatus.DONE,
        ArticleStatus.DONE,
    ]
    assert await character_count(driver) == 2
    assert len(llm.judge_calls) == 1


class LockObservingLLM(FakeLLMClient):
    def __init__(self, redis: Redis, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.redis = redis
        self.events: list[tuple[str, bool]] = []

    async def _observe(self, kind: str) -> None:
        self.events.append((kind, bool(await self.redis.exists(RESOLVE_LOCK_NAME))))

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        await self._observe("embed")
        return await super().embed(texts)

    async def judge_same_character(
        self, candidate: CharacterProfile, existing: CharacterProfile
    ) -> DedupVerdict:
        await self._observe("judge")
        return await super().judge_same_character(candidate, existing)


async def test_embeddings_computed_before_lock(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = LockObservingLLM(
        redis,
        extractions={
            "first": extraction(person("Anna Petrova")),
            "second": extraction(
                person("Anya"), person("Pyotr"), relations=[link("Anya", "Pyotr", "MENTOR_OF")]
            ),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Anya": BORDERLINE, "Pyotr": vec(0, 0, 1)},
        same_pairs=[("Anya", "Anna Petrova")],
    )
    deps = make_deps(repo, redis, settings, llm)
    await ingest(deps, "first")
    llm.events.clear()
    llm.embed_calls.clear()

    await ingest(deps, "second")

    assert llm.events == [("embed", False), ("embed", False), ("judge", True), ("embed", False)]
    profiles, types, refreshed = llm.embed_calls
    assert [text.split(";")[0] for text in profiles] == ["Anya", "Pyotr"]
    assert types == ["MENTOR_OF: mentor of"]
    assert [text.split(";")[0] for text in refreshed] == ["Anna Petrova"]


async def test_judge_cap_creates_new_characters(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(person("Anna Petrova"), person("Mark Petrov")),
            "second": extraction(person("Anya"), person("Marik")),
        },
        embeddings={
            "Anna Petrova": vec(1, 0, 0),
            "Mark Petrov": vec(0, 0, 1),
            "Anya": BORDERLINE,
            "Marik": vec(0, 0, 0.85, (1 - 0.85**2) ** 0.5),
        },
        same_pairs=[("Anya", "Anna Petrova"), ("Marik", "Mark Petrov")],
    )
    deps = make_deps(repo, redis, tuned(settings, dedup_max_judge_calls_per_article=1), llm)
    await ingest(deps, "first")

    with capture_logs() as logs:
        second_id = await ingest(deps, "second")

    assert [(c.name, e.name) for c, e in llm.judge_calls] == [("Anya", "Anna Petrova")]
    assert await character_count(driver) == 3
    assert sorted(match for match, _ in (await appearances(driver, second_id)).values()) == [
        "judge",
        "new",
    ]
    assert [entry["event"] for entry in logs].count("judge_cap_reached") == 1
    assert "judge_unavailable" not in [entry["event"] for entry in logs]
    assert await needs_reconcile(driver, (await ids_by_name(repo, second_id))["Marik"]) is True
    stats = (await status_of(repo, second_id)).stats
    assert stats is not None
    assert stats.unjudged_characters == 1


RATE_LIMITED = LLMError("DedupVerdict request failed: Error code: 429")
MARIK = vec(0, 0, 0.85, (1 - 0.85**2) ** 0.5)


async def needs_reconcile(driver: AsyncDriver, character_id: str) -> bool | None:
    flag = await scalar(
        driver, "MATCH (c:Character {id: $id}) RETURN c.needs_reconcile", id=character_id
    )
    return None if flag is None else bool(flag)


async def not_same_count(driver: AsyncDriver) -> int:
    return int(await scalar(driver, "MATCH ()-[n:NOT_SAME]->() RETURN count(n)"))


def judge_warnings(logs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [entry for entry in logs if entry["event"] == "judge_unavailable"]


def two_borderline_llm(
    *, judge_errors: Sequence[LLMError | None], same_pairs: Sequence[tuple[str, str]] = ()
) -> FakeLLMClient:
    return FakeLLMClient(
        extractions={
            "first": extraction(person("Anna Petrova"), person("Mark Petrov")),
            "second": extraction(person("Anya"), person("Marik")),
        },
        embeddings={
            "Anna Petrova": vec(1, 0, 0),
            "Mark Petrov": vec(0, 0, 1),
            "Anya": BORDERLINE,
            "Marik": MARIK,
        },
        same_pairs=same_pairs,
        judge_errors=judge_errors,
    )


async def ids_by_name(repo: GraphRepository, article_id: str) -> dict[str, str]:
    return {c.name: c.id for c in await repo.list_characters_by_article(article_id)}


async def test_judge_failure_does_not_fail_article(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(person("Anna Petrova")),
            "second": extraction(person("Anya")),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Anya": BORDERLINE},
        same_pairs=[("Anya", "Anna Petrova")],
        judge_errors=[RATE_LIMITED],
    )
    deps = make_deps(repo, redis, settings, llm)
    first_id = await ingest(deps, "first")

    with capture_logs() as logs:
        second_id = await ingest(deps, "second")

    article = await status_of(repo, second_id)
    assert (article.status, article.error) == (ArticleStatus.DONE, None)
    assert article.stats == ArticleStats(
        characters=1, relations=0, images_described=0, unjudged_characters=1
    )
    [anna] = await repo.list_characters_by_article(first_id)
    [anya] = await repo.list_characters_by_article(second_id)
    assert (anya.name, anya.article_count) == ("Anya", 1)
    assert await character_count(driver) == 2
    assert [match for match, _ in (await appearances(driver, second_id)).values()] == ["new"]
    assert await needs_reconcile(driver, anya.id) is True
    assert await needs_reconcile(driver, anna.id) is False
    assert await not_same_count(driver) == 0
    [warning] = judge_warnings(logs)
    assert warning["log_level"] == "warning"
    assert (
        warning["article_id"],
        warning["error_type"],
        warning["error_message"],
        warning["unjudged_characters"],
    ) == (second_id, "LLMError", str(RATE_LIMITED), 1)


async def test_judge_unavailable_skips_further_judge_calls(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = two_borderline_llm(judge_errors=[RATE_LIMITED])
    deps = make_deps(repo, redis, settings, llm)
    await ingest(deps, "first")

    with capture_logs() as logs:
        second_id = await ingest(deps, "second")

    assert len(llm.judge_calls) == 1
    assert await character_count(driver) == 4
    article = await status_of(repo, second_id)
    assert article.status is ArticleStatus.DONE
    assert article.stats is not None
    assert article.stats.unjudged_characters == 2
    for character_id in (await ids_by_name(repo, second_id)).values():
        assert await needs_reconcile(driver, character_id) is True
    [warning] = judge_warnings(logs)
    assert warning["unjudged_characters"] == 2
    assert warning["error_message"] == str(RATE_LIMITED)
    assert "judge_cap_reached" not in [entry["event"] for entry in logs]


async def test_auto_merge_still_applies_when_judge_unavailable(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(person("Anna Petrova"), person("Mark Petrov")),
            "second": extraction(person("Anya"), person("Марк Петров")),
        },
        embeddings={
            "Anna Petrova": vec(1, 0, 0),
            "Mark Petrov": vec(0, 0, 1),
            "Anya": BORDERLINE,
            "Марк Петров": vec(0, 0, 1),
        },
        judge_errors=[RATE_LIMITED],
    )
    deps = make_deps(repo, redis, settings, llm)
    first_id = await ingest(deps, "first")

    second_id = await ingest(deps, "second")

    assert len(llm.judge_calls) == 1
    mark_id = (await ids_by_name(repo, first_id))["Mark Petrov"]
    second = await ids_by_name(repo, second_id)
    assert second["Mark Petrov"] == mark_id
    matches = await appearances(driver, second_id)
    assert (matches[mark_id][0], matches[second["Anya"]][0]) == ("vector", "new")
    assert await needs_reconcile(driver, mark_id) is False
    assert await needs_reconcile(driver, second["Anya"]) is True
    article = await status_of(repo, second_id)
    assert article.stats == ArticleStats(
        characters=2, relations=0, images_described=0, unjudged_characters=1
    )


async def test_verdicts_before_failure_are_kept(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = two_borderline_llm(
        judge_errors=[None, RATE_LIMITED], same_pairs=[("Anya", "Anna Petrova")]
    )
    deps = make_deps(repo, redis, settings, llm)
    first_id = await ingest(deps, "first")

    second_id = await ingest(deps, "second")

    assert [(c.name, e.name) for c, e in llm.judge_calls] == [
        ("Anya", "Anna Petrova"),
        ("Marik", "Mark Petrov"),
    ]
    first = await ids_by_name(repo, first_id)
    second = await ids_by_name(repo, second_id)
    assert second["Anna Petrova"] == first["Anna Petrova"]
    assert second["Marik"] != first["Mark Petrov"]
    matches = await appearances(driver, second_id)
    assert (matches[second["Anna Petrova"]][0], matches[second["Marik"]][0]) == ("judge", "new")
    assert await needs_reconcile(driver, second["Anna Petrova"]) is False
    assert await needs_reconcile(driver, second["Marik"]) is True
    article = await status_of(repo, second_id)
    assert article.status is ArticleStatus.DONE
    assert article.stats is not None
    assert article.stats.unjudged_characters == 1
    assert await not_same_count(driver) == 0


async def test_rejection_before_failure_is_recorded(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = two_borderline_llm(judge_errors=[None, RATE_LIMITED])
    deps = make_deps(repo, redis, settings, llm)
    first_id = await ingest(deps, "first")

    second_id = await ingest(deps, "second")

    first = await ids_by_name(repo, first_id)
    second = await ids_by_name(repo, second_id)
    records, _, _ = await driver.execute_query(
        "MATCH (a:Character)-[:NOT_SAME]-(b:Character) WHERE a.id < b.id RETURN a.id, b.id"
    )
    assert [set(record.values()) for record in records] == [{first["Anna Petrova"], second["Anya"]}]
    assert await needs_reconcile(driver, second["Anya"]) is False
    assert await needs_reconcile(driver, second["Marik"]) is True


async def test_judge_error_message_is_truncated_in_warning(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    long_error = LLMError("x" * 500)
    llm = two_borderline_llm(judge_errors=[long_error])
    deps = make_deps(repo, redis, settings, llm)
    await ingest(deps, "first")

    with capture_logs() as logs:
        await ingest(deps, "second")

    [warning] = judge_warnings(logs)
    assert warning["error_message"] == "x" * 200


async def test_ingestion_keeps_flag_of_matched_existing_node(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(person("Anna Petrova")),
            "second": extraction(person("Анна Петрова")),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Анна Петрова": vec(1, 0)},
    )
    deps = make_deps(repo, redis, settings, llm)
    first_id = await ingest(deps, "first")
    [anna] = await repo.list_characters_by_article(first_id)
    await driver.execute_query(
        "MATCH (c:Character {id: $id}) SET c.needs_reconcile = true", id=anna.id
    )

    second_id = await ingest(deps, "second")

    assert [c.id for c in await repo.list_characters_by_article(second_id)] == [anna.id]
    assert await needs_reconcile(driver, anna.id) is True


async def test_judge_not_configured_still_fails_article(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(person("Anna Petrova")),
            "second": extraction(person("Anya")),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Anya": BORDERLINE},
        judge_errors=[LLMNotConfiguredError(LLM_NOT_CONFIGURED)],
    )
    deps = make_deps(repo, redis, settings, llm)
    await ingest(deps, "first")

    second_id = await ingest(deps, "second")

    article = await status_of(repo, second_id)
    assert article.status is ArticleStatus.FAILED
    assert article.error == f"LLMNotConfiguredError: {LLM_NOT_CONFIGURED}"


class FailingRefreshLLM(FakeLLMClient):
    def __init__(self, failing_prefix: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.failing_prefix = failing_prefix

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if any(text.startswith(self.failing_prefix) for text in texts):
            raise LLMError("embedding request failed: boom")
        return await super().embed(texts)


async def test_reembed_failure_keeps_article_done_and_flags_node(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = FailingRefreshLLM(
        "Anna Petrova; aliases: Анна Петрова",
        extractions={
            "first": extraction(person("Anna Petrova")),
            "second": extraction(person("Анна Петрова")),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Анна Петрова": vec(1, 0)},
    )
    deps = make_deps(repo, redis, settings, llm)
    await ingest(deps, "first")

    with capture_logs() as logs:
        second_id = await ingest(deps, "second")

    assert (await status_of(repo, second_id)).status is ArticleStatus.DONE
    [anna] = await repo.list_characters_by_article(second_id)
    assert anna.aliases == ["Анна Петрова"]
    records, _, _ = await driver.execute_query(
        "MATCH (c:Character {id: $id}) RETURN c.embedding_stale AS stale, c.embedding AS e",
        id=anna.id,
    )
    assert records[0]["stale"] is True
    assert records[0]["e"] == pytest.approx(vec(1, 0))
    [failure] = [entry for entry in logs if entry["event"] == "character_reembed_failed"]
    assert failure["character_ids"] == [anna.id]


def ivan_llm(
    extra_extractions: Mapping[str, ExtractionResult] | None = None,
    extra_embeddings: Mapping[str, list[float]] | None = None,
    same_pairs: Sequence[tuple[str, str]] = (),
) -> FakeLLMClient:
    return FakeLLMClient(
        extractions={
            "article-one": extraction(person("Иван", TEACHER)),
            "article-two": extraction(person("Иван", SAILOR)),
            **(extra_extractions or {}),
        },
        embeddings={
            **(extra_embeddings or {}),
            "Иван; aliases: none; Учитель": IVAN_A,
            "Иван; aliases: Ваня; Учитель": IVAN_A,
            "Иван; aliases: none; Моряк": IVAN_B,
        },
        same_pairs=same_pairs,
    )


async def ingest_two_ivans(deps: PipelineDeps) -> tuple[str, str]:
    first_id = await ingest(deps, "article-one")
    second_id = await ingest(deps, "article-two")
    [ivan_a] = await deps.repo.list_characters_by_article(first_id)
    [ivan_b] = await deps.repo.list_characters_by_article(second_id)
    assert ivan_a.id != ivan_b.id
    return ivan_a.id, ivan_b.id


def judged_descriptions(llm: FakeLLMClient) -> list[str]:
    return [existing.description for _, existing in llm.judge_calls]


async def test_not_same_recorded_on_negative_verdict(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    deps = make_deps(repo, redis, settings, ivan_llm())

    ivan_a, ivan_b = await ingest_two_ivans(deps)

    records, _, _ = await driver.execute_query(
        "MATCH (low:Character)-[n:NOT_SAME]->(high:Character) "
        "RETURN low.id AS low, high.id AS high, n.hash_low AS hash_low, "
        "n.hash_high AS hash_high, n.judged_at AS judged_at, n.reason AS reason"
    )
    [edge] = records
    low, high = sorted([ivan_a, ivan_b])
    assert (edge["low"], edge["high"]) == (low, high)
    assert edge["hash_low"] == profile_hash(await stored_profile(repo, low))
    assert edge["hash_high"] == profile_hash(await stored_profile(repo, high))
    assert edge["judged_at"] is not None
    assert edge["reason"] == "not configured"


async def test_identical_profile_already_judged_different_skips_judge(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = ivan_llm(
        extra_extractions={"article-three": extraction(person("иван", SAILOR))},
        extra_embeddings={"иван;": IVAN_BETWEEN},
        same_pairs=[("иван", "Иван")],
    )
    deps = make_deps(repo, redis, settings, llm)
    _, ivan_b = await ingest_two_ivans(deps)
    candidate = CharacterProfile(name="иван", aliases=[], description=SAILOR)
    assert profile_hash(candidate) == profile_hash(await stored_profile(repo, ivan_b))
    llm.judge_calls.clear()

    third_id = await ingest(deps, "article-three")

    assert judged_descriptions(llm) == [SAILOR]
    assert list(await appearances(driver, third_id)) == [ivan_b]
    assert await character_count(driver) == 2


async def test_changed_candidate_profile_is_rejudged(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = ivan_llm(
        extra_extractions={"article-three": extraction(person("Иван", "Пекарь из Пскова."))},
        extra_embeddings={"Иван; aliases: none; Пекарь": IVAN_BETWEEN},
    )
    deps = make_deps(repo, redis, settings, llm)
    await ingest_two_ivans(deps)
    llm.judge_calls.clear()

    await ingest(deps, "article-three")

    assert judged_descriptions(llm) == [TEACHER, SAILOR]


async def test_stale_verdict_after_existing_profile_changed(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = ivan_llm(
        extra_extractions={
            "article-alias": extraction(person("Иван", TEACHER, aliases=["Ваня"])),
            "article-three": extraction(person("иван", SAILOR)),
        },
        extra_embeddings={"иван;": IVAN_BETWEEN},
    )
    deps = make_deps(repo, redis, settings, llm)
    ivan_a, _ = await ingest_two_ivans(deps)
    alias_id = await ingest(deps, "article-alias")
    assert [c.id for c in await repo.list_characters_by_article(alias_id)] == [ivan_a]
    assert (await stored_profile(repo, ivan_a)).aliases == ["Ваня"]
    llm.judge_calls.clear()

    await ingest(deps, "article-three")

    assert judged_descriptions(llm)[0] == TEACHER


async def test_not_same_does_not_block_auto_merge(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = ivan_llm(
        extra_extractions={"article-three": extraction(person("иван", SAILOR))},
        extra_embeddings={"иван;": IVAN_A},
    )
    deps = make_deps(repo, redis, settings, llm)
    ivan_a, _ = await ingest_two_ivans(deps)
    llm.judge_calls.clear()

    third_id = await ingest(deps, "article-three")

    assert llm.judge_calls == []
    assert {key: match for key, (match, _) in (await appearances(driver, third_id)).items()} == {
        ivan_a: "vector"
    }


async def test_parse_failure_marks_article_failed(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    deps = make_deps(repo, redis, settings, FakeLLMClient())
    article_id = await seed_blob(
        deps, b"%PDF-1.7 not really a pdf", article_format=ArticleFormat.PDF, title="Broken"
    )

    await process_article(article_id, deps)

    article = await status_of(repo, article_id)
    assert article.status is ArticleStatus.FAILED
    assert article.error is not None
    assert article.error.startswith("DocumentParseError")


async def test_blob_deleted_on_failure(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    deps = make_deps(repo, redis, settings, FakeLLMClient())
    article_id = await seed_blob(
        deps, b"%PDF-1.7 not really a pdf", article_format=ArticleFormat.PDF, title="Broken"
    )

    await process_article(article_id, deps)

    assert (await status_of(repo, article_id)).status is ArticleStatus.FAILED
    with pytest.raises(FileNotFoundError):
        await deps.blobs.load(article_id)


async def test_llm_not_configured_marks_failed(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    deps = make_deps(repo, redis, settings, UnconfiguredLLMClient())
    article_id = await seed_article(deps, "# Title\n\nSome text about Anna.")

    await process_article(article_id, deps)

    article = await status_of(repo, article_id)
    assert article.status is ArticleStatus.FAILED
    assert article.error is not None
    assert LLM_NOT_CONFIGURED in article.error


async def test_unknown_article_is_ignored(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient()
    deps = make_deps(repo, redis, settings, llm)
    await deps.blobs.save("missing", b"orphan")

    with capture_logs() as logs:
        await process_article("missing", deps)

    assert [entry["event"] for entry in logs] == ["article_not_found"]
    assert llm.extract_calls == []
    with pytest.raises(FileNotFoundError):
        await deps.blobs.load("missing")


async def test_done_article_is_not_reprocessed(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(extractions={"anna": extraction(person("Anna"))})
    deps = make_deps(repo, redis, settings, llm)
    article_id = await ingest(deps, "anna")
    await deps.blobs.save(article_id, b"anna again")
    llm.extract_calls.clear()

    with capture_logs() as logs:
        await process_article(article_id, deps)

    assert llm.extract_calls == []
    assert [entry["event"] for entry in logs] == ["article_already_done"]
    assert (await status_of(repo, article_id)).status is ArticleStatus.DONE
    with pytest.raises(FileNotFoundError):
        await deps.blobs.load(article_id)


async def test_article_finished_after_the_status_read_is_not_reprocessed(
    repo: GraphRepository, redis: Redis, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = FakeLLMClient(extractions={"anna": extraction(person("Anna"))})
    deps = make_deps(repo, redis, settings, llm)
    article_id = await ingest(deps, "anna")
    done = await status_of(repo, article_id)
    stale = done.model_copy(update={"status": ArticleStatus.PENDING, "stats": None})
    await deps.blobs.save(article_id, b"anna again")
    llm.extract_calls.clear()

    async def stale_read(requested_id: str) -> Article | None:
        return stale if requested_id == article_id else None

    monkeypatch.setattr(repo, "get_article", stale_read)
    with capture_logs() as logs:
        await process_article(article_id, deps)
    monkeypatch.undo()

    assert llm.extract_calls == []
    assert [entry["event"] for entry in logs] == ["article_already_done"]
    assert await status_of(repo, article_id) == done
    with pytest.raises(FileNotFoundError):
        await deps.blobs.load(article_id)


async def test_article_with_too_many_chunks_fails_before_extraction(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(extractions={"anna": extraction(person("Anna"))})
    limited = tuned(settings, chunk_max_chars=12, chunk_overlap_chars=0, max_chunks_per_article=2)
    deps = make_deps(repo, redis, limited, llm)

    article_id = await ingest(deps, "anna one.\n\nanna two.\n\nanna three.")

    article = await status_of(repo, article_id)
    assert article.status is ArticleStatus.FAILED
    assert article.error == (
        "ArticleTooLongError: article has 3 chunks, more than MAX_CHUNKS_PER_ARTICLE=2"
    )
    assert llm.extract_calls == []
    assert llm.embed_calls == []


class LockStealingJudgeLLM(FakeLLMClient):
    def __init__(self, redis: Redis, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.redis = redis

    async def judge_same_character(
        self, candidate: CharacterProfile, existing: CharacterProfile
    ) -> DedupVerdict:
        await self.redis.delete(RESOLVE_LOCK_NAME)
        await asyncio.sleep(0.05)
        return await super().judge_same_character(candidate, existing)


async def test_lost_lock_aborts_before_persist(
    repo: GraphRepository, redis: Redis, settings: Settings, driver: AsyncDriver
) -> None:
    llm = LockStealingJudgeLLM(
        redis,
        extractions={
            "first": extraction(person("Anna Petrova")),
            "second": extraction(person("Anya"), person("Pyotr")),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Anya": BORDERLINE, "Pyotr": vec(0, 0, 1)},
        same_pairs=[("Anya", "Anna Petrova")],
    )
    deps = make_deps(repo, redis, settings, llm)
    first_id = await ingest(deps, "first")

    with capture_logs() as logs:
        second_id = await ingest(deps, "second")

    article = await status_of(repo, second_id)
    assert article.status is ArticleStatus.FAILED
    assert article.error is not None
    assert article.error.startswith("ResolveLockLostError")
    assert len(llm.judge_calls) == 1
    assert await repo.list_characters_by_article(second_id) == []
    assert await character_count(driver) == 1
    [anna] = await repo.list_characters_by_article(first_id)
    assert anna.aliases == []
    assert await scalar(driver, "MATCH ()-[n:NOT_SAME]->() RETURN count(n)") == 0
    with pytest.raises(FileNotFoundError):
        await deps.blobs.load(second_id)
    lost = [entry for entry in logs if entry["event"] == "resolve_lock_lost"]
    assert [entry["article_id"] for entry in lost] == [second_id]


async def test_release_connection_error_keeps_persisted_article_done(
    repo: GraphRepository,
    redis: Redis,
    settings: Settings,
    driver: AsyncDriver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLMClient(
        extractions={
            "first": extraction(person("Anna Petrova")),
            "second": extraction(person("Анна Петрова")),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Анна Петрова": vec(1, 0)},
    )
    deps = make_deps(repo, redis, settings, llm)
    await ingest(deps, "first")

    async def broken_release(self: Lock) -> None:
        raise RedisConnectionError("connection reset by peer")

    monkeypatch.setattr(Lock, "release", broken_release)
    with capture_logs() as logs:
        second_id = await ingest(deps, "second")

    article = await status_of(repo, second_id)
    assert (article.status, article.error) == (ArticleStatus.DONE, None)
    assert await scalar(driver, "MATCH (c:Character) RETURN c.embedding_stale") is False
    assert [len(call) for call in llm.embed_calls] == [1, 1, 1]
    assert "resolve_lock_release_failed" in [entry["event"] for entry in logs]


class StalledExtractLLM(FakeLLMClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.extract_started = asyncio.Event()

    async def extract(self, chunk: str, known_relation_types: Sequence[str]) -> ExtractionResult:
        self.extract_started.set()
        await asyncio.sleep(60)
        return await super().extract(chunk, known_relation_types)


async def test_cancelled_processing_marks_failed(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = StalledExtractLLM()
    deps = make_deps(repo, redis, settings, llm)
    article_id = await seed_article(deps, "anna waits forever")
    task = asyncio.create_task(process_article(article_id, deps))
    await asyncio.wait_for(llm.extract_started.wait(), timeout=10)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    article = await status_of(repo, article_id)
    assert article.status is ArticleStatus.FAILED
    assert article.error == "Cancelled: job timeout or worker shutdown"
    with pytest.raises(FileNotFoundError):
        await deps.blobs.load(article_id)


async def test_lock_acquire_timeout_marks_failed(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = FakeLLMClient(extractions={"anna": extraction(person("Anna"))})
    deps = make_deps(repo, redis, tuned(settings, resolve_lock_acquire_timeout_seconds=0.3), llm)
    await redis.set(RESOLVE_LOCK_NAME, b"another-worker", px=10_000)
    article_id = await seed_article(deps, "anna")

    await process_article(article_id, deps)

    article = await status_of(repo, article_id)
    assert article.status is ArticleStatus.FAILED
    assert article.error is not None
    assert article.error.startswith("ResolveLockTimeoutError")
    assert await repo.list_characters_by_article(article_id) == []


async def test_failure_after_persist_keeps_done(
    repo: GraphRepository,
    redis: Redis,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLMClient(extractions={"anna": extraction(person("Anna"))})
    deps = make_deps(repo, redis, settings, llm)

    async def exploding_release(self: HeldLock) -> None:
        raise RuntimeError("unexpected failure after commit")

    monkeypatch.setattr(HeldLock, "release", exploding_release)
    article_id = await ingest(deps, "anna")

    article = await status_of(repo, article_id)
    assert (article.status, article.error) == (ArticleStatus.DONE, None)
    assert [c.name for c in await repo.list_characters_by_article(article_id)] == ["Anna"]


class BlockingRefreshLLM(FakeLLMClient):
    def __init__(self, blocking_prefix: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.blocking_prefix = blocking_prefix
        self.refresh_started = asyncio.Event()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if any(text.startswith(self.blocking_prefix) for text in texts):
            self.refresh_started.set()
            await asyncio.sleep(60)
        return await super().embed(texts)


async def test_cancel_after_persist_keeps_done(
    repo: GraphRepository, redis: Redis, settings: Settings
) -> None:
    llm = BlockingRefreshLLM(
        "Anna Petrova; aliases: Анна Петрова",
        extractions={
            "first": extraction(person("Anna Petrova")),
            "second": extraction(person("Анна Петрова")),
        },
        embeddings={"Anna Petrova": vec(1, 0), "Анна Петрова": vec(1, 0)},
    )
    deps = make_deps(repo, redis, settings, llm)
    await ingest(deps, "first")
    second_id = await seed_article(deps, "second")
    task = asyncio.create_task(process_article(second_id, deps))
    await asyncio.wait_for(llm.refresh_started.wait(), timeout=10)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    article = await status_of(repo, second_id)
    assert (article.status, article.error) == (ArticleStatus.DONE, None)
    with pytest.raises(FileNotFoundError):
        await deps.blobs.load(second_id)
