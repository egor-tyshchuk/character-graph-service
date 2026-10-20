import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from neo4j import AsyncDriver

from app.domain.models import (
    Article,
    ArticleFormat,
    ArticleStats,
    ArticleStatus,
    Character,
    Direction,
    Evidence,
)
from app.domain.profiles import profile_hash
from app.graph.repository import GraphRepository
from app.graph.upserts import (
    CharacterUpsert,
    EmbeddingRefresh,
    MatchKind,
    Mention,
    NotSameUpsert,
    RelationTypeUpsert,
    RelationUpsert,
)
from app.llm.schemas import CharacterProfile
from tests.fakes import vec


async def seed_article(repo: GraphRepository, title: str = "A tale") -> str:
    now = datetime.now(UTC)
    article = Article(
        id=str(uuid.uuid4()),
        title=title,
        format=ArticleFormat.MARKDOWN,
        content_hash=str(uuid.uuid4()),
        status=ArticleStatus.PROCESSING,
        created_at=now,
        updated_at=now,
    )
    await repo.create_article(article)
    return article.id


def character(
    name: str,
    *,
    embedding: list[float] | None = None,
    character_id: str | None = None,
    aliases: Sequence[str] = (),
    mentions: Sequence[Mention] = (),
) -> CharacterUpsert:
    return CharacterUpsert(
        id=character_id or str(uuid.uuid4()),
        name=name,
        aliases=list(aliases),
        description=f"{name} from the tale.",
        embedding=embedding or vec(1),
        mentions=list(mentions)
        or [Mention(surface_name=name, description="", match=MatchKind.NEW, similarity=None)],
    )


def relation(
    source: CharacterUpsert, target: CharacterUpsert, type_: str = "FRIEND_OF"
) -> RelationUpsert:
    return RelationUpsert(
        source_id=source.id,
        target_id=target.id,
        type=type_,
        description=f"{source.name} {type_} {target.name}",
        evidence=f"quote about {source.name} and {target.name}",
    )


async def persist(
    repo: GraphRepository,
    article_id: str,
    characters: Sequence[CharacterUpsert],
    relations: Sequence[RelationUpsert] = (),
) -> ArticleStats:
    stats = ArticleStats(characters=len(characters), relations=len(relations), images_described=1)
    types = sorted({r.type for r in relations})
    await repo.persist_article(
        article_id,
        characters=characters,
        relations=relations,
        relation_types=[
            RelationTypeUpsert(name=name, description=name.lower(), embedding=vec(1))
            for name in types
        ],
        stats=stats,
    )
    return stats


async def count(driver: AsyncDriver, pattern: str) -> int:
    records, _, _ = await driver.execute_query(f"MATCH {pattern} RETURN count(*) AS n")
    return int(records[0]["n"])


async def graph_counts(driver: AsyncDriver) -> tuple[int, int, int, int]:
    return (
        await count(driver, "(:Character)"),
        await count(driver, "(:Character)-[:APPEARS_IN]->(:Article)"),
        await count(driver, "(:Character)-[:RELATED_TO]->(:Character)"),
        await count(driver, "(:RelationType)"),
    )


async def test_persist_article_writes_nodes_edges_and_status(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    article_id = await seed_article(repo)
    anna, mark = character("Anna Petrova"), character("Mark Petrov")

    stats = await persist(repo, article_id, [anna, mark], [relation(anna, mark, "SIBLING_OF")])

    assert await graph_counts(driver) == (2, 2, 1, 1)
    article = await repo.get_article(article_id)
    assert article is not None
    assert article.status is ArticleStatus.DONE
    assert article.error is None
    assert article.stats == stats


async def test_persist_article_updates_existing_character_and_keeps_created_at(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    anna = character("Anna Petrova")
    await persist(repo, await seed_article(repo), [anna])
    first, _, _ = await driver.execute_query(
        "MATCH (c:Character {id: $id}) RETURN c.created_at AS created_at", id=anna.id
    )

    renamed = anna.model_copy(update={"aliases": ["Anya"], "embedding": vec(0, 1)})
    await persist(repo, await seed_article(repo, "Second"), [renamed])

    records, _, _ = await driver.execute_query(
        "MATCH (c:Character {id: $id}) RETURN c.created_at AS created_at, c.updated_at AS "
        "updated_at, c.aliases AS aliases, c.embedding AS embedding",
        id=anna.id,
    )
    assert records[0]["created_at"] == first[0]["created_at"]
    assert records[0]["updated_at"] >= records[0]["created_at"]
    assert records[0]["aliases"] == ["Anya"]
    assert records[0]["embedding"] == pytest.approx(vec(0, 1))


async def test_persist_article_for_missing_article_raises_and_writes_nothing(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    with pytest.raises(LookupError):
        await persist(repo, "missing", [character("Anna Petrova")])

    assert await count(driver, "(:Character)") == 0


async def test_appears_in_stores_mention_provenance(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    article_id = await seed_article(repo)
    mentions = [
        Mention(surface_name="Анна", description="Sister.", match=MatchKind.JUDGE, similarity=0.87),
        Mention(surface_name="Anya", description="", match=MatchKind.VECTOR, similarity=0.97),
    ]
    anna = character("Anna Petrova", mentions=mentions)

    await persist(repo, article_id, [anna])

    records, _, _ = await driver.execute_query(
        "MATCH (:Character {id: $id})-[m:APPEARS_IN]->(:Article {id: $article_id}) "
        "RETURN m.match AS match, m.similarity AS similarity, m.mentions_json AS mentions",
        id=anna.id,
        article_id=article_id,
    )
    assert records[0]["match"] == "judge"
    assert records[0]["similarity"] == pytest.approx(0.87)
    stored = [Mention.model_validate(item) for item in json.loads(records[0]["mentions"])]
    assert stored == mentions


async def test_get_judge_context_lists_relations_and_titles(repo: GraphRepository) -> None:
    anna, mark, pyotr = character("Anna Petrova"), character("Mark Petrov"), character("Pyotr")
    lonely = character("Lonely")
    await persist(
        repo,
        await seed_article(repo, "The lighthouse"),
        [anna, mark, lonely],
        [relation(anna, mark, "SIBLING_OF")],
    )
    await persist(
        repo,
        await seed_article(repo, "The harbour"),
        [anna, pyotr],
        [relation(pyotr, anna, "EMPLOYS")],
    )

    context = await repo.get_judge_context([anna.id, mark.id, lonely.id])

    assert context[anna.id] == (
        ["SIBLING_OF → Mark Petrov", "← EMPLOYS Pyotr"],
        ["The harbour", "The lighthouse"],
    )
    assert context[mark.id] == (["← SIBLING_OF Anna Petrova"], ["The lighthouse"])
    assert context[lonely.id] == ([], ["The lighthouse"])


async def test_get_judge_context_caps_relations_and_titles(repo: GraphRepository) -> None:
    hub = character("Hub")
    for index in range(6):
        friends = [character(f"Friend {index}-{n}") for n in range(4)]
        await persist(
            repo,
            await seed_article(repo, f"Article {index}"),
            [hub, *friends],
            [relation(hub, friend) for friend in friends],
        )

    relations, titles = (await repo.get_judge_context([hub.id]))[hub.id]

    assert len(relations) == 20
    assert len(titles) == 5


async def test_persist_is_idempotent(repo: GraphRepository, driver: AsyncDriver) -> None:
    article_id = await seed_article(repo)
    anna, mark = character("Anna Petrova"), character("Mark Petrov")

    await persist(repo, article_id, [anna, mark], [relation(anna, mark, "SIBLING_OF")])
    await persist(repo, article_id, [anna, mark], [relation(anna, mark, "SIBLING_OF")])

    assert await graph_counts(driver) == (2, 2, 1, 1)


async def test_same_relation_from_two_articles_keeps_both_edges(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    anna, mark = character("Anna Petrova"), character("Mark Petrov")

    for title in ("First", "Second"):
        await persist(repo, await seed_article(repo, title), [anna, mark], [relation(anna, mark)])

    assert await count(driver, "(:Character)-[:RELATED_TO {type: 'FRIEND_OF'}]->(:Character)") == 2
    assert await count(driver, "(:RelationType)") == 1


async def test_relation_type_keeps_first_description_and_embedding(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    article_id = await seed_article(repo)
    anna, mark = character("Anna Petrova"), character("Mark Petrov")
    stats = ArticleStats(characters=2, relations=1, images_described=0)
    for description, embedding in (("first", vec(1)), ("second", vec(0, 1))):
        await repo.persist_article(
            article_id,
            characters=[anna, mark],
            relations=[relation(anna, mark)],
            relation_types=[
                RelationTypeUpsert(name="FRIEND_OF", description=description, embedding=embedding)
            ],
            stats=stats,
        )

    records, _, _ = await driver.execute_query(
        "MATCH (t:RelationType {name: 'FRIEND_OF'}) RETURN t.description AS d, t.embedding AS e"
    )
    assert records[0]["d"] == "first"
    assert records[0]["e"] == pytest.approx(vec(1))


async def test_find_similar_characters_returns_cosine(repo: GraphRepository) -> None:
    anna = character("Anna Petrova", embedding=vec(1, 0), aliases=["Anya"])
    await persist(repo, await seed_article(repo), [anna])

    same = await repo.find_similar_characters(vec(1, 0), k=5)
    orthogonal = await repo.find_similar_characters(vec(0, 1), k=5)

    assert [c.id for c in same] == [anna.id]
    assert same[0].similarity == pytest.approx(1.0, abs=1e-3)
    assert same[0].name == "Anna Petrova"
    assert same[0].aliases == ["Anya"]
    assert same[0].description == anna.description
    assert same[0].created_at.tzinfo is not None
    assert orthogonal[0].similarity == pytest.approx(0.0, abs=1e-3)


async def test_find_similar_characters_orders_by_similarity_and_limits_k(
    repo: GraphRepository,
) -> None:
    near = character("Near", embedding=vec(1, 0.1))
    far = character("Far", embedding=vec(1, 1))
    exact = character("Exact", embedding=vec(1, 0))
    await persist(repo, await seed_article(repo), [far, near, exact])

    results = await repo.find_similar_characters(vec(1, 0), k=2)

    assert [c.name for c in results] == ["Exact", "Near"]
    assert results[0].similarity > results[1].similarity


async def test_vector_index_reads_its_own_committed_writes(repo: GraphRepository) -> None:
    article_id = await seed_article(repo)
    for index in range(1, 8):
        embedding = vec(*([0.0] * index), 1.0)
        written = character(f"Fresh {index}", embedding=embedding)

        await persist(repo, article_id, [written])

        found = await repo.find_similar_characters(embedding, k=1)
        assert [c.id for c in found] == [written.id]


async def test_find_similar_relation_types_returns_cosine(repo: GraphRepository) -> None:
    anna, mark = character("Anna Petrova"), character("Mark Petrov")
    await repo.persist_article(
        await seed_article(repo),
        characters=[anna, mark],
        relations=[relation(anna, mark, "FRIEND_OF"), relation(mark, anna, "ENEMY_OF")],
        relation_types=[
            RelationTypeUpsert(name="FRIEND_OF", description="friends", embedding=vec(1, 0)),
            RelationTypeUpsert(name="ENEMY_OF", description="enemies", embedding=vec(0, 1)),
        ],
        stats=ArticleStats(characters=2, relations=2, images_described=0),
    )

    results = await repo.find_similar_relation_types(vec(1, 0), k=2)

    assert [name for name, _ in results] == ["FRIEND_OF", "ENEMY_OF"]
    assert results[0][1] == pytest.approx(1.0, abs=1e-3)
    assert results[1][1] == pytest.approx(0.0, abs=1e-3)


async def test_list_characters_paginates_and_counts_articles(repo: GraphRepository) -> None:
    bob, anna, carl = character("bob"), character("Anna"), character("Carl")
    await persist(repo, await seed_article(repo, "First"), [bob, anna, carl])
    await persist(repo, await seed_article(repo, "Second"), [anna])

    first_page, total = await repo.list_characters(limit=2, offset=0)
    second_page, second_total = await repo.list_characters(limit=2, offset=2)
    beyond, beyond_total = await repo.list_characters(limit=2, offset=10)

    assert total == second_total == beyond_total == 3
    assert [c.name for c in first_page] == ["Anna", "bob"]
    assert [c.name for c in second_page] == ["Carl"]
    assert beyond == []
    assert first_page[0] == Character(
        id=anna.id, name="Anna", aliases=[], description=anna.description, article_count=2
    )
    assert first_page[1].article_count == 1


async def test_list_characters_by_article(repo: GraphRepository) -> None:
    first_id = await seed_article(repo, "First")
    second_id = await seed_article(repo, "Second")
    zoe, anna, mark = character("Zoe"), character("anna"), character("Mark")
    await persist(repo, first_id, [zoe, anna])
    await persist(repo, second_id, [mark, anna])

    in_first = await repo.list_characters_by_article(first_id)

    assert [c.name for c in in_first] == ["anna", "Zoe"]
    assert [c.article_count for c in in_first] == [2, 1]
    assert await repo.list_characters_by_article("missing") == []


async def test_get_character(repo: GraphRepository) -> None:
    anna = character("Anna Petrova", aliases=["Anya"])
    await persist(repo, await seed_article(repo), [anna])

    assert await repo.get_character(anna.id) == Character(
        id=anna.id,
        name="Anna Petrova",
        aliases=["Anya"],
        description=anna.description,
        article_count=1,
    )
    assert await repo.get_character("missing") is None


async def test_list_relation_types_with_usage_counts(repo: GraphRepository) -> None:
    anna, mark, lena = character("Anna"), character("Mark"), character("Lena")
    await persist(
        repo,
        await seed_article(repo, "First"),
        [anna, mark, lena],
        [relation(anna, mark, "SIBLING_OF"), relation(anna, lena, "FRIEND_OF")],
    )
    await persist(
        repo,
        await seed_article(repo, "Second"),
        [anna, mark, lena],
        [relation(mark, lena, "FRIEND_OF"), relation(lena, anna, "ENEMY_OF")],
    )
    await repo.persist_article(
        await seed_article(repo, "Unused"),
        characters=[],
        relations=[],
        relation_types=[RelationTypeUpsert(name="UNUSED_OF", description="", embedding=vec(1))],
        stats=ArticleStats(characters=0, relations=0, images_described=0),
    )

    types = await repo.list_relation_types()

    assert [(t.name, t.usage_count) for t in types] == [
        ("FRIEND_OF", 2),
        ("ENEMY_OF", 1),
        ("SIBLING_OF", 1),
    ]
    assert types[0].description == "friend_of"


async def test_find_connected_both_directions(repo: GraphRepository) -> None:
    a, b, c = character("A"), character("B"), character("C")
    await persist(
        repo, await seed_article(repo), [a, b, c], [relation(a, b, "FRIEND_OF"), relation(c, a)]
    )

    connections = await repo.find_connected(a.id, "FRIEND_OF")

    assert [(x.character.name, x.direction) for x in connections] == [
        ("B", Direction.OUTGOING),
        ("C", Direction.INCOMING),
    ]
    assert connections[0].relation_type == "FRIEND_OF"
    assert connections[0].character.article_count == 1
    assert await repo.find_connected(a.id, "ENEMY_OF") == []


async def test_find_connected_groups_evidence_per_neighbour(repo: GraphRepository) -> None:
    anna, mark = character("Anna"), character("Mark")
    article_ids = sorted([await seed_article(repo, "First"), await seed_article(repo, "Second")])
    for article_id in reversed(article_ids):
        await persist(repo, article_id, [anna, mark], [relation(anna, mark), relation(mark, anna)])

    connections = await repo.find_connected(anna.id, "FRIEND_OF")

    assert [(x.character.id, x.direction) for x in connections] == [
        (mark.id, Direction.OUTGOING),
        (mark.id, Direction.INCOMING),
    ]
    outgoing = relation(anna, mark)
    assert connections[0].evidence == [
        Evidence(
            article_id=article_id, description=outgoing.description, evidence=outgoing.evidence
        )
        for article_id in article_ids
    ]
    assert [e.article_id for e in connections[1].evidence] == article_ids


async def test_service_edges_never_leak_into_relation_reads(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    anna, mark = character("Anna"), character("Mark")
    article_id = await seed_article(repo)
    await persist(repo, article_id, [anna, mark], [relation(anna, mark, "SIBLING_OF")])
    await driver.execute_query(
        "MATCH (a:Character {id: $a}), (m:Character {id: $m}) "
        "CREATE (a)-[:NOT_SAME {type: 'SIBLING_OF', article_id: $article_id, reason: 'x'}]->(m) "
        "MERGE (:RelationType {name: 'NOT_SAME'})",
        a=anna.id,
        m=mark.id,
        article_id=article_id,
    )

    assert [(t.name, t.usage_count) for t in await repo.list_relation_types()] == [
        ("SIBLING_OF", 1)
    ]
    connections = await repo.find_connected(anna.id, "SIBLING_OF")
    assert [len(x.evidence) for x in connections] == [1]
    assert await repo.find_connected(anna.id, "NOT_SAME") == []
    relations, _ = (await repo.get_judge_context([anna.id]))[anna.id]
    assert relations == ["SIBLING_OF → Mark"]


def _profile(name: str, aliases: list[str] | None = None) -> CharacterProfile:
    return CharacterProfile(name=name, aliases=aliases or [], description=f"{name} the sailor.")


async def test_not_same_fresh_until_profile_changes(repo: GraphRepository) -> None:
    ivan_a, ivan_b = character("Иван"), character("Иван")
    await persist(repo, await seed_article(repo), [ivan_a, ivan_b])
    hash_a, hash_b = profile_hash(_profile("Иван")), profile_hash(_profile("Иван", ["Ваня"]))

    await repo.record_not_same(
        ivan_a.id, ivan_b.id, reason="different", hash_a=hash_a, hash_b=hash_b
    )

    pair = frozenset({ivan_a.id, ivan_b.id})
    assert await repo.fresh_not_same([(ivan_a.id, ivan_b.id, hash_a, hash_b)]) == {pair}
    assert await repo.fresh_not_same([(ivan_b.id, ivan_a.id, hash_b, hash_a)]) == {pair}
    changed = profile_hash(_profile("Иван", ["Ваня", "Иван Иванович"]))
    assert await repo.fresh_not_same([(ivan_a.id, ivan_b.id, hash_a, changed)]) == set()
    assert await repo.fresh_not_same([(ivan_a.id, ivan_b.id, hash_b, hash_a)]) == set()
    assert await repo.fresh_not_same([]) == set()


async def test_not_same_stored_once_per_pair(repo: GraphRepository, driver: AsyncDriver) -> None:
    first, second = character("Иван"), character("Иван")
    await persist(repo, await seed_article(repo), [first, second])
    low, high = sorted([first.id, second.id])

    await repo.record_not_same(low, high, reason="old", hash_a="h-low", hash_b="h-high")
    await repo.record_not_same(high, low, reason="new", hash_a="h-high-2", hash_b="h-low-2")

    records, _, _ = await driver.execute_query(
        "MATCH (l:Character)-[n:NOT_SAME]->(h:Character) "
        "RETURN l.id AS low, h.id AS high, n.reason AS reason, n.hash_low AS hash_low, "
        "n.hash_high AS hash_high, n.judged_at AS judged_at"
    )
    assert len(records) == 1
    stored = records[0]
    assert (stored["low"], stored["high"], stored["reason"]) == (low, high, "new")
    assert (stored["hash_low"], stored["hash_high"]) == ("h-low-2", "h-high-2")
    assert stored["judged_at"] is not None
    assert await repo.fresh_not_same([(low, high, "h-low-2", "h-high-2")]) == {
        frozenset({low, high})
    }


async def test_persist_rejects_relation_with_unknown_endpoint_and_rolls_back(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    article_id = await seed_article(repo)
    anna, mark = character("Anna Petrova"), character("Mark Petrov")
    ghost = character("Ghost")

    with pytest.raises(LookupError, match="relations"):
        await persist(repo, article_id, [anna, mark], [relation(anna, mark), relation(anna, ghost)])

    assert await graph_counts(driver) == (0, 0, 0, 0)
    article = await repo.get_article(article_id)
    assert article is not None
    assert article.status is ArticleStatus.PROCESSING


async def test_persist_keeps_stored_embedding_and_flags_stale_without_new_one(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    anna = character("Anna Petrova", embedding=vec(1, 0))
    await persist(repo, await seed_article(repo), [anna])
    renamed = anna.model_copy(update={"aliases": ["Anya"], "embedding": None})

    await persist(repo, await seed_article(repo, "Second"), [renamed])

    records, _, _ = await driver.execute_query(
        "MATCH (c:Character {id: $id}) RETURN c.embedding AS e, c.embedding_stale AS stale, "
        "c.aliases AS aliases",
        id=anna.id,
    )
    assert records[0]["e"] == pytest.approx(vec(1, 0))
    assert records[0]["stale"] is True
    assert records[0]["aliases"] == ["Anya"]


async def test_refresh_character_embeddings_updates_current_profiles_only(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    anna = character("Anna Petrova", embedding=vec(1, 0), aliases=["Anya"])
    mark = character("Mark Petrov", embedding=vec(0, 1))
    await persist(repo, await seed_article(repo), [anna, mark])
    await persist(
        repo, await seed_article(repo, "Second"), [anna.model_copy(update={"embedding": None})]
    )

    refreshed = await repo.refresh_character_embeddings(
        [
            EmbeddingRefresh(
                id=anna.id,
                name=anna.name,
                aliases=anna.aliases,
                description=anna.description,
                embedding=vec(1, 1),
            ),
            EmbeddingRefresh(
                id=mark.id,
                name=mark.name,
                aliases=["Outdated alias"],
                description=mark.description,
                embedding=vec(1, 1),
            ),
        ]
    )

    assert refreshed == {anna.id}
    records, _, _ = await driver.execute_query(
        "MATCH (c:Character) RETURN c.id AS id, c.embedding AS e, c.embedding_stale AS stale"
    )
    stored = {record["id"]: (record["e"], record["stale"]) for record in records}
    assert stored[anna.id][0] == pytest.approx(vec(1, 1))
    assert stored[anna.id][1] is False
    assert stored[mark.id][0] == pytest.approx(vec(0, 1))


async def test_existing_relation_types_matches_exact_names(repo: GraphRepository) -> None:
    anna, mark = character("Anna"), character("Mark")
    await persist(repo, await seed_article(repo), [anna, mark], [relation(anna, mark, "PARENT_OF")])

    existing = await repo.existing_relation_types(["PARENT_OF", "CHILD_OF", "parent_of"])

    assert existing == {"PARENT_OF"}
    assert await repo.existing_relation_types([]) == set()


async def test_persist_records_not_same_verdicts_in_the_same_transaction(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    ivan_a, ivan_b = character("Иван"), character("Иван")
    await persist(repo, await seed_article(repo), [ivan_a])
    article_id = await seed_article(repo, "Second")

    await repo.persist_article(
        article_id,
        characters=[ivan_b],
        relations=[],
        relation_types=[],
        not_same=[
            NotSameUpsert(
                a_id=ivan_b.id, b_id=ivan_a.id, reason="different", hash_a="h-b", hash_b="h-a"
            )
        ],
        stats=ArticleStats(characters=1, relations=0, images_described=0),
    )

    assert await repo.fresh_not_same([(ivan_a.id, ivan_b.id, "h-a", "h-b")]) == {
        frozenset({ivan_a.id, ivan_b.id})
    }
    assert await count(driver, "(:Character)-[:NOT_SAME]->(:Character)") == 1


async def test_persist_rejects_not_same_with_unknown_character_and_rolls_back(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    ivan = character("Иван")
    article_id = await seed_article(repo)

    with pytest.raises(LookupError, match="NOT_SAME"):
        await repo.persist_article(
            article_id,
            characters=[ivan],
            relations=[],
            relation_types=[],
            not_same=[
                NotSameUpsert(a_id=ivan.id, b_id="missing", reason="x", hash_a="a", hash_b="b")
            ],
            stats=ArticleStats(characters=1, relations=0, images_described=0),
        )

    assert await count(driver, "(:Character)") == 0


async def test_record_not_same_rejects_self_pair(repo: GraphRepository) -> None:
    ivan = character("Иван")
    await persist(repo, await seed_article(repo), [ivan])

    with pytest.raises(ValueError, match="itself"):
        await repo.record_not_same(ivan.id, ivan.id, reason="x", hash_a="a", hash_b="a")


async def test_record_not_same_rejects_unknown_ids(
    repo: GraphRepository, driver: AsyncDriver
) -> None:
    ivan = character("Иван")
    await persist(repo, await seed_article(repo), [ivan])

    with pytest.raises(LookupError):
        await repo.record_not_same(ivan.id, "missing", reason="x", hash_a="a", hash_b="b")
    with pytest.raises(LookupError):
        await repo.record_not_same("missing", ivan.id, reason="x", hash_a="a", hash_b="b")

    assert await count(driver, "()-[:NOT_SAME]->()") == 0


def _stored_profile(upsert: CharacterUpsert) -> CharacterProfile:
    return CharacterProfile(
        name=upsert.name, aliases=upsert.aliases, description=upsert.description
    )


async def test_fresh_not_same_hashes_lists_unchanged_verdict_partners(
    repo: GraphRepository,
) -> None:
    ivan_a, ivan_b, ivan_c = character("Иван"), character("Иван", aliases=["Ваня"]), character("И")
    await persist(repo, await seed_article(repo), [ivan_a, ivan_b, ivan_c])
    hash_a, hash_b = (profile_hash(_stored_profile(c)) for c in (ivan_a, ivan_b))
    await repo.record_not_same(ivan_a.id, ivan_b.id, reason="x", hash_a=hash_a, hash_b=hash_b)
    await repo.record_not_same(ivan_a.id, ivan_c.id, reason="x", hash_a=hash_a, hash_b="old")

    verdicts = await repo.fresh_not_same_hashes([ivan_a.id, ivan_b.id, ivan_c.id, "missing"])

    assert verdicts == {ivan_a.id: {hash_b}, ivan_b.id: {hash_a}}
