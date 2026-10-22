from collections.abc import Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings
from app.domain.models import CharacterProfile
from app.graph.repository import GraphRepository
from app.graph.upserts import SimilarCharacter
from app.pipeline.deps import PipelineDeps
from app.pipeline.matching import JudgeBudget, find_match
from app.pipeline.relation_types import resolve_relation_type
from tests.fakes import FakeLLMClient


class RecordingRepo:
    def __init__(
        self,
        *,
        relation_types: Sequence[tuple[str, float]] = (),
        characters: Sequence[SimilarCharacter] = (),
    ) -> None:
        self.relation_types = list(relation_types)
        self.characters = list(characters)
        self.requested_k: list[int] = []

    async def existing_relation_types(self, names: Sequence[str]) -> set[str]:
        return set()

    async def find_similar_relation_types(
        self, embedding: Sequence[float], k: int
    ) -> list[tuple[str, float]]:
        self.requested_k.append(k)
        return self.relation_types[:k]

    async def find_similar_characters(
        self, embedding: Sequence[float], k: int
    ) -> list[SimilarCharacter]:
        self.requested_k.append(k)
        return self.characters[:k]

    async def get_judge_context(
        self, character_ids: Sequence[str]
    ) -> dict[str, tuple[list[str], list[str]]]:
        return {}

    async def fresh_not_same_hashes(self, character_ids: Sequence[str]) -> dict[str, set[str]]:
        return {}


def make_settings(**overrides: object) -> Settings:
    return Settings.model_validate({"jwt_secret": SecretStr("x" * 32), **overrides})


def similar(character_id: str, similarity: float) -> SimilarCharacter:
    return SimilarCharacter(
        id=character_id,
        name=character_id,
        aliases=[],
        description="",
        similarity=similarity,
        created_at=datetime.now(UTC),
    )


def relation_deps(repo: RecordingRepo, settings: Settings) -> PipelineDeps:
    return cast(PipelineDeps, SimpleNamespace(repo=repo, llm=FakeLLMClient(), settings=settings))


async def match(
    repo: RecordingRepo,
    settings: Settings,
    exclude_ids: Sequence[str] = (),
    llm: FakeLLMClient | None = None,
) -> str | None:
    outcome = await find_match(
        CharacterProfile(name="Anna", aliases=[], description=""),
        [1.0],
        repo=cast(GraphRepository, repo),
        llm=llm or FakeLLMClient(),
        settings=settings,
        budget=JudgeBudget(10),
        exclude_ids=exclude_ids,
    )
    return outcome.match[0].id if outcome.match else None


def test_vector_search_k_defaults_to_twenty_and_must_be_positive() -> None:
    assert make_settings().vector_search_k == 20
    with pytest.raises(ValidationError, match="vector_search_k"):
        make_settings(vector_search_k=0)


@pytest.mark.parametrize("k", [1, 7, 20])
async def test_relation_type_resolution_requests_vector_search_k(k: int) -> None:
    repo = RecordingRepo()

    await resolve_relation_type("enemy", relation_deps(repo, make_settings(vector_search_k=k)))

    assert repo.requested_k == [k]


async def test_relation_type_resolution_takes_the_most_similar_candidate() -> None:
    repo = RecordingRepo(relation_types=[("WORKS_FOR", 0.605), ("ENEMY_OF", 0.799)])

    resolved = await resolve_relation_type("enemy", relation_deps(repo, make_settings()))

    assert resolved == "ENEMY_OF"


async def test_relation_type_resolution_rejects_a_best_candidate_below_threshold() -> None:
    repo = RecordingRepo(relation_types=[("WORKS_FOR", 0.5), ("ENEMY_OF", 0.69)])

    resolved = await resolve_relation_type("enemy", relation_deps(repo, make_settings()))

    assert resolved is None


@pytest.mark.parametrize(
    ("vector_search_k", "dedup_candidates", "excluded", "expected_k"),
    [(20, 5, 0, 20), (20, 5, 2, 20), (3, 5, 0, 5), (3, 5, 2, 7), (20, 30, 1, 31)],
)
async def test_dedup_requests_at_least_vector_search_k(
    vector_search_k: int, dedup_candidates: int, excluded: int, expected_k: int
) -> None:
    repo = RecordingRepo()
    settings = make_settings(vector_search_k=vector_search_k, dedup_candidates=dedup_candidates)

    await match(repo, settings, exclude_ids=[f"excluded-{i}" for i in range(excluded)])

    assert repo.requested_k == [expected_k]


async def test_dedup_takes_the_most_similar_candidate_that_is_not_excluded() -> None:
    repo = RecordingRepo(
        characters=[similar("weak", 0.5), similar("self", 0.999), similar("best", 0.97)]
    )

    assert await match(repo, make_settings(), exclude_ids=["self"]) == "best"


async def test_dedup_judges_only_the_top_dedup_candidates_after_exclusion() -> None:
    repo = RecordingRepo(
        characters=[
            similar("third", 0.7),
            similar("self", 0.9),
            similar("first", 0.85),
            similar("second", 0.8),
        ]
    )
    llm = FakeLLMClient()

    await match(repo, make_settings(dedup_candidates=2), exclude_ids=["self"], llm=llm)

    assert [existing.name for _, existing in llm.judge_calls] == ["first", "second"]
