from collections.abc import Collection, Sequence
from dataclasses import dataclass

import structlog

from app.config import Settings
from app.domain.models import CharacterProfile
from app.domain.profiles import character_profile, profile_hash
from app.graph.repository import GraphRepository
from app.graph.upserts import MatchKind, SimilarCharacter
from app.llm.client import LLMClient, LLMError, LLMNotConfiguredError

MatchResult = tuple[SimilarCharacter, MatchKind]
JUDGE_ERROR_MAX_CHARS = 200

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class JudgeFailure:
    error_type: str
    error_message: str


class JudgeBudget:
    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.calls = 0
        self.failure: LLMError | None = None
        self._exhaustion_logged = False

    def take(self) -> bool:
        if self.failure is not None:
            return False
        if self.calls < self.max_calls:
            self.calls += 1
            return True
        if not self._exhaustion_logged:
            logger.warning("judge_cap_reached", max_calls=self.max_calls)
            self._exhaustion_logged = True
        return False

    def fail(self, error: LLMError) -> None:
        self.failure = error

    def failure_summary(self) -> JudgeFailure | None:
        if self.failure is None:
            return None
        return JudgeFailure(
            error_type=type(self.failure).__name__,
            error_message=str(self.failure)[:JUDGE_ERROR_MAX_CHARS],
        )


@dataclass(frozen=True)
class Rejection:
    existing: SimilarCharacter
    reason: str


@dataclass(frozen=True)
class MatchOutcome:
    match: MatchResult | None
    rejections: list[Rejection]
    best_similarity: float | None
    unjudged: bool = False


async def find_match(
    profile: CharacterProfile,
    embedding: Sequence[float],
    *,
    repo: GraphRepository,
    llm: LLMClient,
    settings: Settings,
    budget: JudgeBudget,
    exclude_ids: Collection[str] = (),
) -> MatchOutcome:
    k = max(settings.vector_search_k, settings.dedup_candidates + len(exclude_ids))
    found = await repo.find_similar_characters(embedding, k)
    ranked = sorted(found, key=lambda c: (-c.similarity, c.id))
    candidates = [c for c in ranked if c.id not in exclude_ids][: settings.dedup_candidates]
    if not candidates:
        return MatchOutcome(match=None, rejections=[], best_similarity=None)
    best = candidates[0]
    if best.similarity >= settings.dedup_auto_merge_threshold:
        return MatchOutcome(
            match=(best, MatchKind.VECTOR), rejections=[], best_similarity=best.similarity
        )
    band = [c for c in candidates if c.similarity >= settings.dedup_judge_threshold]
    match, rejections, unjudged = await _judge_band(
        profile, band, repo=repo, llm=llm, budget=budget
    )
    return MatchOutcome(
        match=match, rejections=rejections, best_similarity=best.similarity, unjudged=unjudged
    )


async def _judge_band(
    profile: CharacterProfile,
    band: list[SimilarCharacter],
    *,
    repo: GraphRepository,
    llm: LLMClient,
    budget: JudgeBudget,
) -> tuple[MatchResult | None, list[Rejection], bool]:
    if not band:
        return None, [], False
    ids = [candidate.id for candidate in band]
    contexts = await repo.get_judge_context(ids)
    judged_different = await repo.fresh_not_same_hashes(ids)
    candidate_hash = profile_hash(profile)
    rejections: list[Rejection] = []
    for existing in band:
        if candidate_hash in judged_different.get(existing.id, set()):
            logger.info("judge_skipped_known_different", existing_id=existing.id)
            continue
        if not budget.take():
            return None, rejections, True
        relations, articles = contexts.get(existing.id, ([], []))
        try:
            verdict = await llm.judge_same_character(
                profile, character_profile(existing, relations=relations, articles=articles)
            )
        except LLMNotConfiguredError:
            raise
        except LLMError as error:
            budget.fail(error)
            return None, rejections, True
        if verdict.same:
            return (existing, MatchKind.JUDGE), rejections, False
        rejections.append(Rejection(existing=existing, reason=verdict.reason))
    return None, rejections, False
