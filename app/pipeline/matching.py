from collections.abc import Collection, Sequence
from dataclasses import dataclass

import structlog

from app.config import Settings
from app.domain.models import CharacterProfile
from app.domain.profiles import character_profile, profile_hash
from app.graph.repository import GraphRepository
from app.graph.upserts import MatchKind, SimilarCharacter
from app.llm.client import LLMClient

MatchResult = tuple[SimilarCharacter, MatchKind]

logger = structlog.get_logger(__name__)


class JudgeBudget:
    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.calls = 0
        self._exhaustion_logged = False

    def take(self) -> bool:
        if self.calls < self.max_calls:
            self.calls += 1
            return True
        if not self._exhaustion_logged:
            logger.warning("judge_cap_reached", max_calls=self.max_calls)
            self._exhaustion_logged = True
        return False


@dataclass(frozen=True)
class Rejection:
    existing: SimilarCharacter
    reason: str


@dataclass(frozen=True)
class MatchOutcome:
    match: MatchResult | None
    rejections: list[Rejection]
    best_similarity: float | None


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
    found = await repo.find_similar_characters(
        embedding, settings.dedup_candidates + len(exclude_ids)
    )
    candidates = [c for c in found if c.id not in exclude_ids][: settings.dedup_candidates]
    if not candidates:
        return MatchOutcome(match=None, rejections=[], best_similarity=None)
    best = candidates[0]
    if best.similarity >= settings.dedup_auto_merge_threshold:
        return MatchOutcome(
            match=(best, MatchKind.VECTOR), rejections=[], best_similarity=best.similarity
        )
    band = [c for c in candidates if c.similarity >= settings.dedup_judge_threshold]
    match, rejections = await _judge_band(profile, band, repo=repo, llm=llm, budget=budget)
    return MatchOutcome(match=match, rejections=rejections, best_similarity=best.similarity)


async def _judge_band(
    profile: CharacterProfile,
    band: list[SimilarCharacter],
    *,
    repo: GraphRepository,
    llm: LLMClient,
    budget: JudgeBudget,
) -> tuple[MatchResult | None, list[Rejection]]:
    if not band:
        return None, []
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
            break
        relations, articles = contexts.get(existing.id, ([], []))
        verdict = await llm.judge_same_character(
            profile, character_profile(existing, relations=relations, articles=articles)
        )
        if verdict.same:
            return (existing, MatchKind.JUDGE), rejections
        rejections.append(Rejection(existing=existing, reason=verdict.reason))
    return None, rejections
