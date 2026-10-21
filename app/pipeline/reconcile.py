import time
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel

from app.domain.models import CharacterProfile
from app.domain.profiles import character_profile, profile_embedding_text, profile_hash
from app.errors import CharactersNotFoundError, NotFoundError
from app.graph.upserts import EmbeddingRefresh, MatchKind, SimilarCharacter, StoredCharacter
from app.pipeline.deps import PipelineDeps
from app.pipeline.lock import ResolveLockTimeoutError
from app.pipeline.matching import JudgeBudget, Rejection, find_match
from app.pipeline.merge_characters import merge_characters

logger = structlog.get_logger(__name__)


class ReconcileReport(BaseModel):
    checked: int
    merged: int
    reembedded: int
    merges: list[tuple[str, str]]


async def reconcile_characters(deps: PipelineDeps) -> ReconcileReport:
    started = time.perf_counter()
    reembedded = await _repair_stale_embeddings(deps)
    run = _ReconcileRun(deps, JudgeBudget(deps.settings.reconcile_max_judge_calls))
    batch_size = deps.settings.reconcile_batch_size
    async for ids in deps.repo.iter_flagged_character_ids(batch_size=batch_size):
        await run.check_batch(ids)
    async for ids in deps.repo.iter_character_ids(batch_size=batch_size):
        await run.check_batch(ids)
    report = ReconcileReport(
        checked=len(run.checked), merged=len(run.merges), reembedded=reembedded, merges=run.merges
    )
    if run.budget.failure is not None:
        logger.warning(
            "judge_unavailable",
            error_type=type(run.budget.failure).__name__,
            unjudged_characters=run.unjudged,
        )
    logger.info(
        "reconcile_finished",
        checked=report.checked,
        merged=report.merged,
        reembedded=report.reembedded,
        judge_calls=run.budget.calls,
        duration_seconds=round(time.perf_counter() - started, 3),
    )
    return report


async def _repair_stale_embeddings(deps: PipelineDeps) -> int:
    repaired = 0
    async for batch in deps.repo.iter_stale_characters(
        batch_size=deps.settings.reconcile_batch_size
    ):
        vectors = await deps.llm.embed([profile_embedding_text(profile) for _, profile in batch])
        refreshed = await deps.repo.refresh_character_embeddings(
            [
                EmbeddingRefresh(
                    id=character_id,
                    name=profile.name,
                    aliases=profile.aliases,
                    description=profile.description,
                    embedding=vector,
                )
                for (character_id, profile), vector in zip(batch, vectors, strict=True)
            ]
        )
        repaired += len(refreshed)
    return repaired


def _older_first(current: StoredCharacter, match: SimilarCharacter) -> tuple[str, str]:
    if (match.created_at, match.id) < (current.created_at, current.id):
        return match.id, current.id
    return current.id, match.id


@dataclass
class _ReconcileRun:
    deps: PipelineDeps
    budget: JudgeBudget
    checked: set[str] = field(default_factory=set)
    unjudged: int = 0
    merges: list[tuple[str, str]] = field(default_factory=list)
    merged_away: set[str] = field(default_factory=set)

    async def check_batch(self, ids: list[str]) -> None:
        stored = await self.deps.repo.get_stored_characters(ids)
        for character_id in ids:
            character = stored.get(character_id)
            if character is None or self._seen(character_id):
                continue
            self.checked.add(character_id)
            await self._absorb_duplicates(character)

    def _seen(self, character_id: str) -> bool:
        return character_id in self.merged_away or character_id in self.checked

    async def _absorb_duplicates(self, character: StoredCharacter) -> None:
        current = character
        while True:
            merged_into, settled = await self._merge_best_match(current)
            if merged_into != current.id:
                break
            refreshed = await self.deps.repo.get_stored_characters([current.id])
            if current.id not in refreshed:
                return
            current = refreshed[current.id]
        if merged_into is None and settled and current.needs_reconcile:
            await self.deps.repo.clear_needs_reconcile(current.id)

    async def _merge_best_match(self, character: StoredCharacter) -> tuple[str | None, bool]:
        own = character_profile(character)
        contexts = await self.deps.repo.get_judge_context([character.id])
        relations, articles = contexts.get(character.id, ([], []))
        outcome = await find_match(
            character_profile(character, relations=relations, articles=articles),
            character.embedding,
            repo=self.deps.repo,
            llm=self.deps.llm,
            settings=self.deps.settings,
            budget=self.budget,
            exclude_ids={character.id},
        )
        await self._record_rejections(character.id, own, outcome.rejections)
        if outcome.unjudged:
            self.unjudged += 1
        if outcome.match is None:
            return None, not outcome.unjudged
        match, kind = outcome.match
        target_id, source_id = _older_first(character, match)
        if await self._known_different(character.id, own, match):
            logger.info(
                "reconcile_skipped_known_different", target_id=target_id, source_id=source_id
            )
            return None, True
        try:
            await merge_characters(target_id, [source_id], self.deps)
        except ResolveLockTimeoutError:
            self._log_skip(target_id, source_id, "resolve_lock_busy")
            return None, False
        except NotFoundError:
            self._log_skip(target_id, source_id, "character_gone")
            return None, False
        self.merged_away.add(source_id)
        self.merges.append((target_id, source_id))
        logger.info(
            "character_reconciled",
            target_id=target_id,
            source_id=source_id,
            kind=MatchKind.RECONCILE.value,
            match=kind.value,
            similarity=match.similarity,
        )
        return target_id, True

    async def _known_different(
        self, character_id: str, own: CharacterProfile, match: SimilarCharacter
    ) -> bool:
        pair = (character_id, match.id, profile_hash(own), profile_hash(character_profile(match)))
        return bool(await self.deps.repo.fresh_not_same([pair]))

    async def _record_rejections(
        self, character_id: str, own: CharacterProfile, rejections: list[Rejection]
    ) -> None:
        for rejection in rejections:
            try:
                await self.deps.repo.record_not_same(
                    character_id,
                    rejection.existing.id,
                    reason=rejection.reason,
                    hash_a=profile_hash(own),
                    hash_b=profile_hash(character_profile(rejection.existing)),
                )
            except CharactersNotFoundError:
                logger.info(
                    "reconcile_not_same_skipped",
                    character_id=character_id,
                    other_id=rejection.existing.id,
                )

    def _log_skip(self, target_id: str, source_id: str, reason: str) -> None:
        logger.warning(
            "reconcile_merge_skipped", target_id=target_id, source_id=source_id, reason=reason
        )
