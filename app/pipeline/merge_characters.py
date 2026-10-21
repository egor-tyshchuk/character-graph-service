from collections.abc import Sequence
from functools import reduce

import structlog

from app.domain.models import Character
from app.domain.profiles import merge_profiles, profile_embedding_text
from app.errors import NotFoundError
from app.pipeline.lock import resolve_lock
from app.pipeline.process import PipelineDeps

logger = structlog.get_logger(__name__)


async def merge_characters(
    target_id: str, source_ids: Sequence[str], deps: PipelineDeps
) -> Character:
    merged_ids = [target_id, *source_ids]
    async with resolve_lock(
        deps.redis,
        deps.settings,
        acquire_timeout=deps.settings.merge_lock_acquire_timeout_seconds,
        target_id=target_id,
    ) as held:
        profiles = await deps.repo.get_character_profiles(merged_ids)
        missing = [character_id for character_id in merged_ids if character_id not in profiles]
        if missing:
            raise NotFoundError(f"Characters not found: {', '.join(missing)}")
        profile = reduce(
            merge_profiles, (profiles[source_id] for source_id in source_ids), profiles[target_id]
        )
        [embedding] = await deps.llm.embed([profile_embedding_text(profile)])
        await held.ensure_owned()
        try:
            await deps.repo.merge_characters(
                target_id, source_ids, profile=profile, embedding=embedding
            )
        except LookupError as exc:
            raise NotFoundError(str(exc)) from exc
        merged = await deps.repo.get_character(target_id)
    if merged is None:
        raise NotFoundError(f"Character {target_id} not found")
    logger.info("characters_merged", target_id=target_id, source_ids=list(source_ids))
    return merged
