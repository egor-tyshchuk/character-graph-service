import structlog

from app.pipeline.deps import PipelineDeps

STUCK_ERROR = "Stuck: no progress within job timeout"

logger = structlog.get_logger(__name__)


async def sweep_stuck_articles(deps: PipelineDeps) -> int:
    failed = await deps.repo.fail_stuck_articles(
        older_than_seconds=deps.settings.stuck_article_after_seconds, error=STUCK_ERROR
    )
    if failed:
        logger.warning("stuck_articles_failed", article_ids=failed, count=len(failed))
    return len(failed)
