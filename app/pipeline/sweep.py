import structlog

from app.pipeline.deps import PipelineDeps
from app.pipeline.jobs import articles_with_live_jobs

STUCK_ERROR = "Stuck: no progress within job timeout"
SWEEP_PAGE_SIZE = 500

logger = structlog.get_logger(__name__)


async def sweep_stuck_articles(deps: PipelineDeps) -> int:
    older_than = deps.settings.stuck_article_after_seconds
    candidates = await deps.repo.list_stuck_articles(older_than_seconds=older_than)
    failed: list[str] = []
    for start in range(0, len(candidates), SWEEP_PAGE_SIZE):
        page = candidates[start : start + SWEEP_PAGE_SIZE]
        live = await articles_with_live_jobs(deps.redis, page)
        if live:
            logger.info("stuck_sweep_skipped_live_jobs", article_ids=sorted(live))
        failed += await deps.repo.fail_stuck_articles(
            [article_id for article_id in page if article_id not in live],
            older_than_seconds=older_than,
            error=STUCK_ERROR,
        )
    if failed:
        logger.warning("stuck_articles_failed", article_ids=failed, count=len(failed))
    return len(failed)
