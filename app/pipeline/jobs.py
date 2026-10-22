from collections.abc import Sequence

from arq.constants import in_progress_key_prefix, job_key_prefix
from redis.asyncio import Redis


def article_job_id(article_id: str) -> str:
    return f"article:{article_id}"


async def articles_with_live_jobs(redis: Redis, article_ids: Sequence[str]) -> set[str]:
    if not article_ids:
        return set()
    async with redis.pipeline(transaction=False) as pipe:
        for article_id in article_ids:
            job_id = article_job_id(article_id)
            pipe.exists(job_key_prefix + job_id, in_progress_key_prefix + job_id)
        counts = await pipe.execute()
    return {article_id for article_id, count in zip(article_ids, counts, strict=True) if count}
