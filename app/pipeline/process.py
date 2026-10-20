import asyncio
import time
from dataclasses import dataclass

import structlog
from redis.asyncio import Redis

from app.config import Settings
from app.domain.models import Article, ArticleFormat, ArticleStats, ArticleStatus
from app.domain.profiles import profile_embedding_text
from app.graph.repository import GraphRepository
from app.graph.upserts import CharacterUpsert, EmbeddingRefresh
from app.ingestion.chunking import chunk_text
from app.ingestion.markdown import parse_markdown
from app.ingestion.models import ParsedDocument
from app.ingestion.pdf import parse_pdf
from app.llm.client import LLMClient
from app.pipeline.lock import resolve_lock
from app.pipeline.merge import merge_extractions
from app.pipeline.resolve import (
    ResolvedArticle,
    character_profile,
    embed_article,
    resolve_article,
)
from app.storage import BlobStore

ERROR_MAX_CHARS = 500

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PipelineDeps:
    repo: GraphRepository
    llm: LLMClient
    blobs: BlobStore
    redis: Redis
    settings: Settings


async def process_article(article_id: str, deps: PipelineDeps) -> None:
    article = await deps.repo.get_article(article_id)
    if article is None:
        logger.warning("article_not_found", article_id=article_id)
        return
    if article.status is ArticleStatus.DONE:
        logger.info("article_already_done", article_id=article_id)
        return
    started = time.perf_counter()
    try:
        await deps.repo.set_article_status(article_id, ArticleStatus.PROCESSING)
        resolved, stats = await _process(article, deps)
    except Exception as exc:
        logger.exception("article_processing_failed", article_id=article_id)
        error = f"{type(exc).__name__}: {exc}"[:ERROR_MAX_CHARS]
        await deps.repo.set_article_status(article_id, ArticleStatus.FAILED, error=error)
    else:
        logger.info(
            "article_processed",
            article_id=article_id,
            characters=stats.characters,
            matched_existing=sum(c.embedding is None for c in resolved.characters),
            relations=stats.relations,
            relation_types_created=len(resolved.relation_types),
            not_same_recorded=len(resolved.not_same),
            images_described=stats.images_described,
            duration_seconds=round(time.perf_counter() - started, 3),
        )
    finally:
        await deps.blobs.delete(article_id)


async def _process(article: Article, deps: PipelineDeps) -> tuple[ResolvedArticle, ArticleStats]:
    document = await _parse(article, deps)
    chunks = chunk_text(
        document.text,
        max_chars=deps.settings.chunk_max_chars,
        overlap_chars=deps.settings.chunk_overlap_chars,
    )
    known_types = [relation_type.name for relation_type in await deps.repo.list_relation_types()]
    extractions = await asyncio.gather(*(deps.llm.extract(chunk, known_types) for chunk in chunks))
    merged = merge_extractions(extractions)
    embeddings = await embed_article(merged, repo=deps.repo, llm=deps.llm)
    async with resolve_lock(deps.redis, deps.settings):
        resolved = await resolve_article(
            merged,
            embeddings,
            article_title=article.title,
            repo=deps.repo,
            llm=deps.llm,
            settings=deps.settings,
        )
        stats = ArticleStats(
            characters=len(resolved.characters),
            relations=len(resolved.relations),
            images_described=document.images_described,
        )
        await deps.repo.persist_article(
            article.id,
            characters=resolved.characters,
            relations=resolved.relations,
            relation_types=resolved.relation_types,
            not_same=resolved.not_same,
            stats=stats,
        )
    await _refresh_embeddings(article.id, resolved.characters, deps)
    return resolved, stats


async def _parse(article: Article, deps: PipelineDeps) -> ParsedDocument:
    data = await deps.blobs.load(article.id)
    if article.format is ArticleFormat.PDF:
        return await parse_pdf(data, llm=deps.llm, settings=deps.settings)
    return parse_markdown(data)


async def _refresh_embeddings(
    article_id: str, characters: list[CharacterUpsert], deps: PipelineDeps
) -> None:
    stale = [character for character in characters if character.embedding is None]
    if not stale:
        return
    try:
        vectors = await deps.llm.embed(
            [profile_embedding_text(character_profile(character)) for character in stale]
        )
        refreshed = await deps.repo.refresh_character_embeddings(
            [
                EmbeddingRefresh(
                    id=character.id,
                    name=character.name,
                    aliases=character.aliases,
                    description=character.description,
                    embedding=vector,
                )
                for character, vector in zip(stale, vectors, strict=True)
            ]
        )
    except Exception:
        logger.exception(
            "character_reembed_failed",
            article_id=article_id,
            character_ids=[character.id for character in stale],
        )
        return
    superseded = sorted({character.id for character in stale} - refreshed)
    if superseded:
        logger.info("character_reembed_superseded", article_id=article_id, character_ids=superseded)
