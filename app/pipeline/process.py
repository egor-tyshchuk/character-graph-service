import asyncio
import time

import structlog

from app.domain.models import ERROR_MAX_CHARS, Article, ArticleFormat, ArticleStats, ArticleStatus
from app.domain.profiles import character_profile, profile_embedding_text
from app.graph.upserts import CharacterUpsert, EmbeddingRefresh
from app.ingestion.chunking import chunk_text
from app.ingestion.markdown import parse_markdown
from app.ingestion.models import ParsedDocument
from app.ingestion.pdf import parse_pdf
from app.pipeline.deps import PipelineDeps
from app.pipeline.lock import resolve_lock
from app.pipeline.merge import merge_extractions
from app.pipeline.resolve import ResolvedArticle, embed_article, resolve_article

CANCELLED_ERROR = "Cancelled: job timeout or worker shutdown"

logger = structlog.get_logger(__name__)


async def process_article(article_id: str, deps: PipelineDeps) -> None:
    try:
        article = await deps.repo.get_article(article_id)
        if article is None:
            logger.warning("article_not_found", article_id=article_id)
        elif article.status is ArticleStatus.DONE:
            logger.info("article_already_done", article_id=article_id)
        else:
            await _process_tracked(article, deps)
    finally:
        await deps.blobs.delete(article_id)


async def _process_tracked(article: Article, deps: PipelineDeps) -> None:
    started = time.perf_counter()
    try:
        await deps.repo.set_article_status(article.id, ArticleStatus.PROCESSING)
        resolved, stats = await _ingest(article, deps)
    except asyncio.CancelledError:
        logger.warning("article_processing_cancelled", article_id=article.id)
        await asyncio.shield(deps.repo.mark_article_failed(article.id, CANCELLED_ERROR))
        raise
    except Exception as exc:
        logger.exception("article_processing_failed", article_id=article.id)
        error = f"{type(exc).__name__}: {exc}"[:ERROR_MAX_CHARS]
        await deps.repo.mark_article_failed(article.id, error)
        return
    await _refresh_embeddings(article.id, resolved.characters, deps)
    logger.info(
        "article_processed",
        article_id=article.id,
        characters=stats.characters,
        matched_existing=sum(c.embedding is None for c in resolved.characters),
        relations=stats.relations,
        relation_types_created=len(resolved.relation_types),
        not_same_recorded=len(resolved.not_same),
        images_described=stats.images_described,
        duration_seconds=round(time.perf_counter() - started, 3),
    )


async def _ingest(article: Article, deps: PipelineDeps) -> tuple[ResolvedArticle, ArticleStats]:
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
    async with resolve_lock(deps.redis, deps.settings, article_id=article.id) as held:
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
        await held.ensure_owned()
        await deps.repo.persist_article(
            article.id,
            characters=resolved.characters,
            relations=resolved.relations,
            relation_types=resolved.relation_types,
            not_same=resolved.not_same,
            stats=stats,
        )
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
