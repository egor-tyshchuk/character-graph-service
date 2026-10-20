import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, Query, Request, Response

from app.api.deps import get_container, require_access
from app.api.schemas import ErrorResponse, TextUploadRequest
from app.api.uploads import UploadedArticle, read_upload
from app.auth.tokens import AccessClaims
from app.container import Container
from app.domain.models import Article, ArticleStatus
from app.errors import NotFoundError
from app.graph.repository import DuplicateContentError, GraphRepository
from app.pipeline.process import ERROR_MAX_CHARS

router = APIRouter(prefix="/articles", tags=["articles"])
logger = structlog.get_logger(__name__)

POLL_INTERVAL_SECONDS = 0.5
FINISHED = frozenset({ArticleStatus.DONE, ArticleStatus.FAILED})

_UPLOAD_REQUEST_BODY: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {"schema": TextUploadRequest.model_json_schema()},
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Markdown text"},
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "PDF, Markdown or plain-text file",
                        },
                        "title": {"type": "string"},
                    },
                }
            },
        },
    }
}

Protected = Annotated[AccessClaims, Depends(require_access)]
ContainerDep = Annotated[Container, Depends(get_container)]


@router.post(
    "/upload",
    status_code=202,
    summary="Upload a Markdown text or a PDF/Markdown file for character extraction",
    description=(
        "202: accepted for processing. 200: identical content already exists, or `wait=true` "
        "and processing finished within `wait_timeout` seconds."
    ),
    responses={
        200: {"model": Article},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    openapi_extra=_UPLOAD_REQUEST_BODY,
)
async def upload_article(
    request: Request,
    response: Response,
    container: ContainerDep,
    _: Protected,
    wait: bool = False,
    wait_timeout: Annotated[float, Query(ge=1, le=120)] = 60,
) -> Article:
    upload = await read_upload(request, container.settings.max_upload_bytes)
    article, accepted = await _submit(upload, container)
    if wait:
        article = await _wait_until_finished(container.repo, article.id, wait_timeout)
        response.status_code = 200 if article.status in FINISHED else 202
    else:
        response.status_code = 202 if accepted else 200
    return article


@router.get(
    "/{article_id}",
    summary="Article processing status, error and stats",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def get_article(article_id: str, container: ContainerDep, _: Protected) -> Article:
    return await _require_article(container.repo, article_id)


async def _submit(upload: UploadedArticle, container: Container) -> tuple[Article, bool]:
    existing = await container.repo.get_article_by_hash(upload.content_hash)
    if existing is None:
        return await _create(upload, container)
    if existing.status is not ArticleStatus.FAILED:
        return existing, False
    if not await container.repo.reset_failed_article(existing.id):
        return await _require_article(container.repo, existing.id), False
    requeued = await _require_article(container.repo, existing.id)
    async with _fail_article_on_error(container, existing.id):
        await container.blobs.save(existing.id, upload.data)
        await container.queue.enqueue_article(existing.id)
    return requeued, True


async def _create(upload: UploadedArticle, container: Container) -> tuple[Article, bool]:
    now = datetime.now(UTC)
    article = Article(
        id=str(uuid.uuid4()),
        title=await upload.resolve_title(),
        format=upload.format,
        content_hash=upload.content_hash,
        status=ArticleStatus.PENDING,
        created_at=now,
        updated_at=now,
    )
    await container.blobs.save(article.id, upload.data)
    try:
        await container.repo.create_article(article)
    except DuplicateContentError as duplicate:
        await container.blobs.delete(article.id)
        return await _require_article(container.repo, duplicate.existing_id), False
    except BaseException:
        await container.blobs.delete(article.id)
        raise
    async with _fail_article_on_error(container, article.id):
        await container.queue.enqueue_article(article.id)
    return article, True


@asynccontextmanager
async def _fail_article_on_error(container: Container, article_id: str) -> AsyncIterator[None]:
    try:
        yield
    except Exception as exc:
        logger.warning("article_dispatch_failed", article_id=article_id, error=repr(exc))
        error = f"Dispatch failed: {type(exc).__name__}: {exc}"[:ERROR_MAX_CHARS]
        await _best_effort(container.blobs.delete(article_id), "blob_cleanup_failed", article_id)
        await _best_effort(
            container.repo.mark_article_failed(article_id, error),
            "article_mark_failed_failed",
            article_id,
        )
        raise


async def _best_effort(action: Awaitable[object], event: str, article_id: str) -> None:
    try:
        await action
    except Exception:
        logger.exception(event, article_id=article_id)


async def _wait_until_finished(
    repo: GraphRepository, article_id: str, wait_seconds: float
) -> Article:
    deadline = time.monotonic() + wait_seconds
    while True:
        article = await _require_article(repo, article_id)
        remaining = deadline - time.monotonic()
        if article.status in FINISHED or remaining <= 0:
            return article
        await asyncio.sleep(min(POLL_INTERVAL_SECONDS, remaining))


async def _require_article(repo: GraphRepository, article_id: str) -> Article:
    article = await repo.get_article(article_id)
    if article is None:
        raise NotFoundError(f"Article {article_id} not found")
    return article
