from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import ContainerDep, Protected, require_article
from app.api.schemas import ErrorResponse, MergeCharactersRequest, Page
from app.domain.models import Character
from app.pipeline.merge_characters import merge_characters

router = APIRouter(prefix="/characters", tags=["characters"])


@router.get(
    "",
    summary="All characters, ordered by name",
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def list_characters(
    container: ContainerDep,
    _: Protected,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[Character]:
    items, total = await container.repo.list_characters(limit=limit, offset=offset)
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.get(
    "/by_article/{article_id}",
    summary="Characters appearing in an article, ordered by name",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def list_characters_by_article(
    article_id: str, container: ContainerDep, _: Protected
) -> list[Character]:
    await require_article(container.repo, article_id)
    return await container.repo.list_characters_by_article(article_id)


@router.post(
    "/merge",
    summary="Merge duplicate characters into a target character",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def merge_duplicate_characters(
    request: MergeCharactersRequest, container: ContainerDep, _: Protected
) -> Character:
    return await merge_characters(request.target_id, request.source_ids, container.pipeline_deps)
