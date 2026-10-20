from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_container, require_access
from app.api.schemas import ErrorResponse, Page
from app.auth.tokens import AccessClaims
from app.container import Container
from app.domain.models import Character
from app.errors import NotFoundError

router = APIRouter(prefix="/characters", tags=["characters"])

Protected = Annotated[AccessClaims, Depends(require_access)]
ContainerDep = Annotated[Container, Depends(get_container)]


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
    if await container.repo.get_article(article_id) is None:
        raise NotFoundError(f"Article {article_id} not found")
    return await container.repo.list_characters_by_article(article_id)
