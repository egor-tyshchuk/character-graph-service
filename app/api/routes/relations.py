from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import ContainerDep, Protected
from app.api.schemas import ConnectedResponse, ErrorResponse
from app.domain.models import RelationTypeInfo
from app.errors import NotFoundError
from app.pipeline.relation_types import resolve_relation_type

router = APIRouter(prefix="/relations", tags=["relations"])

RELATION_TYPE_QUERY_MAX_CHARS = 100
RelationTypeQuery = Annotated[
    str,
    Query(
        alias="type",
        min_length=1,
        max_length=RELATION_TYPE_QUERY_MAX_CHARS,
        pattern=r"\S",
        description="Relation type in any wording or language, e.g. FRIEND_OF, friend of, друг",
    ),
]


@router.get(
    "/types",
    summary="Relation types in use, most used first",
    responses={401: {"model": ErrorResponse}},
)
async def list_relation_types(container: ContainerDep, _: Protected) -> list[RelationTypeInfo]:
    return await container.repo.list_relation_types()


@router.get(
    "/connected/{character_id}",
    summary="Characters connected to a character by a relation type",
    description=(
        "The type is matched exactly after normalization to UPPER_SNAKE_CASE, otherwise "
        "semantically against known relation types. `resolved_type` is null and `items` is "
        "empty when no known type is similar enough."
    ),
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def find_connected(
    character_id: str, relation_type: RelationTypeQuery, container: ContainerDep, _: Protected
) -> ConnectedResponse:
    if await container.repo.get_character(character_id) is None:
        raise NotFoundError(f"Character {character_id} not found")
    resolved = await resolve_relation_type(relation_type, container.pipeline_deps)
    items = [] if resolved is None else await container.repo.find_connected(character_id, resolved)
    return ConnectedResponse(
        character_id=character_id,
        requested_type=relation_type,
        resolved_type=resolved,
        items=items,
    )
