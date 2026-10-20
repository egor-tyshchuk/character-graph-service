from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_container, require_access
from app.api.schemas import ConnectedResponse, ErrorResponse
from app.auth.tokens import AccessClaims
from app.container import Container
from app.domain.models import RelationTypeInfo
from app.errors import NotFoundError
from app.pipeline.merge import normalize_relation_type

router = APIRouter(prefix="/relations", tags=["relations"])

Protected = Annotated[AccessClaims, Depends(require_access)]
ContainerDep = Annotated[Container, Depends(get_container)]
RelationTypeQuery = Annotated[
    str,
    Query(
        alias="type",
        min_length=1,
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
    },
)
async def find_connected(
    character_id: str, relation_type: RelationTypeQuery, container: ContainerDep, _: Protected
) -> ConnectedResponse:
    if await container.repo.get_character(character_id) is None:
        raise NotFoundError(f"Character {character_id} not found")
    resolved = await _resolve_relation_type(relation_type, container)
    items = [] if resolved is None else await container.repo.find_connected(character_id, resolved)
    return ConnectedResponse(
        character_id=character_id,
        requested_type=relation_type,
        resolved_type=resolved,
        items=items,
    )


async def _resolve_relation_type(requested: str, container: Container) -> str | None:
    normalized = normalize_relation_type(requested)
    if normalized and normalized in await container.repo.existing_relation_types([normalized]):
        return normalized
    [embedding] = await container.llm.embed([requested])
    for name, similarity in await container.repo.find_similar_relation_types(embedding, 1):
        if similarity >= container.settings.relation_type_query_threshold:
            return name
    return None
