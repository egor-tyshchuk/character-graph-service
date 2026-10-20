from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.api.schemas import ErrorResponse, RefreshRequest, TokenRequest
from app.auth.tokens import TokenPair
from app.container import Container

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/token",
    summary="Exchange an app token for an access/refresh token pair",
    responses={401: {"model": ErrorResponse}},
)
async def issue_token(
    body: TokenRequest, container: Annotated[Container, Depends(get_container)]
) -> TokenPair:
    app_id = container.tokens.authenticate_app(body.app_token)
    return await container.tokens.issue_pair(app_id)


@router.post(
    "/refresh",
    summary="Rotate a refresh token; reuse of a rotated token revokes the whole family",
    responses={401: {"model": ErrorResponse}},
)
async def refresh_token(
    body: RefreshRequest, container: Annotated[Container, Depends(get_container)]
) -> TokenPair:
    return await container.tokens.refresh(body.refresh_token)
