from fastapi import APIRouter

from app.api.deps import ContainerDep
from app.api.schemas import ErrorResponse, RefreshRequest, TokenRequest
from app.auth.tokens import TokenPair

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/token",
    summary="Exchange an app token for an access/refresh token pair",
    responses={401: {"model": ErrorResponse}},
)
async def issue_token(body: TokenRequest, container: ContainerDep) -> TokenPair:
    app_id = container.tokens.authenticate_app(body.app_token)
    return await container.tokens.issue_pair(app_id)


@router.post(
    "/refresh",
    summary="Rotate a refresh token; reuse of a rotated token revokes the whole family",
    responses={401: {"model": ErrorResponse}},
)
async def refresh_token(body: RefreshRequest, container: ContainerDep) -> TokenPair:
    return await container.tokens.refresh(body.refresh_token)
