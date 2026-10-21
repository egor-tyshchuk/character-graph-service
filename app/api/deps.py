from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.tokens import AccessClaims
from app.container import Container
from app.domain.models import Article
from app.errors import AuthError, NotFoundError
from app.graph.repository import GraphRepository

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


async def require_access(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    container: Annotated[Container, Depends(get_container)],
) -> AccessClaims:
    if credentials is None:
        raise AuthError("Missing bearer token")
    return container.tokens.verify_access(credentials.credentials)


Protected = Annotated[AccessClaims, Depends(require_access)]
ContainerDep = Annotated[Container, Depends(get_container)]


async def require_article(repo: GraphRepository, article_id: str) -> Article:
    article = await repo.get_article(article_id)
    if article is None:
        raise NotFoundError(f"Article {article_id} not found")
    return article
