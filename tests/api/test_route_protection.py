import re

import httpx
from fastapi import FastAPI
from starlette.routing import Route

OPEN_PATHS = {
    "/auth/token",
    "/auth/refresh",
    "/docs",
    "/docs/oauth2-redirect",
    "/openapi.json",
    "/redoc",
}
OPEN_PREFIXES = ("/health/",)
PATH_PARAMETER = re.compile(r"\{[^}]+\}")
IMPLICIT_METHODS = {"HEAD", "OPTIONS"}
HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def _is_open(path: str) -> bool:
    return path in OPEN_PATHS or path.startswith(OPEN_PREFIXES)


def _endpoints(app: FastAPI) -> list[tuple[str, str]]:
    documented = {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method.upper() in HTTP_METHODS
    }
    mounted_directly = {
        (method, route.path)
        for route in app.routes
        if isinstance(route, Route)
        for method in (route.methods or set()) - IMPLICIT_METHODS
    }
    return sorted(documented | mounted_directly)


def test_only_auth_health_and_docs_routes_are_open(app: FastAPI) -> None:
    open_paths = {path for _, path in _endpoints(app) if _is_open(path)}

    assert open_paths == {*OPEN_PATHS, "/health/live", "/health/ready"}


async def test_every_other_route_rejects_requests_without_a_token(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    protected = [(method, path) for method, path in _endpoints(app) if not _is_open(path)]
    assert len(protected) >= 8

    for method, path in protected:
        response = await client.request(method, PATH_PARAMETER.sub("some-id", path))

        assert (method, path, response.status_code) == (method, path, 401)
        assert response.json()["error"]["code"] == "unauthorized"
        assert response.headers["www-authenticate"] == "Bearer"
