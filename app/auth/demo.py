import structlog

from app.auth.tokens import hash_app_token
from app.config import Settings

DEMO_JWT_SECRET = "dev-insecure-jwt-secret-change-me-please"
DEMO_APP_TOKEN = "demo-app-token"
DEMO_SECRET_EVENT = "insecure_demo_secret"

logger = structlog.get_logger(__name__)


def demo_app_ids(settings: Settings) -> list[str]:
    demo_digest = hash_app_token(DEMO_APP_TOKEN)
    return sorted(app_id for app_id, digest in settings.app_tokens.items() if digest == demo_digest)


def warn_on_demo_secrets(settings: Settings) -> None:
    if settings.jwt_secret.get_secret_value() == DEMO_JWT_SECRET:
        logger.warning(
            DEMO_SECRET_EVENT,
            setting="JWT_SECRET",
            detail="the docker-compose demo secret signs tokens; set a long random JWT_SECRET",
        )
    if app_ids := demo_app_ids(settings):
        logger.warning(
            DEMO_SECRET_EVENT,
            setting="APP_TOKENS",
            app_ids=app_ids,
            detail=f"the public demo app token {DEMO_APP_TOKEN!r} is accepted; remove its entry",
        )
