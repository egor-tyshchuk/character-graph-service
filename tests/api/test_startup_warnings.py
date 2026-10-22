from collections.abc import MutableMapping
from typing import Any

from asgi_lifespan import LifespanManager
from pydantic import SecretStr
from structlog.testing import capture_logs

from app.auth.demo import DEMO_APP_TOKEN, DEMO_JWT_SECRET
from app.auth.tokens import hash_app_token
from app.config import Settings
from app.main import create_app
from tests.fakes import FakeLLMClient


async def startup_warnings(
    settings: Settings, fake_llm: FakeLLMClient
) -> list[MutableMapping[str, Any]]:
    application = create_app(settings, llm=fake_llm)
    with capture_logs() as logs:
        async with LifespanManager(application):
            pass
    return [entry for entry in logs if entry["event"] == "insecure_demo_secret"]


async def test_api_startup_warns_on_the_demo_secret_and_app_token(
    settings: Settings, fake_llm: FakeLLMClient
) -> None:
    demo = settings.model_copy(
        update={
            "jwt_secret": SecretStr(DEMO_JWT_SECRET),
            "app_tokens": {**settings.app_tokens, "demo": hash_app_token(DEMO_APP_TOKEN)},
        }
    )

    warnings = await startup_warnings(demo, fake_llm)

    assert [entry["setting"] for entry in warnings] == ["JWT_SECRET", "APP_TOKENS"]


async def test_api_startup_is_silent_with_other_values(
    settings: Settings, fake_llm: FakeLLMClient
) -> None:
    assert await startup_warnings(settings, fake_llm) == []
