import re
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from structlog.testing import capture_logs

from app.auth.demo import DEMO_APP_TOKEN, DEMO_JWT_SECRET, warn_on_demo_secrets
from app.auth.tokens import hash_app_token
from app.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
OTHER_SECRET = SecretStr("a-real-secret-a-real-secret-a-real")
OTHER_DIGEST = hash_app_token("a-real-app-token")


def make_settings(**overrides: Any) -> Settings:
    return Settings.model_validate({"jwt_secret": OTHER_SECRET, **overrides})


def demo_warnings(settings: Settings) -> list[MutableMapping[str, Any]]:
    with capture_logs() as logs:
        warn_on_demo_secrets(settings)
    warnings = [entry for entry in logs if entry["event"] == "insecure_demo_secret"]
    for warning in warnings:
        assert DEMO_JWT_SECRET not in repr(warning)
        assert DEMO_APP_TOKEN not in repr(warning)
    return warnings


def test_warns_with_the_demo_jwt_secret() -> None:
    [warning] = demo_warnings(make_settings(jwt_secret=SecretStr(DEMO_JWT_SECRET)))

    assert warning["log_level"] == "warning"
    assert warning["setting"] == "JWT_SECRET"


def test_warns_with_the_demo_app_token() -> None:
    settings = make_settings(
        app_tokens={"demo": hash_app_token(DEMO_APP_TOKEN), "django": OTHER_DIGEST}
    )

    [warning] = demo_warnings(settings)

    assert warning["log_level"] == "warning"
    assert warning["setting"] == "APP_TOKENS"
    assert warning["app_ids"] == ["demo"]


def test_silent_with_other_values() -> None:
    settings = make_settings(app_tokens={"django": OTHER_DIGEST})

    assert demo_warnings(settings) == []


@pytest.mark.parametrize("path", ["docker-compose.yml", ".env.example"])
def test_demo_values_match_the_shipped_defaults(path: str) -> None:
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    secret = re.search(r"JWT_SECRET(?::-|=)([^}\s]+)", text)
    app_tokens = re.search(r"APP_TOKENS(?::-|=)([^}\s]+)", text)

    assert secret is not None
    assert secret.group(1) == DEMO_JWT_SECRET
    assert app_tokens is not None
    assert app_tokens.group(1) == f"demo:{hash_app_token(DEMO_APP_TOKEN)}"
