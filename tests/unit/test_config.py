import pytest
from pydantic import ValidationError

from app.config import Settings


def test_app_tokens_parsed_from_comma_separated_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    monkeypatch.setenv("APP_TOKENS", "django:abc123, other:def456")
    s = Settings()
    assert s.app_tokens == {"django": "abc123", "other": "def456"}


def test_app_tokens_rejects_malformed_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    monkeypatch.setenv("APP_TOKENS", "no-colon-here")
    with pytest.raises(ValidationError):
        Settings()
