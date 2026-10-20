import pytest
from pydantic import SecretStr, ValidationError

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


def test_threshold_defaults() -> None:
    s = Settings(jwt_secret=SecretStr("x" * 32))
    assert s.dedup_auto_merge_threshold == 0.95
    assert s.relation_type_query_threshold == 0.7


def test_vision_mode_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    monkeypatch.setenv("VISION_MODE", "every_page")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pdf_drawings_min_count", 0),
        ("pdf_drawings_min_area_ratio", 0.0),
        ("pdf_drawings_min_area_ratio", 1.5),
    ],
)
def test_pdf_drawings_bounds_enforced(
    monkeypatch: pytest.MonkeyPatch, field: str, value: float
) -> None:
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    monkeypatch.setenv(field.upper(), str(value))
    with pytest.raises(ValidationError):
        Settings()
