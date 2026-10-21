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


def test_resolve_lock_defaults_outlast_the_judge_budget() -> None:
    s = Settings(jwt_secret=SecretStr("x" * 32))
    assert s.resolve_lock_timeout_seconds == 900
    assert s.resolve_lock_acquire_timeout_seconds == 600
    assert s.merge_lock_acquire_timeout_seconds == 10
    assert s.dedup_max_judge_calls_per_article == 10
    assert not hasattr(s, "relation_type_merge_threshold")


def test_merge_lock_acquire_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="merge_lock_acquire_timeout_seconds"):
        Settings(jwt_secret=SecretStr("x" * 32), merge_lock_acquire_timeout_seconds=0)


@pytest.mark.parametrize(
    ("lock_timeout", "judge_calls", "llm_timeout"),
    [(600, 10, 60.0), (100, 10, 60.0), (1, 1, 1.0)],
)
def test_resolve_lock_timeout_must_exceed_judge_budget(
    lock_timeout: float, judge_calls: int, llm_timeout: float
) -> None:
    with pytest.raises(ValidationError, match="resolve_lock_timeout_seconds"):
        Settings(
            jwt_secret=SecretStr("x" * 32),
            resolve_lock_timeout_seconds=lock_timeout,
            dedup_max_judge_calls_per_article=judge_calls,
            llm_timeout_seconds=llm_timeout,
        )


def test_tiny_resolve_lock_timeout_allowed_with_tiny_judge_budget() -> None:
    s = Settings(
        jwt_secret=SecretStr("x" * 32),
        resolve_lock_timeout_seconds=0.6,
        dedup_max_judge_calls_per_article=2,
        llm_timeout_seconds=0.1,
    )
    assert s.resolve_lock_timeout_seconds == 0.6
