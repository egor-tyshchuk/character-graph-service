import os

import pytest
from pydantic import SecretStr, ValidationError

from app.auth.demo import DEMO_JWT_SECRET
from app.config import Settings


def test_settings_env_is_cleared_for_every_test() -> None:
    leaked = [name for name in os.environ if name.lower() in Settings.model_fields]
    assert leaked == []
    assert Settings(jwt_secret=SecretStr("x" * 32)).llm_model == "gpt-4.1-mini"


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
    assert s.dedup_judge_threshold == 0.65
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


def test_judge_call_budget_defaults() -> None:
    s = Settings(jwt_secret=SecretStr("x" * 32))
    assert s.judge_timeout_seconds == 20.0
    assert s.judge_max_retries == 1


@pytest.mark.parametrize(
    ("field", "value"), [("judge_timeout_seconds", 0), ("judge_max_retries", -1)]
)
def test_judge_call_budget_bounds_enforced(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        Settings.model_validate({"jwt_secret": "x" * 32, field: value})


@pytest.mark.parametrize(
    ("lock_timeout", "judge_calls", "judge_timeout", "judge_retries"),
    [(800, 10, 20.0, 1), (600, 10, 20.0, 1), (100, 5, 10.0, 0), (1, 1, 0.5, 0)],
)
def test_resolve_lock_timeout_must_exceed_judge_budget(
    lock_timeout: float, judge_calls: int, judge_timeout: float, judge_retries: int
) -> None:
    with pytest.raises(ValidationError, match="resolve_lock_timeout_seconds"):
        Settings(
            jwt_secret=SecretStr("x" * 32),
            resolve_lock_timeout_seconds=lock_timeout,
            dedup_max_judge_calls_per_article=judge_calls,
            judge_timeout_seconds=judge_timeout,
            judge_max_retries=judge_retries,
        )


def test_judge_budget_counts_sdk_retries_and_structured_output_attempts() -> None:
    with pytest.raises(ValidationError, match="judge_max_retries"):
        Settings(jwt_secret=SecretStr("x" * 32), judge_max_retries=2)
    s = Settings(jwt_secret=SecretStr("x" * 32), resolve_lock_timeout_seconds=801)
    assert s.resolve_lock_timeout_seconds == 801


def test_llm_timeout_and_retries_do_not_count_against_the_resolve_lock() -> None:
    s = Settings(jwt_secret=SecretStr("x" * 32), llm_timeout_seconds=600, llm_max_retries=6)
    assert s.resolve_lock_timeout_seconds == 900


def test_tiny_resolve_lock_timeout_allowed_with_tiny_judge_budget() -> None:
    s = Settings(
        jwt_secret=SecretStr("x" * 32),
        resolve_lock_timeout_seconds=0.6,
        dedup_max_judge_calls_per_article=2,
        judge_timeout_seconds=0.1,
        judge_max_retries=0,
    )
    assert s.resolve_lock_timeout_seconds == 0.6


def test_background_job_defaults() -> None:
    s = Settings(jwt_secret=SecretStr("x" * 32))
    assert s.reconcile_enabled is False
    assert s.reconcile_interval_minutes == 60
    assert s.reconcile_batch_size == 200
    assert s.reconcile_max_judge_calls == 50
    assert s.stuck_sweep_enabled is True
    assert s.stuck_sweep_interval_minutes == 10
    assert s.stuck_article_after_seconds == 4200


@pytest.mark.parametrize("field", ["reconcile_interval_minutes", "stuck_sweep_interval_minutes"])
@pytest.mark.parametrize("minutes", [1, 10, 15, 30, 60, 120, 360, 1440])
def test_job_interval_accepts_divisors_of_an_hour_or_a_day(field: str, minutes: int) -> None:
    s = Settings.model_validate({"jwt_secret": "x" * 32, field: minutes})
    assert getattr(s, field) == minutes


@pytest.mark.parametrize("field", ["reconcile_interval_minutes", "stuck_sweep_interval_minutes"])
@pytest.mark.parametrize("minutes", [0, -10, 7, 45, 90, 300, 2880])
def test_job_interval_rejects_uneven_schedules(field: str, minutes: int) -> None:
    with pytest.raises(ValidationError, match=field):
        Settings.model_validate({"jwt_secret": "x" * 32, field: minutes})


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_empty_openai_key_is_none(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    monkeypatch.setenv("OPENAI_API_KEY", blank)
    monkeypatch.setenv("OPENAI_BASE_URL", blank)
    s = Settings()
    assert s.openai_api_key is None
    assert s.openai_base_url is None


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_judge_model_is_none(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    monkeypatch.setenv("JUDGE_MODEL", blank)
    assert Settings().judge_model is None


def test_judge_model_is_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    monkeypatch.setenv("JUDGE_MODEL", "gpt-judge")
    assert Settings().judge_model == "gpt-judge"


def test_compose_dev_secret_is_accepted() -> None:
    s = Settings(jwt_secret=SecretStr(DEMO_JWT_SECRET))
    assert s.jwt_secret.get_secret_value() == DEMO_JWT_SECRET


def test_short_jwt_secret_is_rejected() -> None:
    with pytest.raises(ValidationError, match="jwt_secret"):
        Settings(jwt_secret=SecretStr("x" * 31))


@pytest.mark.parametrize("algorithm", ["HS256", "HS384", "HS512"])
def test_hmac_jwt_algorithms_are_accepted(algorithm: str) -> None:
    s = Settings.model_validate({"jwt_secret": "x" * 32, "jwt_algorithm": algorithm})
    assert s.jwt_algorithm == algorithm


@pytest.mark.parametrize("algorithm", ["none", "RS256", "hs256"])
def test_other_jwt_algorithms_are_rejected(algorithm: str) -> None:
    with pytest.raises(ValidationError, match="jwt_algorithm"):
        Settings.model_validate({"jwt_secret": "x" * 32, "jwt_algorithm": algorithm})


def test_new_hardening_defaults() -> None:
    s = Settings(jwt_secret=SecretStr("x" * 32))
    assert s.job_timeout_seconds == 3600
    assert s.max_chunks_per_article == 200
    assert s.refresh_family_max_lifetime_seconds == 604800


def test_stuck_article_threshold_must_exceed_job_timeout() -> None:
    with pytest.raises(ValidationError, match="stuck_article_after_seconds"):
        Settings(
            jwt_secret=SecretStr("x" * 32),
            job_timeout_seconds=3600,
            stuck_article_after_seconds=3600,
        )


def test_job_timeout_must_exceed_lock_acquire_plus_lock_timeouts() -> None:
    with pytest.raises(ValidationError, match="job_timeout_seconds"):
        Settings(
            jwt_secret=SecretStr("x" * 32),
            job_timeout_seconds=1500,
            stuck_article_after_seconds=4200,
            resolve_lock_acquire_timeout_seconds=600,
            resolve_lock_timeout_seconds=900,
        )


def test_judge_threshold_must_not_exceed_auto_merge_threshold() -> None:
    with pytest.raises(ValidationError, match="dedup_judge_threshold"):
        Settings(
            jwt_secret=SecretStr("x" * 32),
            dedup_judge_threshold=0.96,
            dedup_auto_merge_threshold=0.95,
        )


def test_equal_dedup_thresholds_disable_the_judge_band() -> None:
    s = Settings(
        jwt_secret=SecretStr("x" * 32), dedup_judge_threshold=0.9, dedup_auto_merge_threshold=0.9
    )
    assert s.dedup_judge_threshold == s.dedup_auto_merge_threshold


@pytest.mark.parametrize(
    ("field", "value"), [("job_max_tries", 0), ("job_retry_delay_seconds", -1)]
)
def test_job_retry_bounds_enforced(field: str, value: int) -> None:
    with pytest.raises(ValidationError, match=field):
        Settings.model_validate({"jwt_secret": "x" * 32, field: value})


def test_retry_delay_must_fit_between_job_timeout_and_stuck_sweep() -> None:
    with pytest.raises(ValidationError, match="job_retry_delay_seconds"):
        Settings(jwt_secret=SecretStr("x" * 32), job_retry_delay_seconds=600)
    accepted = Settings(jwt_secret=SecretStr("x" * 32), job_retry_delay_seconds=599)
    assert accepted.job_retry_delay_seconds == 599
