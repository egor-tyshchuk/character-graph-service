from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24


def fits_cron_grid(minutes: int) -> bool:
    if minutes <= 0:
        return False
    if minutes < MINUTES_PER_HOUR:
        return MINUTES_PER_HOUR % minutes == 0
    hours, remainder = divmod(minutes, MINUTES_PER_HOUR)
    return remainder == 0 and HOURS_PER_DAY % hours == 0


class ScheduleSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    reconcile_enabled: bool = False
    reconcile_interval_minutes: int = 60
    stuck_sweep_enabled: bool = True
    stuck_sweep_interval_minutes: int = 10

    @field_validator("reconcile_interval_minutes", "stuck_sweep_interval_minutes")
    @classmethod
    def _interval_fits_cron_grid(cls, value: int, info: ValidationInfo) -> int:
        if not fits_cron_grid(value):
            raise ValueError(
                f"{info.field_name} must divide 60 (e.g. 5, 10, 30) or be a whole number of "
                f"hours that divides 24 (e.g. 60, 180, 1440), got {value}"
            )
        return value


class Settings(ScheduleSettings):
    jwt_secret: SecretStr
    jwt_algorithm: str = "HS256"
    access_token_ttl_seconds: int = 900
    refresh_token_ttl_seconds: int = 86400
    app_tokens: Annotated[dict[str, str], NoDecode] = {}

    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("password")
    redis_url: str = "redis://redis:6379/0"

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    llm_model: str = "gpt-4.1-mini"
    vision_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    llm_max_concurrency: int = 4
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    job_queue: Literal["arq", "inline"] = "arq"
    upload_dir: Path = Path("/data/uploads")
    max_upload_bytes: int = 20 * 1024 * 1024

    chunk_max_chars: int = 6000
    chunk_overlap_chars: int = 500
    pdf_max_images: int = 20
    pdf_min_image_side: int = 100
    pdf_scanned_page_min_chars: int = 50
    pdf_render_dpi: int = 150
    vision_mode: Literal["auto", "all_pages"] = "auto"
    pdf_drawings_min_count: int = Field(default=20, ge=1)
    pdf_drawings_min_area_ratio: float = Field(default=0.12, gt=0, le=1)

    dedup_candidates: int = 5
    dedup_auto_merge_threshold: float = 0.95
    dedup_judge_threshold: float = 0.80
    dedup_max_judge_calls_per_article: int = Field(default=10, ge=0)
    relation_type_query_threshold: float = 0.7
    resolve_lock_timeout_seconds: float = Field(default=900, gt=0)
    resolve_lock_acquire_timeout_seconds: float = Field(default=600, gt=0)
    merge_lock_acquire_timeout_seconds: float = Field(default=10, gt=0)

    reconcile_batch_size: int = Field(default=200, ge=1)
    reconcile_max_judge_calls: int = Field(default=50, ge=0)
    stuck_article_after_seconds: int = Field(default=4200, gt=0)

    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("app_tokens", mode="before")
    @classmethod
    def _parse_app_tokens(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        tokens: dict[str, str] = {}
        for pair in filter(None, (part.strip() for part in value.split(","))):
            app_id, sep, digest = pair.partition(":")
            app_id, digest = app_id.strip(), digest.strip()
            if not sep or not app_id or not digest:
                raise ValueError(f"malformed app token entry {pair!r}, expected 'app_id:sha256hex'")
            tokens[app_id] = digest
        return tokens

    @model_validator(mode="after")
    def _resolve_lock_outlasts_judge_calls(self) -> Self:
        judge_budget = self.dedup_max_judge_calls_per_article * self.llm_timeout_seconds
        if self.resolve_lock_timeout_seconds <= judge_budget:
            raise ValueError(
                f"resolve_lock_timeout_seconds ({self.resolve_lock_timeout_seconds}) must exceed "
                "dedup_max_judge_calls_per_article * llm_timeout_seconds "
                f"({judge_budget})"
            )
        return self
