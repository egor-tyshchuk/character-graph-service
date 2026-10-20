from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

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
    pdf_drawings_min_count: int = 20
    pdf_drawings_min_area_ratio: float = 0.12

    dedup_candidates: int = 5
    dedup_auto_merge_threshold: float = 0.95
    dedup_judge_threshold: float = 0.80
    relation_type_merge_threshold: float = 0.85
    relation_type_query_threshold: float = 0.7
    resolve_lock_timeout_seconds: int = 120

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
