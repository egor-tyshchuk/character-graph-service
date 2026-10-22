import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.llm import prompts

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
SETTINGS_DOCS = (REPO_ROOT / "docs" / "configuration.md", REPO_ROOT / ".env.example")
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
COMPOSE_DEMO_VALUES = frozenset({"jwt_secret", "app_tokens"})
PROMPT_SUFFIXES = ("_PROMPT", "_TEMPLATE")
PROMPT_NAMES = sorted(
    name
    for name, value in vars(prompts).items()
    if name.endswith(PROMPT_SUFFIXES) and isinstance(value, str)
)


def test_prompt_constants_are_discovered() -> None:
    assert {
        "EXTRACTION_SYSTEM_PROMPT",
        "EXTRACTION_USER_TEMPLATE",
        "VISION_PROMPT",
        "DEDUP_JUDGE_PROMPT",
    } <= set(PROMPT_NAMES)


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_readme_contains_every_prompt_verbatim(name: str) -> None:
    prompt = getattr(prompts, name)
    assert f"```text\n{prompt}\n```" in README.read_text(encoding="utf-8")


@pytest.mark.parametrize("document", SETTINGS_DOCS, ids=lambda path: path.name)
def test_every_setting_is_documented(document: Path) -> None:
    text = document.read_text(encoding="utf-8")
    undocumented = [name.upper() for name in Settings.model_fields if name.upper() not in text]
    assert undocumented == []


def test_compose_passes_every_setting_to_the_app() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    missing = [name.upper() for name in Settings.model_fields if f"  {name.upper()}: " not in text]
    assert missing == []


def _compose_defaults() -> dict[str, str]:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    return {
        name.lower(): default
        for name, default in re.findall(r"^\s+([A-Z_]+): \$\{\1:-([^}]*)\}$", text, re.MULTILINE)
    }


def _comparable(value: Any) -> Any:
    return value.get_secret_value() if isinstance(value, SecretStr) else value


def test_compose_defaults_match_the_settings_defaults() -> None:
    defaults = Settings(jwt_secret=SecretStr("x" * 32))
    compose = _compose_defaults()
    mismatched = {
        name: raw
        for name, raw in compose.items()
        if name not in COMPOSE_DEMO_VALUES
        and _comparable(getattr(Settings.model_validate({"jwt_secret": "x" * 32, name: raw}), name))
        != _comparable(getattr(defaults, name))
    }
    assert set(compose) >= {"llm_timeout_seconds", "judge_model", "vector_search_k"}
    assert mismatched == {}
