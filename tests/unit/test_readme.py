from pathlib import Path

import pytest

from app.llm import prompts

README = Path(__file__).resolve().parents[2] / "README.md"
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
