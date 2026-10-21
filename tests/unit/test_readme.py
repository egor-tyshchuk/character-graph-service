from pathlib import Path

import pytest

from app.llm.prompts import (
    DEDUP_JUDGE_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
    VISION_PROMPT,
)

README = Path(__file__).resolve().parents[2] / "README.md"


@pytest.mark.parametrize(
    "prompt",
    [EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE, VISION_PROMPT, DEDUP_JUDGE_PROMPT],
    ids=["extraction_system", "extraction_user", "vision", "dedup_judge"],
)
def test_readme_contains_every_prompt_verbatim(prompt: str) -> None:
    assert prompt in README.read_text(encoding="utf-8")
