from pathlib import Path
from typing import Any, cast

import pymupdf
from pydantic import SecretStr

from app.config import Settings
from app.ingestion.pdf import _is_graphical, _significant_drawings, parse_pdf
from examples.make_sample_pdf import build_sample_pdf
from tests.fakes import FakeLLMClient

mupdf = cast(Any, pymupdf)
SAMPLE_PDF = Path(__file__).resolve().parents[2] / "examples" / "sample.pdf"
SETTINGS = Settings(jwt_secret=SecretStr("unit-test-secret-unit-test-secret"))


def test_committed_sample_matches_the_generator() -> None:
    assert build_sample_pdf() == SAMPLE_PDF.read_bytes()


def test_diagram_page_is_vector_graphics_without_a_text_layer() -> None:
    with mupdf.open(stream=SAMPLE_PDF.read_bytes(), filetype="pdf") as doc:
        assert doc.page_count == 3
        diagram = doc[2]
        drawings = _significant_drawings(diagram)
        assert diagram.get_text().strip() == ""
        assert diagram.get_images() == []
        assert len(drawings) >= SETTINGS.pdf_drawings_min_count
        assert _is_graphical(diagram, drawings, SETTINGS)


async def test_sample_sends_the_photo_and_the_diagram_to_vision() -> None:
    llm = FakeLLMClient()

    document = await parse_pdf(SAMPLE_PDF.read_bytes(), llm=llm, settings=SETTINGS)

    assert len(llm.image_calls) == 2
    assert (document.images_described, document.images_failed, document.images_skipped) == (
        2,
        0,
        0,
    )
    assert "[Image on page 2:" in document.text
    assert "[Image on page 3:" in document.text
