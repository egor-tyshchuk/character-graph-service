from typing import Any, Literal

import pymupdf
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.ingestion.models import DocumentParseError
from app.ingestion.pdf import parse_pdf
from app.llm.client import LLMError, LLMNotConfiguredError, UnconfiguredLLMClient
from tests.fakes import FakeLLMClient
from tests.pdf_factory import (
    PageSpec,
    filled_axis_polygons,
    filled_shapes,
    make_pdf,
    make_png,
    org_chart,
    rotated_rectangles,
    set_title,
    table_grid,
)

LONG_TEXT = (
    "Anna Petrova is a surgeon in Kazan. Her brother Mark Petrov teaches history. "
    "They meet every summer at their grandmother's dacha."
)


class FailingFirstImageLLM(FakeLLMClient):
    async def describe_image(self, image_png: bytes, page_context: str) -> str:
        failed_before = bool(self.image_calls)
        description = await super().describe_image(image_png, page_context)
        if not failed_before:
            raise LLMError("vision request failed")
        return description


def _distinct_images(count: int) -> list[bytes]:
    return [make_png(200 + index * 10, 200, f"image {index}") for index in range(count)]


def _settings(
    *, pdf_max_images: int = 20, vision_mode: Literal["auto", "all_pages"] = "auto"
) -> Settings:
    return Settings(
        jwt_secret=SecretStr("unit-test-secret-unit-test-secret"),
        pdf_max_images=pdf_max_images,
        vision_mode=vision_mode,
    )


async def test_text_pdf_without_images() -> None:
    llm = FakeLLMClient()
    data = make_pdf(
        [PageSpec(text=LONG_TEXT), PageSpec(text=f"Second page about Mark Petrov. {LONG_TEXT}")]
    )

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert "Anna Petrova is a surgeon" in document.text
    assert "Second page about Mark Petrov." in document.text
    assert document.images_described == 0
    assert document.title is None
    assert llm.image_calls == []


async def test_embedded_image_described_inline() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(text=LONG_TEXT, images=[make_png(300, 300, "siblings")])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert "[Image on page 1: A photo of Anna Petrova" in document.text
    assert document.text.index("Anna Petrova is a surgeon") < document.text.index("[Image on")
    assert document.images_described == 1
    assert len(llm.image_calls) == 1
    assert "Anna Petrova is a surgeon" in llm.image_calls[0]


async def test_small_images_skipped() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(text=LONG_TEXT, images=[make_png(50, 50)])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert llm.image_calls == []
    assert document.images_described == 0
    assert "[Image on page" not in document.text


async def test_scanned_page_rendered_once() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(text=LONG_TEXT), PageSpec(images=[make_png(300, 300, "scan")])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert "[Image on page 2:" in document.text
    assert "[Image on page 1:" not in document.text
    assert document.images_described == 1


async def test_scanned_first_page_block() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(images=[make_png(300, 300, "scan")])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert document.text.startswith("[Image on page 1:")


async def test_repeated_image_described_once() -> None:
    llm = FakeLLMClient()
    png = make_png(300, 300, "logo")
    data = make_pdf([PageSpec(text=f"{LONG_TEXT} Page {n}.", images=[png]) for n in range(3)])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert document.images_described == 1
    assert document.text.count("[Image on page") == 1


async def test_image_cap_respected() -> None:
    llm = FakeLLMClient()
    images = _distinct_images(5)
    data = make_pdf([PageSpec(text=LONG_TEXT, images=images)])

    document = await parse_pdf(data, llm=llm, settings=_settings(pdf_max_images=2))

    assert len(llm.image_calls) == 2
    assert document.images_described == 2


async def test_title_from_metadata() -> None:
    data = set_title(make_pdf([PageSpec(text=LONG_TEXT)]), "The Petrovs")

    document = await parse_pdf(data, llm=FakeLLMClient(), settings=_settings())

    assert document.title == "The Petrovs"


async def test_corrupt_pdf_raises() -> None:
    with pytest.raises(DocumentParseError):
        await parse_pdf(b"%PDF-1.7 garbage", llm=FakeLLMClient(), settings=_settings())


async def test_encrypted_pdf_raises() -> None:
    data = make_pdf([PageSpec(text=LONG_TEXT)], password="x")

    with pytest.raises(DocumentParseError):
        await parse_pdf(data, llm=FakeLLMClient(), settings=_settings())


async def test_blank_page_skipped_and_not_counted_toward_cap() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(), PageSpec(images=[make_png(300, 300, "scan")])])

    document = await parse_pdf(data, llm=llm, settings=_settings(pdf_max_images=1))

    assert len(llm.image_calls) == 1
    assert document.images_described == 1
    assert document.text == (
        "[Image on page 2: A photo of Anna Petrova and Mark Petrov, captioned 'siblings'.]"
    )


async def test_page_level_failure_raises_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_get_text(self: Any, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("corrupt content stream")

    monkeypatch.setattr(pymupdf.Page, "get_text", broken_get_text)
    data = make_pdf([PageSpec(text=LONG_TEXT)])

    with pytest.raises(DocumentParseError, match="corrupt content stream"):
        await parse_pdf(data, llm=FakeLLMClient(), settings=_settings())


async def test_broken_image_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    original_tobytes: Any = pymupdf.Pixmap.tobytes
    encodings: list[str] = []

    def flaky_tobytes(self: Any, output: str = "png", *args: Any) -> bytes:
        encodings.append(output)
        if len(encodings) == 1:
            raise RuntimeError("unsupported image filter")
        result: bytes = original_tobytes(self, output, *args)
        return result

    data = make_pdf([PageSpec(text=LONG_TEXT, images=_distinct_images(2))])
    monkeypatch.setattr(pymupdf.Pixmap, "tobytes", flaky_tobytes)
    llm = FakeLLMClient()

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert encodings == ["png", "png"]
    assert len(llm.image_calls) == 1
    assert document.images_described == 1
    assert "Anna Petrova is a surgeon" in document.text


async def test_vision_failure_skips_image() -> None:
    llm = FailingFirstImageLLM()
    data = make_pdf([PageSpec(text=LONG_TEXT, images=_distinct_images(2))])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 2
    assert document.images_described == 1
    assert document.text.count("[Image on page 1:") == 1


async def test_unconfigured_llm_propagates() -> None:
    data = make_pdf([PageSpec(text=LONG_TEXT, images=_distinct_images(1))])

    with pytest.raises(LLMNotConfiguredError):
        await parse_pdf(data, llm=UnconfiguredLLMClient(), settings=_settings())


async def test_vector_diagram_on_text_page_is_rendered() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30)])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert "Anna Petrova is a surgeon" in llm.image_calls[0]
    assert document.images_described == 1
    assert "[Image on page 1:" in document.text


async def test_bordered_table_does_not_trigger_vision() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(text=LONG_TEXT, drawings=[table_grid(10, 10)])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert llm.image_calls == []
    assert document.images_described == 0


async def test_small_graphic_below_area_ratio_ignored() -> None:
    llm = FakeLLMClient()
    small_region = (50.0, 420.0, 200.0, 520.0)
    data = make_pdf([PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30, small_region)])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert llm.image_calls == []
    assert document.images_described == 0


async def test_few_large_shapes_below_count_ignored() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(text=LONG_TEXT, drawings=[filled_shapes(6)])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert llm.image_calls == []
    assert document.images_described == 0


async def test_graphical_page_absorbs_its_embedded_images() -> None:
    llm = FakeLLMClient()
    png = make_png(300, 300, "siblings")
    data = make_pdf(
        [
            PageSpec(text=LONG_TEXT, images=[png], drawings=[filled_shapes(30)]),
            PageSpec(text=f"{LONG_TEXT} Again.", images=[png]),
        ]
    )

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert document.images_described == 1
    assert "[Image on page 1:" in document.text
    assert "[Image on page 2:" not in document.text


async def test_graphical_page_counts_toward_cap() -> None:
    llm = FakeLLMClient()
    data = make_pdf(
        [
            PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30)]),
            PageSpec(text=LONG_TEXT, images=_distinct_images(1)),
        ]
    )

    document = await parse_pdf(data, llm=llm, settings=_settings(pdf_max_images=1))

    assert len(llm.image_calls) == 1
    assert "[Image on page 1:" in document.text
    assert "[Image on page 2:" not in document.text


async def test_low_text_org_chart_page_is_rendered() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(drawings=[org_chart()])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert document.text.startswith("[Image on page 1:")


async def test_diagram_on_rotated_page_is_rendered() -> None:
    llm = FakeLLMClient()
    lower_region = (50.0, 620.0, 545.0, 840.0)
    data = make_pdf(
        [PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30, lower_region)], rotation=90)]
    )

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert document.images_described == 1


async def test_filled_axis_line_polygons_are_significant() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(text=LONG_TEXT, drawings=[filled_axis_polygons(30)])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert document.images_described == 1


async def test_unfilled_rectangular_quads_are_decorative() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(text=LONG_TEXT, drawings=[rotated_rectangles(30)])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert llm.image_calls == []
    assert document.images_described == 0


async def test_low_text_page_with_a_shape_is_rendered() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(drawings=[filled_shapes(1)])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert document.text.startswith("[Image on page 1:")


async def test_all_pages_mode_renders_every_non_blank_page() -> None:
    llm = FakeLLMClient()
    data = make_pdf(
        [
            PageSpec(text=LONG_TEXT),
            PageSpec(),
            PageSpec(text=LONG_TEXT, images=_distinct_images(2)),
        ]
    )

    document = await parse_pdf(data, llm=llm, settings=_settings(vision_mode="all_pages"))

    assert len(llm.image_calls) == 2
    assert document.images_described == 2
    assert document.text.count("[Image on page 1:") == 1
    assert "[Image on page 2:" not in document.text
    assert document.text.count("[Image on page 3:") == 1
