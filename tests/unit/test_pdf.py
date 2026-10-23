from typing import Any, Literal, cast

import pymupdf
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.ingestion.models import DocumentParseError
from app.ingestion.pdf import parse_pdf, read_pdf_title
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

mupdf = cast(Any, pymupdf)
LONG_TEXT = (
    "Anna Petrova is a surgeon in Kazan. Her brother Mark Petrov teaches history. "
    "They meet every summer at their grandmother's dacha."
)


class RecordingImageLLM(FakeLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.pngs: list[bytes] = []

    async def describe_image(self, image_png: bytes, page_context: str) -> str:
        self.pngs.append(image_png)
        return await super().describe_image(image_png, page_context)


def png_size(png: bytes) -> tuple[int, int]:
    pixmap = mupdf.Pixmap(png)
    return pixmap.width, pixmap.height


def scanned_page_pdf(width: float, height: float) -> bytes:
    with mupdf.open() as doc:
        page = doc.new_page(width=width, height=height)
        page.insert_image(mupdf.Rect(0, 0, width, height), stream=make_png(300, 300, "scan"))
        data: bytes = doc.tobytes()
    return data


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
    *,
    pdf_max_images: int = 20,
    vision_mode: Literal["auto", "all_pages"] = "auto",
    pdf_max_render_pixels: int = 25_000_000,
) -> Settings:
    return Settings(
        jwt_secret=SecretStr("unit-test-secret-unit-test-secret"),
        pdf_max_images=pdf_max_images,
        vision_mode=vision_mode,
        pdf_max_render_pixels=pdf_max_render_pixels,
    )


async def test_page_within_pixel_cap_renders_at_configured_dpi() -> None:
    llm = RecordingImageLLM()
    data = make_pdf([PageSpec(images=[make_png(300, 300, "scan")])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert [png_size(png) for png in llm.pngs] == [(1240, 1755)]
    assert (document.images_described, document.images_skipped) == (1, 0)


async def test_huge_page_renders_at_reduced_dpi_within_pixel_cap() -> None:
    llm = RecordingImageLLM()
    data = scanned_page_pdf(5000, 3000)

    document = await parse_pdf(data, llm=llm, settings=_settings(pdf_max_render_pixels=4_000_000))

    [(width, height)] = [png_size(png) for png in llm.pngs]
    assert width * height <= 4_000_000
    assert width * height > 3_000_000
    assert width > 5000 * 36 / 72
    assert width / height == pytest.approx(5 / 3, rel=0.01)
    assert (document.images_described, document.images_skipped) == (1, 0)


async def test_page_needing_less_than_the_minimum_dpi_is_skipped_and_counted() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(images=[make_png(300, 300, "scan")])])

    document = await parse_pdf(data, llm=llm, settings=_settings(pdf_max_render_pixels=100_000))

    assert llm.image_calls == []
    assert (document.images_described, document.images_skipped) == (0, 1)
    assert document.text == ""


async def test_oversized_embedded_image_is_skipped_without_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = make_pdf([PageSpec(text=LONG_TEXT, images=[make_png(300, 300), make_png(400, 400)])])
    decoded: list[tuple[int, int]] = []

    original_init: Any = pymupdf.Pixmap.__init__

    def recording_init(self: Any, *args: Any) -> None:
        original_init(self, *args)
        decoded.append((self.width, self.height))

    monkeypatch.setattr(pymupdf.Pixmap, "__init__", recording_init)
    llm = FakeLLMClient()

    document = await parse_pdf(data, llm=llm, settings=_settings(pdf_max_render_pixels=100_000))

    assert decoded == [(300, 300)]
    assert len(llm.image_calls) == 1
    assert (document.images_described, document.images_skipped) == (1, 1)


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
    assert (document.images_failed, document.images_skipped) == (0, 3)


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
    assert (document.images_failed, document.images_skipped) == (0, 1)
    assert "Anna Petrova is a surgeon" in document.text


async def test_vision_failure_skips_image() -> None:
    llm = FailingFirstImageLLM()
    data = make_pdf([PageSpec(text=LONG_TEXT, images=_distinct_images(2))])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 2
    assert document.images_described == 1
    assert (document.images_failed, document.images_skipped) == (1, 0)
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


async def test_page_render_takes_priority_over_standalone_images_under_the_cap() -> None:
    llm = FakeLLMClient()
    data = make_pdf(
        [
            PageSpec(text=LONG_TEXT, images=_distinct_images(5)),
            PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30)]),
        ]
    )

    document = await parse_pdf(data, llm=llm, settings=_settings(pdf_max_images=1))

    assert len(llm.image_calls) == 1
    assert "[Image on page 2:" in document.text
    assert "[Image on page 1:" not in document.text
    assert (document.images_described, document.images_failed, document.images_skipped) == (
        1,
        0,
        5,
    )


async def test_cap_left_after_page_renders_goes_to_images_in_page_order() -> None:
    llm = RecordingImageLLM()
    data = make_pdf(
        [
            PageSpec(
                text=LONG_TEXT, images=[make_png(200, 200, "first"), make_png(260, 200, "second")]
            ),
            PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30)]),
            PageSpec(text=LONG_TEXT, images=[make_png(320, 200, "third")]),
        ]
    )

    document = await parse_pdf(data, llm=llm, settings=_settings(pdf_max_images=2))

    assert sorted(png_size(png) for png in llm.pngs) == [(200, 200), (1240, 1755)]
    assert document.text.count("[Image on page 1:") == 1
    assert document.text.count("[Image on page 2:") == 1
    assert "[Image on page 3:" not in document.text
    assert (document.images_described, document.images_failed, document.images_skipped) == (
        2,
        0,
        2,
    )


async def test_graphical_page_over_the_cap_is_marked_as_undescribed_labels() -> None:
    llm = FakeLLMClient()
    data = make_pdf(
        [
            PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30)]),
            PageSpec(text=f"Labels. {LONG_TEXT}", drawings=[filled_shapes(30)]),
        ]
    )

    document = await parse_pdf(data, llm=llm, settings=_settings(pdf_max_images=1))

    opening = (
        "[Figure on page 2 not described: the text below includes its labels, "
        "which state no relationships]"
    )
    block = document.text.split(f"{opening}\n\n", 1)[1]
    assert block.startswith("Labels. Anna Petrova")
    assert block.endswith("dacha.\n\n[End of figure labels for page 2]")
    assert "[Figure on page 1" not in document.text
    assert "figure labels for page 1" not in document.text
    assert (document.images_described, document.images_skipped) == (1, 1)


async def test_graphical_page_too_large_to_render_is_marked() -> None:
    data = make_pdf([PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30)])])

    document = await parse_pdf(
        data, llm=FakeLLMClient(), settings=_settings(pdf_max_render_pixels=100_000)
    )

    assert document.text.startswith("[Figure on page 1 not described:")
    assert (document.images_described, document.images_skipped) == (0, 1)


async def test_graphical_page_whose_vision_call_fails_is_marked() -> None:
    data = make_pdf([PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30)])])

    document = await parse_pdf(data, llm=FailingFirstImageLLM(), settings=_settings())

    assert document.text.startswith("[Figure on page 1 not described:")
    assert "Anna Petrova is a surgeon" in document.text
    assert (document.images_described, document.images_failed) == (0, 1)


async def test_text_page_skipped_in_all_pages_mode_is_not_marked() -> None:
    data = make_pdf([PageSpec(text=LONG_TEXT)])

    document = await parse_pdf(
        data,
        llm=FakeLLMClient(),
        settings=_settings(vision_mode="all_pages", pdf_max_images=0),
    )

    assert document.text.startswith("Anna Petrova is a surgeon")
    assert "[Figure" not in document.text


async def test_marked_figure_labels_end_before_the_next_page() -> None:
    data = make_pdf(
        [
            PageSpec(text=f"Labels. {LONG_TEXT}", drawings=[filled_shapes(30)]),
            PageSpec(text=f"Prose. {LONG_TEXT}"),
        ]
    )

    document = await parse_pdf(data, llm=FakeLLMClient(), settings=_settings(pdf_max_images=0))

    end = document.text.index("[End of figure labels for page 1]")
    assert document.text.index("Labels.") < end < document.text.index("Prose.")


async def test_render_skipped_by_the_pixel_cap_frees_its_budget() -> None:
    llm = FakeLLMClient()
    data = make_pdf(
        [
            PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30)]),
            PageSpec(text=LONG_TEXT, images=[make_png(200, 200, "photo")]),
        ]
    )

    document = await parse_pdf(
        data,
        llm=llm,
        settings=_settings(pdf_max_images=1, pdf_max_render_pixels=100_000),
    )

    assert len(llm.image_calls) == 1
    assert "[Image on page 2:" in document.text
    assert (document.images_described, document.images_failed, document.images_skipped) == (
        1,
        0,
        1,
    )


async def test_image_covered_by_a_described_page_render_is_not_described_again() -> None:
    llm = FakeLLMClient()
    png = make_png(300, 300, "siblings")
    data = make_pdf(
        [
            PageSpec(text=LONG_TEXT, images=[png]),
            PageSpec(text=f"{LONG_TEXT} Again.", images=[png], drawings=[filled_shapes(30)]),
        ]
    )

    document = await parse_pdf(data, llm=llm, settings=_settings(pdf_max_images=1))

    assert len(llm.image_calls) == 1
    assert "[Image on page 2:" in document.text
    assert "[Image on page 1:" not in document.text
    assert (document.images_described, document.images_failed, document.images_skipped) == (
        1,
        0,
        0,
    )


async def test_all_pages_mode_renders_figure_pages_first() -> None:
    llm = FakeLLMClient()
    data = make_pdf(
        [
            PageSpec(text=LONG_TEXT),
            PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30)]),
        ]
    )

    document = await parse_pdf(
        data, llm=llm, settings=_settings(vision_mode="all_pages", pdf_max_images=1)
    )

    assert len(llm.image_calls) == 1
    assert "[Image on page 2:" in document.text
    assert "[Image on page 1:" not in document.text
    assert "[Figure" not in document.text
    assert (document.images_described, document.images_skipped) == (1, 1)


async def test_undescribed_figure_page_in_all_pages_mode_is_marked() -> None:
    data = make_pdf([PageSpec(text=LONG_TEXT, drawings=[filled_shapes(30)])])

    document = await parse_pdf(
        data,
        llm=FakeLLMClient(),
        settings=_settings(vision_mode="all_pages", pdf_max_images=0),
    )

    assert document.text.startswith("[Figure on page 1 not described:")
    assert document.text.endswith("[End of figure labels for page 1]")


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


def test_read_pdf_title_returns_metadata_title() -> None:
    data = set_title(make_pdf([PageSpec(text=LONG_TEXT)]), "  The Petrovs  ")

    assert read_pdf_title(data) == "The Petrovs"


def test_read_pdf_title_none_without_metadata() -> None:
    assert read_pdf_title(make_pdf([PageSpec(text=LONG_TEXT)])) is None


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"%PDF-1.7 garbage", id="corrupt"),
        pytest.param(make_pdf([PageSpec(text=LONG_TEXT)], password="x"), id="encrypted"),
    ],
)
def test_read_pdf_title_rejects_unreadable_pdf(data: bytes) -> None:
    with pytest.raises(DocumentParseError):
        read_pdf_title(data)
