import asyncio
import math
import threading
from dataclasses import dataclass, field
from typing import Any, cast

import pymupdf
import structlog

from app.config import Settings
from app.ingestion.models import DocumentParseError, ParsedDocument
from app.llm.client import LLMClient, LLMError, LLMNotConfiguredError

mupdf = cast(Any, pymupdf)
logger = structlog.get_logger(__name__)

PAGE_SEPARATOR = "\n\n"
CMYK_COMPONENTS = 4
AXIS_TOLERANCE = 0.5
MIN_RENDER_DPI = 36
PYMUPDF_ERRORS: tuple[type[Exception], ...] = (
    RuntimeError,
    ValueError,
    mupdf.mupdf.FzErrorBase,
)
_PYMUPDF_LOCK = threading.Lock()


@dataclass
class _Page:
    number: int
    text: str
    figure: bool = False
    images: list[bytes] = field(default_factory=list)


@dataclass
class _PagePlan:
    page: _Page
    renders_whole: bool
    xrefs: list[int]


@dataclass
class _Extraction:
    title: str | None
    pages: list[_Page]
    dropped_images: int
    skipped_images: int


class _ImageQueue:
    def __init__(self, capacity: int) -> None:
        self._remaining = capacity
        self.dropped = 0
        self.unusable = 0

    def has_room(self) -> bool:
        if self._remaining > 0:
            return True
        self.dropped += 1
        return False

    def take(self) -> None:
        self._remaining -= 1

    def skip_unusable(self) -> None:
        self.unusable += 1


async def parse_pdf(data: bytes, *, llm: LLMClient, settings: Settings) -> ParsedDocument:
    extraction = await asyncio.to_thread(_extract, data, settings)
    if extraction.dropped_images:
        logger.warning(
            "pdf_image_cap_reached",
            max_images=settings.pdf_max_images,
            dropped=extraction.dropped_images,
        )
    results = await asyncio.gather(*(_describe_page(llm, page) for page in extraction.pages))
    descriptions = [[text for text in page_results if text is not None] for page_results in results]
    rendered = [
        _render_page(page, page_descriptions)
        for page, page_descriptions in zip(extraction.pages, descriptions, strict=True)
    ]
    described = sum(len(page_descriptions) for page_descriptions in descriptions)
    return ParsedDocument(
        title=extraction.title,
        text=PAGE_SEPARATOR.join(block for block in rendered if block),
        images_described=described,
        images_failed=sum(len(page_results) for page_results in results) - described,
        images_skipped=extraction.skipped_images,
    )


async def _describe_page(llm: LLMClient, page: _Page) -> list[str | None]:
    return list(await asyncio.gather(*(_describe(llm, png, page) for png in page.images)))


async def _describe(llm: LLMClient, png: bytes, page: _Page) -> str | None:
    try:
        return await llm.describe_image(png, page_context=page.text)
    except LLMNotConfiguredError:
        raise
    except LLMError as exc:
        logger.warning("pdf_image_description_failed", page=page.number, error=str(exc))
        return None


def _render_page(page: _Page, descriptions: list[str]) -> str:
    blocks = [page.text] if page.text else []
    if blocks and page.figure and not descriptions:
        blocks = [
            _undescribed_figure_marker(page.number),
            page.text,
            f"[End of figure labels for page {page.number}]",
        ]
    blocks.extend(f"[Image on page {page.number}: {text}]" for text in descriptions)
    return PAGE_SEPARATOR.join(blocks)


def _undescribed_figure_marker(page_number: int) -> str:
    return (
        f"[Figure on page {page_number} not described: the text below includes its labels, "
        "which state no relationships]"
    )


def _extract(data: bytes, settings: Settings) -> _Extraction:
    with _PYMUPDF_LOCK:
        doc = _open(data)
        try:
            queue = _ImageQueue(settings.pdf_max_images)
            seen_xrefs: set[int] = set()
            plans = [_plan_page(page, settings, queue, seen_xrefs) for page in doc]
            covered = _collect_page_renders(doc, plans, settings, queue)
            _collect_images(doc, plans, queue, covered)
            return _Extraction(
                title=_title(doc),
                pages=[plan.page for plan in plans],
                dropped_images=queue.dropped,
                skipped_images=queue.dropped + queue.unusable,
            )
        except PYMUPDF_ERRORS as exc:
            raise DocumentParseError(f"cannot read PDF: {exc}") from exc
        finally:
            doc.close()


def read_pdf_title(data: bytes) -> str | None:
    with _PYMUPDF_LOCK:
        doc = _open(data)
        try:
            return _title(doc)
        finally:
            doc.close()


def _open(data: bytes) -> Any:
    try:
        doc = mupdf.open(stream=data, filetype="pdf")
    except PYMUPDF_ERRORS as exc:
        raise DocumentParseError(f"cannot open PDF: {exc}") from exc
    if doc.needs_pass:
        doc.close()
        raise DocumentParseError("PDF is password-protected")
    return doc


def _title(doc: Any) -> str | None:
    title = str((doc.metadata or {}).get("title") or "").strip()
    return title or None


def _plan_page(
    page: Any, settings: Settings, queue: _ImageQueue, seen_xrefs: set[int]
) -> _PagePlan:
    text = str(page.get_text("text")).strip()
    result = _Page(number=page.number + 1, text=text, figure=_is_figure_page(page, text, settings))
    if result.figure or _renders_in_all_pages_mode(page, text, settings):
        embedded = [xref for xref, *_ in page.get_images(full=True)]
        seen_xrefs.update(embedded)
        return _PagePlan(page=result, renders_whole=True, xrefs=embedded)
    xrefs: list[int] = []
    for xref, _smask, width, height, *_ in page.get_images(full=True):
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        if min(width, height) < settings.pdf_min_image_side:
            continue
        if width * height > settings.pdf_max_render_pixels:
            logger.warning(
                "pdf_image_too_large",
                page=result.number,
                xref=xref,
                pixels=width * height,
                max_pixels=settings.pdf_max_render_pixels,
            )
            queue.skip_unusable()
            continue
        xrefs.append(xref)
    return _PagePlan(page=result, renders_whole=False, xrefs=xrefs)


def _collect_page_renders(
    doc: Any, plans: list[_PagePlan], settings: Settings, queue: _ImageQueue
) -> set[int]:
    renders = [plan for plan in plans if plan.renders_whole]
    covered: set[int] = set()
    for plan in sorted(renders, key=lambda plan: not plan.page.figure):
        if not queue.has_room():
            continue
        page = doc[plan.page.number - 1]
        if _collect(plan.page, queue, _page_png(page, settings, page_number=plan.page.number)):
            covered.update(plan.xrefs)
    return covered


def _collect_images(
    doc: Any, plans: list[_PagePlan], queue: _ImageQueue, covered: set[int]
) -> None:
    for plan in plans:
        if plan.renders_whole:
            continue
        for xref in plan.xrefs:
            if xref not in covered and queue.has_room():
                _collect(plan.page, queue, _image_png(doc, xref, page_number=plan.page.number))


def _collect(page: _Page, queue: _ImageQueue, png: bytes | None) -> bool:
    if png is None:
        queue.skip_unusable()
        return False
    page.images.append(png)
    queue.take()
    return True


def _page_png(page: Any, settings: Settings, *, page_number: int) -> bytes | None:
    dpi = _render_dpi(page.rect, settings)
    if dpi is None:
        logger.warning(
            "pdf_page_too_large",
            page=page_number,
            max_pixels=settings.pdf_max_render_pixels,
            min_dpi=MIN_RENDER_DPI,
        )
        return None
    png: bytes = page.get_pixmap(dpi=dpi).tobytes("png")
    return png


def _render_dpi(rect: Any, settings: Settings) -> int | None:
    cap = settings.pdf_max_render_pixels
    area_at_72_dpi = max(float(rect.width * rect.height), 1.0)
    dpi = min(settings.pdf_render_dpi, math.floor(72 * math.sqrt(cap / area_at_72_dpi)))
    while dpi >= MIN_RENDER_DPI and _rendered_pixels(rect, dpi) > cap:
        dpi -= 1
    return dpi if dpi >= MIN_RENDER_DPI else None


def _rendered_pixels(rect: Any, dpi: int) -> int:
    pixels = (rect * mupdf.Matrix(dpi / 72, dpi / 72)).irect
    return int(pixels.width * pixels.height)


def _renders_in_all_pages_mode(page: Any, text: str, settings: Settings) -> bool:
    return settings.vision_mode == "all_pages" and (bool(text) or _has_visual_content(page))


def _is_figure_page(page: Any, text: str, settings: Settings) -> bool:
    if len(text) < settings.pdf_scanned_page_min_chars:
        return _has_visual_content(page)
    return _is_graphical(page, _significant_drawings(page), settings)


def _has_visual_content(page: Any) -> bool:
    return bool(page.get_images()) or bool(page.get_drawings())


def _significant_drawings(page: Any) -> list[dict[str, Any]]:
    return [path for path in page.get_drawings() if not _is_decorative(path)]


def _is_decorative(path: dict[str, Any]) -> bool:
    return path.get("fill") is None and all(_is_frame_item(item) for item in path["items"])


def _is_frame_item(item: tuple[Any, ...]) -> bool:
    kind = item[0]
    if kind == "l":
        start, end = item[1], item[2]
        return float(min(abs(start.x - end.x), abs(start.y - end.y))) <= AXIS_TOLERANCE
    if kind == "qu":
        return bool(item[1].is_rectangular)
    return bool(kind == "re")


def _is_graphical(page: Any, drawings: list[dict[str, Any]], settings: Settings) -> bool:
    if len(drawings) < settings.pdf_drawings_min_count:
        return False
    rects = [path["rect"] for path in drawings]
    bounds = mupdf.Rect(
        min(rect.x0 for rect in rects),
        min(rect.y0 for rect in rects),
        max(rect.x1 for rect in rects),
        max(rect.y1 for rect in rects),
    )
    unrotated_page = page.rect * page.derotation_matrix
    covered = bounds & unrotated_page
    page_area = float(unrotated_page.width * unrotated_page.height)
    covered_area = float(max(covered.width, 0) * max(covered.height, 0))
    return covered_area / page_area >= settings.pdf_drawings_min_area_ratio


def _image_png(doc: Any, xref: int, *, page_number: int) -> bytes | None:
    try:
        pixmap = mupdf.Pixmap(doc, xref)
        if pixmap.n - pixmap.alpha >= CMYK_COMPONENTS:
            pixmap = mupdf.Pixmap(mupdf.csRGB, pixmap)
        png: bytes = pixmap.tobytes("png")
    except PYMUPDF_ERRORS as exc:
        logger.warning("pdf_image_skipped", page=page_number, xref=xref, error=str(exc))
        return None
    return png
