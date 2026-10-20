import asyncio
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
    images: list[bytes] = field(default_factory=list)


@dataclass
class _Extraction:
    title: str | None
    pages: list[_Page]
    dropped_images: int


class _ImageQueue:
    def __init__(self, capacity: int) -> None:
        self._remaining = capacity
        self.dropped = 0

    def has_room(self) -> bool:
        if self._remaining > 0:
            return True
        self.dropped += 1
        return False

    def take(self) -> None:
        self._remaining -= 1


async def parse_pdf(data: bytes, *, llm: LLMClient, settings: Settings) -> ParsedDocument:
    extraction = await asyncio.to_thread(_extract, data, settings)
    if extraction.dropped_images:
        logger.warning(
            "pdf_image_cap_reached",
            max_images=settings.pdf_max_images,
            dropped=extraction.dropped_images,
        )
    descriptions = await asyncio.gather(*(_describe_page(llm, page) for page in extraction.pages))
    rendered = [
        _render_page(page, page_descriptions)
        for page, page_descriptions in zip(extraction.pages, descriptions, strict=True)
    ]
    return ParsedDocument(
        title=extraction.title,
        text=PAGE_SEPARATOR.join(block for block in rendered if block),
        images_described=sum(len(page_descriptions) for page_descriptions in descriptions),
    )


async def _describe_page(llm: LLMClient, page: _Page) -> list[str]:
    results = await asyncio.gather(*(_describe(llm, png, page) for png in page.images))
    return [description for description in results if description is not None]


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
    blocks.extend(f"[Image on page {page.number}: {text}]" for text in descriptions)
    return PAGE_SEPARATOR.join(blocks)


def _extract(data: bytes, settings: Settings) -> _Extraction:
    with _PYMUPDF_LOCK:
        doc = _open(data)
        try:
            queue = _ImageQueue(settings.pdf_max_images)
            seen_xrefs: set[int] = set()
            pages = [_extract_page(doc, page, settings, queue, seen_xrefs) for page in doc]
            return _Extraction(title=_title(doc), pages=pages, dropped_images=queue.dropped)
        except PYMUPDF_ERRORS as exc:
            raise DocumentParseError(f"cannot read PDF: {exc}") from exc
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


def _extract_page(
    doc: Any, page: Any, settings: Settings, queue: _ImageQueue, seen_xrefs: set[int]
) -> _Page:
    text = str(page.get_text("text")).strip()
    result = _Page(number=page.number + 1, text=text)
    if _renders_whole_page(page, text, settings):
        seen_xrefs.update(xref for xref, *_ in page.get_images(full=True))
        if queue.has_room():
            result.images.append(page.get_pixmap(dpi=settings.pdf_render_dpi).tobytes("png"))
            queue.take()
        return result
    for xref, _smask, width, height, *_ in page.get_images(full=True):
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        if min(width, height) < settings.pdf_min_image_side or not queue.has_room():
            continue
        png = _image_png(doc, xref, page_number=result.number)
        if png is not None:
            result.images.append(png)
            queue.take()
    return result


def _renders_whole_page(page: Any, text: str, settings: Settings) -> bool:
    if settings.vision_mode == "all_pages":
        return bool(text) or _has_visual_content(page)
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
