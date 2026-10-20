import asyncio
from dataclasses import dataclass, field
from typing import Any, cast

import pymupdf
import structlog

from app.config import Settings
from app.ingestion.models import DocumentParseError, ParsedDocument
from app.llm.client import LLMClient

mupdf = cast(Any, pymupdf)
logger = structlog.get_logger(__name__)

PAGE_SEPARATOR = "\n\n"
CMYK_COMPONENTS = 4


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

    def offer(self) -> bool:
        if self._remaining <= 0:
            self.dropped += 1
            return False
        self._remaining -= 1
        return True


async def parse_pdf(data: bytes, *, llm: LLMClient, settings: Settings) -> ParsedDocument:
    extraction = await asyncio.to_thread(_extract, data, settings)
    if extraction.dropped_images:
        logger.warning(
            "pdf_image_cap_reached",
            max_images=settings.pdf_max_images,
            dropped=extraction.dropped_images,
        )
    descriptions = await asyncio.gather(
        *(
            asyncio.gather(
                *(llm.describe_image(png, page_context=page.text) for png in page.images)
            )
            for page in extraction.pages
        )
    )
    rendered = [
        _render_page(page, page_descriptions)
        for page, page_descriptions in zip(extraction.pages, descriptions, strict=True)
    ]
    return ParsedDocument(
        title=extraction.title,
        text=PAGE_SEPARATOR.join(block for block in rendered if block),
        images_described=sum(len(page_descriptions) for page_descriptions in descriptions),
    )


def _render_page(page: _Page, descriptions: list[str]) -> str:
    blocks = [page.text] if page.text else []
    blocks.extend(f"[Image on page {page.number}: {text}]" for text in descriptions)
    return PAGE_SEPARATOR.join(blocks)


def _extract(data: bytes, settings: Settings) -> _Extraction:
    doc = _open(data)
    try:
        queue = _ImageQueue(settings.pdf_max_images)
        seen_xrefs: set[int] = set()
        pages = [_extract_page(doc, page, settings, queue, seen_xrefs) for page in doc]
        return _Extraction(title=_title(doc), pages=pages, dropped_images=queue.dropped)
    finally:
        doc.close()


def _open(data: bytes) -> Any:
    try:
        doc = mupdf.open(stream=data, filetype="pdf")
    except (RuntimeError, ValueError) as exc:
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
    if len(text) < settings.pdf_scanned_page_min_chars:
        if queue.offer():
            result.images.append(page.get_pixmap(dpi=settings.pdf_render_dpi).tobytes("png"))
        return result
    for xref, _smask, width, height, *_ in page.get_images(full=True):
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        if min(width, height) < settings.pdf_min_image_side:
            continue
        if queue.offer():
            result.images.append(_image_png(doc, xref))
    return result


def _image_png(doc: Any, xref: int) -> bytes:
    pixmap = mupdf.Pixmap(doc, xref)
    if pixmap.n - pixmap.alpha >= CMYK_COMPONENTS:
        pixmap = mupdf.Pixmap(mupdf.csRGB, pixmap)
    png: bytes = pixmap.tobytes("png")
    return png
