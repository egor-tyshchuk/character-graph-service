from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pymupdf

mupdf = cast(Any, pymupdf)

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
TEXT_RECT = mupdf.Rect(50, 50, 545, 400)
IMAGE_ORIGIN = (50, 420)
IMAGE_SIDE = 300


@dataclass(frozen=True)
class PageSpec:
    text: str = ""
    images: Sequence[bytes] = ()


def _image_rect(index: int) -> Any:
    x, y = (coordinate + index * 10 for coordinate in IMAGE_ORIGIN)
    return mupdf.Rect(x, y, x + IMAGE_SIDE, y + IMAGE_SIDE)


def make_png(width: int, height: int, text: str = "") -> bytes:
    with mupdf.open() as doc:
        page = doc.new_page(width=width, height=height)
        page.draw_rect(page.rect, color=(0.2, 0.4, 0.6), fill=(0.9, 0.8, 0.7))
        if text:
            page.insert_text((2, height / 2), text, fontsize=max(4, min(12, height / 3)))
        png: bytes = page.get_pixmap().tobytes("png")
    return png


def make_pdf(pages: Sequence[PageSpec], *, password: str | None = None) -> bytes:
    with mupdf.open() as doc:
        for spec in pages:
            page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            if spec.text:
                page.insert_textbox(TEXT_RECT, spec.text, fontsize=9)
            for index, png in enumerate(spec.images):
                page.insert_image(_image_rect(index), stream=png)
        if password is None:
            data: bytes = doc.tobytes()
        else:
            data = doc.tobytes(
                encryption=mupdf.PDF_ENCRYPT_AES_256, owner_pw=password, user_pw=password
            )
    return data


def set_title(pdf: bytes, title: str) -> bytes:
    with mupdf.open(stream=pdf, filetype="pdf") as doc:
        doc.set_metadata({"title": title})
        data: bytes = doc.tobytes()
    return data
