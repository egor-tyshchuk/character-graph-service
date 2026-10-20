from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pymupdf

mupdf = cast(Any, pymupdf)

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
TEXT_RECT = mupdf.Rect(50, 50, 545, 400)
IMAGE_ORIGIN = (50, 420)
IMAGE_SIDE = 300
DRAWING_REGION = (50.0, 420.0, 545.0, 800.0)
STROKE = (0.1, 0.1, 0.1)
FILL = (0.3, 0.6, 0.9)

Region = tuple[float, float, float, float]
Drawing = Callable[[Any], None]


@dataclass(frozen=True)
class PageSpec:
    text: str = ""
    images: Sequence[bytes] = ()
    drawings: Sequence[Drawing] = ()
    rotation: int = 0


def _grid_cells(rows: int, columns: int, region: Region) -> list[Any]:
    x0, y0, x1, y1 = region
    width, height = (x1 - x0) / columns, (y1 - y0) / rows
    return [
        mupdf.Rect(
            x0 + column * width,
            y0 + row * height,
            x0 + (column + 1) * width,
            y0 + (row + 1) * height,
        )
        for row in range(rows)
        for column in range(columns)
    ]


def _inset(rect: Any, dx: float, dy: float) -> Any:
    return mupdf.Rect(rect.x0 + dx, rect.y0 + dy, rect.x1 - dx, rect.y1 - dy)


def filled_shapes(count: int, region: Region = DRAWING_REGION) -> Drawing:
    columns = 6
    rows = -(-count // columns)

    def draw(page: Any) -> None:
        for index, cell in enumerate(_grid_cells(rows, columns, region)[:count]):
            if index % 3 == 0:
                page.draw_rect(cell, color=STROKE, fill=FILL)
            elif index % 3 == 1:
                page.draw_circle(
                    (cell.tl + cell.br) / 2, min(cell.width, cell.height) / 2.5, fill=FILL
                )
            else:
                page.draw_bezier(cell.bl, cell.tl, cell.br, cell.tr, color=STROKE)

    return draw


def filled_axis_polygons(count: int, region: Region = DRAWING_REGION) -> Drawing:
    columns = 6
    rows = -(-count // columns)

    def draw(page: Any) -> None:
        for cell in _grid_cells(rows, columns, region)[:count]:
            middle = (cell.x0 + cell.x1) / 2, (cell.y0 + cell.y1) / 2
            corners = [
                cell.bl,
                cell.br,
                (cell.x1, middle[1]),
                middle,
                (middle[0], cell.y0),
                cell.tl,
            ]
            page.draw_polyline(corners, fill=FILL, closePath=True)

    return draw


def rotated_rectangles(count: int, region: Region = DRAWING_REGION) -> Drawing:
    columns = 6
    rows = -(-count // columns)

    def draw(page: Any) -> None:
        for cell in _grid_cells(rows, columns, region)[:count]:
            frame = _inset(cell, 4, 4).quad
            page.draw_quad(frame.morph((cell.tl + cell.br) / 2, mupdf.Matrix(20)), color=STROKE)

    return draw


def org_chart(region: Region = DRAWING_REGION) -> Drawing:
    def draw(page: Any) -> None:
        x0, y0, x1, y1 = region
        root = mupdf.Rect((x0 + x1) / 2 - 60, y0, (x0 + x1) / 2 + 60, y0 + 40)
        children = _grid_cells(1, 3, (x0, y1 - 40, x1, y1))
        bus_y = (root.y1 + children[0].y0) / 2
        page.draw_rect(root, color=STROKE)
        root_center = (root.x0 + root.x1) / 2
        page.draw_line((root_center, root.y1), (root_center, bus_y), color=STROKE)
        centers = [(child.x0 + child.x1) / 2 for child in children]
        page.draw_line((centers[0], bus_y), (centers[-1], bus_y), color=STROKE)
        for child, center in zip(children, centers, strict=True):
            page.draw_rect(_inset(child, 10, 0), color=STROKE)
            page.draw_line((center, bus_y), (center, child.y0), color=STROKE)

    return draw


def table_grid(rows: int, columns: int, region: Region = DRAWING_REGION) -> Drawing:
    def draw(page: Any) -> None:
        for cell in _grid_cells(rows, columns, region):
            page.draw_rect(cell, color=STROKE)
        x0, y0, x1, y1 = region
        for row in range(rows + 1):
            y = y0 + row * (y1 - y0) / rows
            page.draw_line((x0, y), (x1, y), color=STROKE)
        for column in range(columns + 1):
            x = x0 + column * (x1 - x0) / columns
            page.draw_line((x, y0), (x, y1), color=STROKE)

    return draw


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
            for drawing in spec.drawings:
                drawing(page)
            page.set_rotation(spec.rotation)
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
