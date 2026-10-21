from pathlib import Path
from typing import Any

import pymupdf

OUTPUT = Path(__file__).with_name("sample.pdf")
TITLE = "Linden Street Bakery Newsletter"
PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN = 56
INK = (0.1, 0.1, 0.1)
BOX_FILL = (0.86, 0.92, 0.98)
DIAGRAM_WIDTH = 800
DIAGRAM_HEIGHT = 420
DIAGRAM_ZOOM = 2

NEWSLETTER_TEXT = """\
Linden Street Bakery Newsletter, November 2026

The painted window on Linden Street is finished. Mark Petrov, an architect at the studio across \
the river, drew the design: a large spaniel holding a loaf of rye bread. Lena Ivanova, who \
teaches chemistry at the secondary school on the hill, brought eight of her students to help \
with the paint.

The neighbourhood council liked the window so much that it started a small sketching club for \
local residents. The club meets on Thursday evenings in the back room of the bakery. Coffee and \
the last rolls of the day are free for members.

Next month the club plans to paint the window of the shoe repair shop. Everyone is welcome; \
bring a pencil and a warm jacket."""


def render_mentorship_diagram() -> bytes:
    with pymupdf.open() as canvas:
        page = canvas.new_page(width=DIAGRAM_WIDTH, height=DIAGRAM_HEIGHT)
        page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
        page.insert_text((40, 60), "Sketching club: mentors and mentees", fontsize=26, color=INK)
        mentor = pymupdf.Rect(40, 150, 300, 270)
        mentee = pymupdf.Rect(500, 150, 760, 270)
        for box, name, role in (
            (mentor, "Mark Petrov", "architect"),
            (mentee, "Lena Ivanova", "chemistry teacher"),
        ):
            page.draw_rect(box, color=INK, fill=BOX_FILL, width=2)
            page.insert_text((box.x0 + 20, box.y0 + 50), name, fontsize=24, color=INK)
            page.insert_text((box.x0 + 20, box.y0 + 90), role, fontsize=18, color=INK)
        _draw_arrow(page, start=(mentor.x1, 210), end=(mentee.x0, 210))
        page.insert_text((330, 195), "mentor of", fontsize=22, color=INK)
        page.insert_text(
            (40, 350),
            "Mark Petrov is the mentor of Lena Ivanova in the sketching club.",
            fontsize=18,
            color=INK,
        )
        png: bytes = page.get_pixmap(matrix=pymupdf.Matrix(DIAGRAM_ZOOM, DIAGRAM_ZOOM)).tobytes(
            "png"
        )
    return png


def _draw_arrow(page: Any, *, start: tuple[float, float], end: tuple[float, float]) -> None:
    page.draw_line(start, (end[0] - 18, end[1]), color=INK, width=3)
    head = [end, (end[0] - 20, end[1] - 10), (end[0] - 20, end[1] + 10)]
    page.draw_polyline(head, color=INK, fill=INK, closePath=True)


def build_sample_pdf() -> bytes:
    with pymupdf.open() as doc:
        text_page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        text_area = pymupdf.Rect(MARGIN, MARGIN, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN)
        text_page.insert_textbox(text_area, NEWSLETTER_TEXT, fontsize=11, color=INK)
        image_page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        image_width = PAGE_WIDTH - 2 * MARGIN
        image_height = image_width * DIAGRAM_HEIGHT / DIAGRAM_WIDTH
        image_area = pymupdf.Rect(MARGIN, MARGIN, MARGIN + image_width, MARGIN + image_height)
        image_page.insert_image(image_area, stream=render_mentorship_diagram())
        doc.set_metadata({"title": TITLE})
        data: bytes = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    return data


def main() -> None:
    OUTPUT.write_bytes(build_sample_pdf())
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
