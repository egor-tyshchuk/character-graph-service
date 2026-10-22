from pathlib import Path
from typing import Any, cast

import pymupdf

mupdf = cast(Any, pymupdf)
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
CARD_FILL = (0.93, 0.96, 0.88)
BANNER_FILL = (0.98, 0.9, 0.7)
LIQUID_FILL = (0.55, 0.8, 0.65)
ATOM_FILL = (0.45, 0.6, 0.9)
GLASS_FILL = (0.97, 0.97, 1.0)

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
    with mupdf.open() as canvas:
        page = canvas.new_page(width=DIAGRAM_WIDTH, height=DIAGRAM_HEIGHT)
        page.draw_rect(page.rect, color=None, fill=(1, 1, 1))
        page.insert_text((40, 60), "Sketching club: mentors and mentees", fontsize=26, color=INK)
        mentor = mupdf.Rect(40, 150, 300, 270)
        mentee = mupdf.Rect(500, 150, 760, 270)
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
        png: bytes = page.get_pixmap(matrix=mupdf.Matrix(DIAGRAM_ZOOM, DIAGRAM_ZOOM)).tobytes("png")
    return png


def _draw_arrow(page: Any, *, start: tuple[float, float], end: tuple[float, float]) -> None:
    page.draw_line(start, (end[0] - 18, end[1]), color=INK, width=3)
    head = [end, (end[0] - 20, end[1] - 10), (end[0] - 20, end[1] + 10)]
    page.draw_polyline(head, color=INK, fill=INK, closePath=True)


def draw_club_diagram(page: Any) -> None:
    banner = mupdf.Rect(80, 90, 515, 150)
    page.draw_rect(banner, color=INK, fill=BANNER_FILL, width=1.5, radius=0.3)
    page.insert_text((190, 131), "Chemistry club", fontsize=28, color=INK)
    student = mupdf.Rect(60, 260, 250, 470)
    teacher = mupdf.Rect(345, 260, 535, 470)
    for card, name in ((student, "Igor Smirnov"), (teacher, "Lena Ivanova")):
        _draw_person_card(page, card, name)
    _draw_curved_arrow(page, start=(student.x1, 330), end=(teacher.x0 - 2, 330))
    label = mupdf.Rect(245, 205, 350, 240)
    page.draw_rect(label, color=INK, fill=GLASS_FILL, width=1, radius=0.5)
    page.insert_text((258, 228), "student of", fontsize=16, color=INK)
    _draw_flask(page, center_x=150, top=540)
    _draw_molecule(page, center=(420, 610))


def _draw_person_card(page: Any, card: Any, name: str) -> None:
    page.draw_rect(card, color=INK, fill=CARD_FILL, width=1.5, radius=0.15)
    center_x = (card.x0 + card.x1) / 2
    page.draw_circle((center_x, card.y0 + 55), 28, color=INK, fill=BOX_FILL, width=1.5)
    shoulders = card.y0 + 150
    page.draw_bezier(
        (center_x - 55, shoulders),
        (center_x - 50, card.y0 + 85),
        (center_x + 50, card.y0 + 85),
        (center_x + 55, shoulders),
        color=INK,
        fill=BOX_FILL,
        width=1.5,
        closePath=True,
    )
    page.insert_text((card.x0 + 22, card.y1 - 28), name, fontsize=20, color=INK)


def _draw_curved_arrow(page: Any, *, start: tuple[float, float], end: tuple[float, float]) -> None:
    bend = (start[0] + 35, start[1] - 70), (end[0] - 35, end[1] - 70)
    tangent = mupdf.Point(end) - mupdf.Point(bend[1])
    direction = tangent / abs(tangent)
    normal = mupdf.Point(-direction.y, direction.x)
    base = mupdf.Point(end) - direction * 18
    page.draw_bezier(start, *bend, base, color=INK, width=3)
    head = [end, base + normal * 9, base - normal * 9]
    page.draw_polyline(head, color=INK, fill=INK, closePath=True)


def _draw_flask(page: Any, *, center_x: float, top: float) -> None:
    outline = [
        (center_x - 12, top),
        (center_x + 12, top),
        (center_x + 12, top + 50),
        (center_x + 60, top + 140),
        (center_x - 60, top + 140),
        (center_x - 12, top + 50),
    ]
    page.draw_polyline(outline, color=INK, fill=GLASS_FILL, width=2, closePath=True)
    liquid = [
        (center_x - 34, top + 92),
        (center_x + 34, top + 92),
        (center_x + 58, top + 137),
        (center_x - 58, top + 137),
    ]
    page.draw_polyline(liquid, color=None, fill=LIQUID_FILL, closePath=True)
    for index, radius in enumerate((7, 5, 9, 4)):
        page.draw_circle(
            (center_x - 20 + index * 13, top + 70 - index * 14), radius, color=INK, fill=LIQUID_FILL
        )


def _draw_molecule(page: Any, *, center: tuple[float, float]) -> None:
    atoms = [
        (center[0], center[1]),
        (center[0] - 70, center[1] - 50),
        (center[0] + 70, center[1] - 50),
        (center[0] - 60, center[1] + 65),
        (center[0] + 65, center[1] + 60),
        (center[0], center[1] - 95),
    ]
    for atom in atoms[1:]:
        page.draw_line(atoms[0], atom, color=INK, width=3)
    for index, atom in enumerate(atoms):
        page.draw_circle(atom, 22 if index == 0 else 14, color=INK, fill=ATOM_FILL, width=1.5)


def outlined_vector_page(draw: Any) -> bytes:
    with mupdf.open() as scratch:
        page = scratch.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        draw(page)
        svg = page.get_svg_image(text_as_path=True)
    with mupdf.open("svg", svg.encode()) as vector:
        data: bytes = vector.convert_to_pdf()
    return data


def build_sample_pdf() -> bytes:
    with mupdf.open() as doc:
        text_page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        text_area = mupdf.Rect(MARGIN, MARGIN, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN)
        text_page.insert_textbox(text_area, NEWSLETTER_TEXT, fontsize=11, color=INK)
        image_page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        image_width = PAGE_WIDTH - 2 * MARGIN
        image_height = image_width * DIAGRAM_HEIGHT / DIAGRAM_WIDTH
        image_area = mupdf.Rect(MARGIN, MARGIN, MARGIN + image_width, MARGIN + image_height)
        image_page.insert_image(image_area, stream=render_mentorship_diagram())
        diagram_page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        with mupdf.open("pdf", outlined_vector_page(draw_club_diagram)) as diagram:
            diagram_page.show_pdf_page(diagram_page.rect, diagram, 0)
        doc.set_metadata({"title": TITLE})
        data: bytes = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    return data


def main() -> None:
    OUTPUT.write_bytes(build_sample_pdf())
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
