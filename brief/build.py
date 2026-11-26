"""Build the two-page Track 2 brief from its Markdown sources."""

import argparse
from html import escape
from pathlib import Path
import re

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepInFrame, PageBreak, Paragraph, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MARGIN = 44
INK = colors.HexColor("#14232c")
TEAL = colors.HexColor("#12685c")
MUTED = colors.HexColor("#475d67")
STYLES = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=27, leading=31, textColor=INK, spaceAfter=18),
    "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=TEAL, spaceBefore=10, spaceAfter=6),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15.5, textColor=INK, spaceAfter=9),
}


def inline(markdown):
    value = escape(markdown.replace("–", "-").replace("—", " - "))
    value = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", value)
    return re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)


def flowables(source, width):
    story = []
    blocks = re.split(r"\n\s*\n", source.read_text(encoding="utf-8").strip())
    for block in blocks:
        rows = block.splitlines()
        if rows[0].startswith("#"):
            level, title = rows[0].split(maxsplit=1)
            story.append(Paragraph(inline(title), STYLES["title" if level == "#" else "heading"]))
            rows = rows[1:]
        if not rows:
            continue
        figure = re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", rows[0])
        if figure:
            path = str(source.parent / figure[1])
            image_width, image_height = ImageReader(path).getSize()
            graphic = Image(path, width=width, height=width * image_height / image_width)
            story.extend((graphic, Spacer(1, 7)))
            rows = rows[1:]
        if rows and all(row.startswith(("- ", "* ")) for row in rows):
            story.extend(Paragraph(inline(row[2:]), STYLES["body"], bulletText="•") for row in rows)
        elif rows:
            story.append(Paragraph(inline(" ".join(rows)), STYLES["body"]))
    return story


def margins(canvas, document):
    width, height = document.pagesize
    canvas.setFillColor(TEAL)
    canvas.rect(MARGIN, height - 47, width - 2 * MARGIN, 2, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(MARGIN, height - 36, "AIRAVAT  /  DEVFORGE TRACK 2")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN, 28, "Multiverse of Monsoon")
    canvas.drawRightString(width - MARGIN, 28, f"{document.page} / 2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page1", type=Path, default=ROOT / "train" / "brief_page1.md")
    parser.add_argument("--output", type=Path, default=HERE / "brief.pdf")
    args = parser.parse_args()
    if not args.page1.is_file():
        parser.error(f"Brief page 1 is missing: {args.page1}")

    rl_config.useA85 = False
    document = SimpleDocTemplate(
        str(args.output), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=62, bottomMargin=47, title="Airavat - Multiverse of Monsoon",
        author="Team Airavat",
    )
    pages = [KeepInFrame(document.width - 12, document.height - 12, flowables(source, document.width - 12))
             for source in (args.page1, HERE / "page2.md")]
    document.build([pages[0], PageBreak(), pages[1]], onFirstPage=margins, onLaterPages=margins)


if __name__ == "__main__":
    main()
