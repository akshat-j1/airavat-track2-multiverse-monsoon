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
from reportlab.platypus import Image, KeepInFrame, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MARGIN = 41
INK = colors.HexColor("#14232c")
TEAL = colors.HexColor("#12685c")
MUTED = colors.HexColor("#475d67")
STYLES = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26, leading=30, textColor=INK, spaceAfter=12),
    "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=TEAL,
                              spaceBefore=9, spaceAfter=5),
    "body": ParagraphStyle("body", fontSize=10.5, leading=14.8, textColor=INK, spaceAfter=9),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=9.6, leading=13,
                              textColor=MUTED, spaceAfter=9),
    "box": ParagraphStyle("box", fontSize=10.2, leading=14.2, textColor=INK),
}


def inline(markdown):
    value = escape(markdown.replace("–", "-").replace("—", " - "))
    value = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#12685c">\1</link>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", value)
    return re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)


def skill_table(rows, width):
    data = [[cell.strip() for cell in row.strip("|").split("|")] for row in (rows[0], *rows[2:])]
    styles = [
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf4f3"), colors.white]),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#bfd1cf")),
    ]
    lead_width = 73
    return Table(data, colWidths=[lead_width, *[(width - lead_width) / (len(data[0]) - 1)] * (len(data[0]) - 1)],
                 style=TableStyle(styles), spaceAfter=5)


def callout(rows, width):
    paragraphs = [Paragraph(inline(row[2:]), STYLES["box"]) for row in rows]
    contents = [flow for paragraph in paragraphs for flow in (paragraph, Spacer(1, 3))][:-1]
    styles = [
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#e9f2f0")),
        ("LINEBEFORE", (0, 0), (0, 0), 2, TEAL),
        ("LEFTPADDING", (0, 0), (0, 0), 12),
        ("RIGHTPADDING", (0, 0), (0, 0), 12),
        ("TOPPADDING", (0, 0), (0, 0), 9),
        ("BOTTOMPADDING", (0, 0), (0, 0), 9),
    ]
    return Table([[contents]], colWidths=[width], style=TableStyle(styles), spaceAfter=9)


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
        if all(row.startswith("|") for row in rows):
            story.append(skill_table(rows, width))
            continue
        if all(row.startswith("> ") for row in rows):
            story.append(callout(rows, width))
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
            text = " ".join(rows)
            if re.fullmatch(r"\*[^*]+\*", text):
                story.append(Paragraph(inline(text[1:-1]), STYLES["caption"]))
            else:
                story.append(Paragraph(inline(text), STYLES["body"]))
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
    pages = [KeepInFrame(document.width - 12, document.height - 12, flowables(source, document.width - 12), mode="error")
             for source in (args.page1, HERE / "page2.md")]
    document.build([pages[0], PageBreak(), pages[1]], onFirstPage=margins, onLaterPages=margins)


if __name__ == "__main__":
    main()
