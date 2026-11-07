import re
import urllib.request
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = ROOT / "data" / "books"
BOOKS = [
    (4928, "Bulfinch's Mythology", "Thomas Bulfinch"),
    (22381, "Myths and Legends of Ancient Greece and Rome", "E. M. Berens"),
    (677, "The Heroes; Or, Greek Fairy Tales for My Children", "Charles Kingsley"),
    (46963, "Astrology: How to Make and Read Your Own Horoscope", "Sepharial"),
    (70850, "Ptolemy's Tetrabiblos", "Claudius Ptolemy"),
]


def get_book_text(book_id):
    url = "https://www.gutenberg.org/cache/epub/{}/pg{}.txt".format(book_id, book_id)
    request = urllib.request.Request(url, headers={"User-Agent": "AskYourBooks/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8-sig")
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*? \*\*\*", text)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*? \*\*\*", text)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Could not locate Project Gutenberg text boundaries for #{}.".format(book_id))
    return text[start.end() : end.start()].strip()


def make_paragraphs(text, body_style):
    story = []
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(line.strip() for line in block.splitlines()).strip()
        if block:
            story.append(Paragraph(escape(block), body_style))
            story.append(Spacer(1, 7))
    return story


def add_page_number(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(letter[0] / 2, 0.45 * inch, "Page {}".format(document.page))
    canvas.restoreState()


def create_pdf(book_id, title, author, text):
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    output = BOOKS_DIR / "{}.pdf".format(re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-"))
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BookTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=25,
    )
    body_style = ParagraphStyle(
        "BookBody",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=10,
        leading=13,
        spaceAfter=2,
    )
    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.8 * inch,
        leftMargin=0.8 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=title,
        author=author,
    )
    story = [
        Spacer(1, 1.8 * inch),
        Paragraph(escape(title), title_style),
        Spacer(1, 0.3 * inch),
        Paragraph(escape(author), styles["Heading2"]),
        Spacer(1, 0.2 * inch),
        Paragraph(
            "A reflowed public-domain text from Project Gutenberg eBook #{}. "
            "Pagination belongs to this sample PDF edition.".format(book_id),
            styles["BodyText"],
        ),
        PageBreak(),
    ]
    story.extend(make_paragraphs(text, body_style))
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print("Created {}".format(output.relative_to(ROOT)))


def main():
    for book_id, title, author in BOOKS:
        create_pdf(book_id, title, author, get_book_text(book_id))


if __name__ == "__main__":
    main()
