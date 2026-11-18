"""PDF report for a school, project or class (reportlab). Set AITUTOR_FONT to a .ttf path to print Ge'ez text."""
from __future__ import annotations
import io
import os
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _font() -> str:
    path = os.environ.get("AITUTOR_FONT")
    if path and os.path.isfile(path):
        pdfmetrics.registerFont(TTFont("TutorFont", path))
        return "TutorFont"
    return "Helvetica"


def build_pdf(title: str, subtitle: str, kpis: list[tuple[str, str]],
              tables: list[tuple[str, list[str], list[list]]]) -> bytes:
    font = _font()
    st = getSampleStyleSheet()
    for s in ("Title", "Heading2", "Normal"):
        st[s].fontName = font
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=title)
    story = [Paragraph(escape(title), st["Title"]), Paragraph(escape(subtitle), st["Normal"]), Spacer(1, 6 * mm)]
    k = Table([[v for _, v in kpis], [l for l, _ in kpis]])
    k.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, 0), 18),
                           ("FONTSIZE", (0, 1), (-1, 1), 8), ("TEXTCOLOR", (0, 1), (-1, 1), colors.grey),
                           ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story += [k, Spacer(1, 6 * mm)]
    for name, head, rows in tables:
        story.append(Paragraph(escape(name), st["Heading2"]))
        if not rows:
            story += [Paragraph("No data.", st["Normal"]), Spacer(1, 4 * mm)]
            continue
        t = Table([head] + [[str(c) for c in r] for r in rows], repeatRows=1)
        t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 9),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f4c5c")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f3")]),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d6d4"))]))
        story += [t, Spacer(1, 5 * mm)]
    doc.build(story)
    return buf.getvalue()
