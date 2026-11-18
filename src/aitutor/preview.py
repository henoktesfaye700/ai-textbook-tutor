"""Render one PDF page to a PNG with the question's words highlighted.
Needs the optional packages pypdfium2 and pillow."""
from __future__ import annotations
import io
import re
from functools import lru_cache
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from .retriever import STOP, stem


def search_terms(query: str) -> list[str]:
    words = re.findall(r"[\w\u1200-\u137F]{3,}", query.lower())
    return sorted({stem(w) for w in words if w not in STOP})[:8]  # stems also match plurals


@lru_cache(maxsize=64)
def render_page(path: Path, page_no: int, query: str = "", scale: float = 1.3) -> bytes:
    pdf = pdfium.PdfDocument(str(path))
    try:
        page = pdf[page_no - 1]
        img = page.render(scale=scale).to_pil().convert("RGBA")
        height, text = page.get_height(), page.get_textpage()
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for term in search_terms(query):
            found = text.search(term)
            for _ in range(25):  # cap highlights per word
                hit = found.get_next()
                if not hit:
                    break
                boxes = [text.get_charbox(i) for i in range(hit[0], hit[0] + hit[1])]
                left, bottom = min(b[0] for b in boxes), min(b[1] for b in boxes)
                right, top = max(b[2] for b in boxes), max(b[3] for b in boxes)
                draw.rectangle([left * scale, (height - top) * scale, right * scale, (height - bottom) * scale],
                               fill=(255, 214, 0, 110))
        out = io.BytesIO()
        Image.alpha_composite(img, overlay).convert("RGB").save(out, "PNG")
        return out.getvalue()
    finally:
        pdf.close()
