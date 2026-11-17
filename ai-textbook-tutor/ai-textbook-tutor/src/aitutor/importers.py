"""Read text-based PDFs and EPUBs into pages with chapter + page-number metadata."""
from __future__ import annotations
import posixpath
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

HEADING = re.compile(r"^\s*(unit|chapter|lesson|section|module|topic)\s+(\d+|[ivxl]+)\b[\s:.\-\u2013]*(.*)$", re.I)
PAGE_ONLY = re.compile(r"^\s*(?:page\s*)?(\d{1,4})\s*$", re.I)


@dataclass
class RawPage:
    page: int      # printed page number; 0 when unknown (e.g. EPUB without page-breaks)
    chapter: str
    text: str
    phys: int = 0  # physical page index in the PDF, 1-based (0 for EPUB)


@dataclass
class Book:
    title: str
    pages: list[RawPage]


def pretty(stem: str) -> str:
    return re.sub(r"[_\-]+", " ", stem).strip().title()


# ---------------------------------------------------------------- PDF
def _outline_map(reader) -> dict[int, str]:
    out: dict[int, str] = {}

    def walk(items):
        for it in items:
            if isinstance(it, list):
                walk(it)
            else:
                try:
                    out.setdefault(reader.get_destination_page_number(it), str(it.title))
                except Exception:
                    pass
    try:
        walk(reader.outline)
    except Exception:
        pass
    return out


def read_pdf(path: Path) -> Book:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    lines = [(p.extract_text() or "").splitlines() for p in reader.pages]
    n = len(lines)
    counts: Counter = Counter()
    for ls in lines:  # running headers/footers repeat on many pages
        counts.update({l.strip() for l in ls if l.strip() and not PAGE_ONLY.match(l) and len(l.strip()) < 80})
    common = {l for l, c in counts.items() if n >= 4 and c >= max(3, 0.4 * n)}
    outline = _outline_map(reader)
    chapter, pages = "", []
    for i, ls in enumerate(lines):
        nonempty = [l.strip() for l in ls if l.strip()]
        printed = 0
        for cand in nonempty[-1:] + nonempty[:1]:  # page number sits in footer or header
            m = PAGE_ONLY.match(cand)
            if m:
                printed = int(m.group(1))
                break
        kept = []
        for l in ls:
            s = l.strip()
            if s in common:
                continue
            if not outline and len(s) <= 80 and not s.endswith(".") and HEADING.match(s):
                chapter = s
                continue  # the heading is metadata, not body text
            kept.append(l)
        if i in outline:
            chapter = outline[i]
        pages.append(RawPage(printed or i + 1, chapter, "\n".join(kept), i + 1))
    return Book(pretty(path.stem), pages)


# ---------------------------------------------------------------- EPUB
class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "section", "article", "blockquote",
             "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.title: str | None = None
        self.head_title = ""
        self._skip, self._cap, self._in_title = 0, None, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style"):
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if a.get("epub:type") == "pagebreak" or a.get("role") == "doc-pagebreak":
            label = (a.get("title") or a.get("aria-label") or "").strip()
            if label.isdigit():
                self.parts.append(f"\n\x00PB{label}\x00\n")
        if tag in ("h1", "h2", "h3") and self.title is None:
            self._cap = []
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        if tag == "title":
            self._in_title = False
        if tag in ("h1", "h2", "h3") and self._cap is not None:
            self.title = " ".join("".join(self._cap).split()) or None
            self._cap = None
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, d):
        if self._skip:
            return
        if self._in_title:
            self.head_title += d
            return
        if self._cap is not None:
            self._cap.append(d)  # chapter title: stored as metadata, kept out of the body
            return
        self.parts.append(d)


def read_epub(path: Path) -> Book:
    pages: list[RawPage] = []
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = root.find(".//{*}rootfile").get("full-path")
        opf = ET.fromstring(z.read(opf_path))
        base = posixpath.dirname(opf_path)
        manifest = {i.get("id"): i.get("href") for i in opf.findall(".//{*}manifest/{*}item")}
        spine = [manifest[r.get("idref")] for r in opf.findall(".//{*}spine/{*}itemref")
                 if r.get("idref") in manifest]
        title = (opf.findtext(".//{*}title") or "").strip() or pretty(path.stem)
        last = 0
        for href in spine:
            name = posixpath.normpath(posixpath.join(base, unquote(href).split("#")[0]))
            parser = _Text()
            parser.feed(z.read(name).decode("utf-8", "ignore"))
            chapter = parser.title or " ".join(parser.head_title.split()) or Path(href).stem
            segs = re.split(r"\x00PB(\d+)\x00", "".join(parser.parts))
            chunks, page = [(last, segs[0])], last
            for j in range(1, len(segs), 2):
                page = int(segs[j])
                chunks.append((page, segs[j + 1]))
            last = page
            pages += [RawPage(pg, chapter, t) for pg, t in chunks]
    return Book(title, pages)


def read_book(path: Path) -> Book:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".epub":
        return read_epub(path)
    raise ValueError(f"unsupported file type: {ext}")
