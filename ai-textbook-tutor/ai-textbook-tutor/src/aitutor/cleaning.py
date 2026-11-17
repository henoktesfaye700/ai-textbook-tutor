"""Turn raw OCR output of a textbook into clean, citable passages."""
from __future__ import annotations
import re
import unicodedata
from .models import Passage

_PAGE_NO = re.compile(r"^\s*(page\s*)?\d{1,3}\s*$", re.I)
_GEEZ_SEP = {"\u1361": " ", "\u1362": ". ", "\u1363": ", ", "\u1364": "; ", "\u1365": ": "}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for g, repl in _GEEZ_SEP.items():
        text = text.replace(g, repl)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    lines = [ln for ln in text.splitlines() if not _PAGE_NO.match(ln)]
    text = " ".join(ln.strip() for ln in lines if ln.strip())
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk(text: str, unit: str, topic: str, page: int, lang: str = "en",
          max_words: int = 60, overlap: int = 1, subject: str = "general",
          grade: int = 0, book: str = "", uid: str = "", source: str = "",
          phys: int = 0) -> list[Passage]:
    """Sentence-aware chunking with sentence overlap so answers are not cut in half."""
    sents = split_sentences(normalize(text))
    out, i, n = [], 0, 0
    while i < len(sents):
        words, j = 0, i
        while j < len(sents) and (words + len(sents[j].split()) <= max_words or j == i):
            words += len(sents[j].split())
            j += 1
        out.append(Passage(f"{uid or book or unit}-p{page}-{n}", topic, unit, page,
                           " ".join(sents[i:j]), lang, subject, grade, book, source, phys))
        n += 1
        if j >= len(sents):
            break
        i = max(j - overlap, i + 1)
    return out
