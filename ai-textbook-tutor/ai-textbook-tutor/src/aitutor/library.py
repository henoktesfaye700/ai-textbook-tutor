"""Import a folder of books, each tagged with subject, grade and language."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from .cleaning import chunk, normalize
from .importers import read_book
from .models import Passage

FILENAME = re.compile(
    r"^(?P<subject>[a-z]+)[_\- ]g(?:rade)?[_\- ]?(?P<grade>\d{1,2})(?:[_\- ](?P<lang>[a-z]{2}))?$", re.I)


def infer_meta(stem: str) -> dict:
    """biology_g8_en -> {subject: biology, grade: 8, lang: en}"""
    m = FILENAME.match(stem)
    if not m:
        return {}
    d = m.groupdict()
    return {"subject": d["subject"].lower(), "grade": int(d["grade"]), **({"lang": d["lang"].lower()} if d["lang"] else {})}


def import_book(path: Path, subject: str = "general", grade: int = 0, lang: str = "en",
                title: str | None = None) -> list[Passage]:
    book = read_book(path)
    title = title or book.title
    slug = re.sub(r"\W+", "-", path.stem.lower()).strip("-")
    out: list[Passage] = []
    for i, rp in enumerate(book.pages):
        if len(rp.text.split()) < 5:
            continue  # blank, cover or navigation page
        topic = " ".join(normalize(rp.chapter).split()) or title
        out += chunk(rp.text, unit=title, topic=topic, page=rp.page, lang=lang,
                     subject=subject, grade=grade, book=slug, uid=f"{slug}-{i}",
                     source=path.name, phys=rp.phys)
    return out


def import_folder(folder: Path) -> tuple[list[Passage], list[dict]]:
    """Per-book metadata: manifest.json entry > filename pattern > defaults."""
    mpath = folder / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    passages: list[Passage] = []
    report: list[dict] = []
    seen: set[str] = set()
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in (".pdf", ".epub"):
            continue
        inferred = infer_meta(f.stem)
        if inferred:  # readable title from the filename, e.g. "Biology Grade 8"
            inferred["title"] = f"{inferred['subject'].title()} Grade {inferred['grade']}"
        meta = {"subject": "general", "grade": 0, "lang": "en", **inferred, **manifest.get(f.name, {})}
        try:
            ps = import_book(f, meta["subject"], int(meta["grade"]), meta["lang"], meta.get("title"))
        except Exception as e:  # one bad file must not stop the batch
            report.append({"file": f.name, "error": f"{type(e).__name__}: {e}"})
            continue
        unique = []
        for p in ps:  # drop exact duplicate passages (repeated pages, re-imported books)
            h = hashlib.sha1(re.sub(r"\W+", "", p.text.lower()).encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(p)
        passages += unique
        report.append({"file": f.name, **{k: meta[k] for k in ("subject", "grade", "lang")},
                       "passages": len(unique), "duplicates_dropped": len(ps) - len(unique),
                       "chapters": len({p.topic for p in unique})})
    return passages, report
