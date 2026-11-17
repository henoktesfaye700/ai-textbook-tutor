import json
import zipfile
from pathlib import Path
import pytest
from reportlab.pdfgen import canvas
from aitutor import ingest
from aitutor.importers import read_epub, read_pdf
from aitutor.library import import_folder, infer_meta
from aitutor.quiz import QuizSession, generate, save_outbox
from aitutor.retriever import BM25Index
from aitutor.tutor import Tutor

BODY = [
    ["Photosynthesis is how green plants make food using sunlight.",
     "Chlorophyll inside the chloroplasts captures light energy for the leaf."],
    ["Plants combine carbon dioxide from the air with water from the soil.",
     "The plant releases oxygen and stores extra glucose as starch."],
    ["Stomata are tiny pores on the underside of the leaf for gas exchange.",
     "Guard cells open and close the stomata during the day."],
    ["The nucleus holds the genetic material and controls the whole cell.",
     "The cell membrane decides which substances enter and leave the cell."],
    ["Mitochondria release usable energy from food inside the cell cytoplasm.",
     "Ribosomes build proteins that the cell needs for growth."],
]


def make_pdf(path: Path, first_page_no=11):
    c = canvas.Canvas(str(path))
    for i, lines in enumerate(BODY):
        c.drawString(72, 800, "Biology Grade 8")  # running header
        head = {0: "Chapter 1: Photosynthesis", 3: "Chapter 2: The cell"}.get(i)
        if head:
            c.drawString(72, 760, head)
        for j, ln in enumerate(lines):
            c.drawString(72, 700 - 20 * j, ln)
        c.drawString(300, 40, str(first_page_no + i))  # printed page number
        c.showPage()
    c.save()


def make_epub(path: Path):
    ch = lambda title, body: (f'<html xmlns:epub="http://www.idpf.org/2007/ops"><head><title>x</title></head><body>'
                              f'<h1>{title}</h1>{body}</body></html>')
    c1 = ch("Force and motion", '<p>A force can change how an object moves through space.</p>'
            '<span epub:type="pagebreak" title="5"></span>'
            '<p>Newton second law says force equals mass times acceleration always.</p>')
    c2 = ch("Density", '<p>Density is mass divided by volume for any material sample.</p>'
            '<span epub:type="pagebreak" title="9"></span>'
            '<p>An object floats when its density is lower than the density of water.</p>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("META-INF/container.xml",
                   '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf",
                   '<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   '<metadata><dc:title>Physics Basics</dc:title></metadata><manifest>'
                   '<item id="a" href="c1.xhtml"/><item id="b" href="c2.xhtml"/></manifest>'
                   '<spine><itemref idref="a"/><itemref idref="b"/></spine></package>')
        z.writestr("OEBPS/c1.xhtml", c1)
        z.writestr("OEBPS/c2.xhtml", c2)


def test_pdf_chapters_pages_and_running_header(tmp_path):
    p = tmp_path / "b.pdf"; make_pdf(p)
    book = read_pdf(p)
    assert [pg.page for pg in book.pages] == [11, 12, 13, 14, 15]
    assert book.pages[0].chapter.startswith("Chapter 1") and book.pages[4].chapter.startswith("Chapter 2")
    assert all("Biology Grade 8" not in pg.text for pg in book.pages)


def test_epub_chapters_and_pagebreaks(tmp_path):
    p = tmp_path / "p.epub"; make_epub(p)
    book = read_epub(p)
    assert book.title == "Physics Basics"
    assert {(pg.chapter, pg.page) for pg in book.pages} == {
        ("Force and motion", 0), ("Force and motion", 5), ("Density", 5), ("Density", 9)}


def test_infer_meta():
    assert infer_meta("biology_g8_en") == {"subject": "biology", "grade": 8, "lang": "en"}
    assert infer_meta("physics-grade7")["grade"] == 7 and infer_meta("random") == {}


@pytest.fixture
def library(tmp_path):
    make_pdf(tmp_path / "biology_g8_en.pdf"); make_epub(tmp_path / "physics.epub")
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / "manifest.json").write_text(json.dumps({"physics.epub": {"subject": "physics", "grade": 7}}))
    return tmp_path


def test_folder_import_metadata_and_report(library):
    ps, report = import_folder(library)
    by = {r["file"]: r for r in report}
    assert set(by) == {"biology_g8_en.pdf", "physics.epub"}
    assert by["biology_g8_en.pdf"]["subject"] == "biology" and by["physics.epub"]["grade"] == 7
    assert {p.subject for p in ps} == {"biology", "physics"}


def test_bad_file_does_not_stop_batch(library):
    (library / "broken_g8.pdf").write_bytes(b"not a pdf")
    _, report = import_folder(library)
    assert any("error" in r for r in report) and any("passages" in r for r in report)


def test_tutor_answers_from_the_right_book(library):
    ps, _ = import_folder(library)
    t = Tutor(BM25Index(ps))
    ok = t.ask("How do plants make food?", subject="biology", grade=8)
    assert ok.grounded and ok.citations[0] == "Biology Grade 8, Chapter 1: Photosynthesis, p. 11"
    assert "Chapter 1" not in ok.hint  # heading must not leak into the question
    assert not t.ask("How do plants make food?", subject="physics").grounded
    assert not t.ask("How do plants make food?", subject="biology", grade=7).grounded
    assert "p. 5" in t.ask("Newton second law", subject="physics").citations[0]


def test_dedupe_on_reimport(library):
    make_pdf(library / "biology_g8_copy.pdf")  # same content again
    ps, report = import_folder(library)
    assert sum(r.get("duplicates_dropped", 0) for r in report) > 0
    assert len({p.text for p in ps}) == len(ps)


def test_quiz_generation_and_adaptive_session(library, tmp_path):
    ps, _ = import_folder(library)
    bio = [p for p in ps if p.subject == "biology"]
    qs = generate(bio, BM25Index(bio))
    assert qs and all(len(q.options) == 4 and "_____" in q.prompt for q in qs)
    s = QuizSession(qs)
    first = s.next()
    s.answer(first, first.answer)          # right answer raises that topic's mastery
    nxt = s.next()
    assert nxt.topic != first.topic        # weakest (untouched) topic comes next
    s.answer(nxt, (nxt.answer + 1) % 4)
    box = tmp_path / "outbox.json"
    save_outbox(box, s.events, "dev-1")
    assert len(json.loads(box.read_text())["events"]) == 2


def test_passages_roundtrip(library, tmp_path):
    ps, _ = import_folder(library)
    ingest.save(ps, tmp_path / "p.jsonl")
    assert ingest.load(tmp_path / "p.jsonl") == ps
